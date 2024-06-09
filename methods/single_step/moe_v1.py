import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import optim
from argparse import ArgumentParser

from backbone.cnn_moe_net import CNN_MoE_Net_V1
from methods.single_step.finetune_normal import Finetune_normal
from utils.toolkit import count_parameters, tensor2numpy


EPSILON = 1e-8

def add_special_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--layer_names', nargs='+', type=str, default=None, help='layers to apply prompt, e.t. [layer1, layer2]')
    parser.add_argument('--expert_kernel_num', type=int, default=None, help='expert kernel num for MoE v1')
    parser.add_argument('--expert_per_task', type=int, default=None, help='expert per task for MoE v1')
    parser.add_argument('--topk', type=int, default=None, help='topk for MoE v1')
    return parser

class MoE_v1(Finetune_normal):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._mode = self._config.mode.split('|') if self._config.mode is not None else []
        self._layer_names = config.layer_names
        self._expert_kernel_num = config.expert_kernel_num
        self.expert_per_task = config.expert_per_task
        self.topk = config.topk

    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = CNN_MoE_Net_V1(self._logger, self._config.backbone, self._config.pretrained,
                    pretrain_path=self._config.pretrain_path, layer_names=self._layer_names, mode=self._mode,
                    expert_kernel_num=self._expert_kernel_num, expert_per_task=self.expert_per_task, topk=self.topk)
        self._network.update_fc(self._total_classes)

        self._network.freeze_FE()
        self._network = self._network.cuda()

        self._logger.info('Initializing task-{} MoEs in network...'.format(self._cur_task))
        with torch.no_grad():
            self._network.eval()
            self._network(torch.rand(1, 3, self._config.img_size, self._config.img_size).cuda())

        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            self._logger.info("Loaded checkpoint model's state_dict !")
        
        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        all_params, all_trainable_params = 0, 0
        for layer_id in self._config.layer_names:
            moe_id = layer_id.replace('.', '_')+'_MoEs'
            if hasattr(self._network, moe_id):
                adapter_module = getattr(self._network, moe_id)
                
                layer_params = count_parameters(adapter_module)
                layer_trainable_params = count_parameters(adapter_module, True)
                self._logger.info('{} params: {} , trainable params: {}'.format(moe_id,
                    layer_params, layer_trainable_params))
                
                all_params += layer_params
                all_trainable_params += layer_trainable_params
        self._logger.info('all MoEs params: {} , trainable params: {}'.format(all_params, all_trainable_params))
        
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                self._logger.info('{} requre grad!'.format(name))

    
    