# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }

import numpy as np
import torch
import torch.nn.functional as F
from argparse import ArgumentParser

from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import tensor2numpy, count_parameters

from backbone.vit_prompts import L2P as Prompt
from backbone.vit_zoo import ViTZoo
# from backbone.vit_zoo_bak import ViTZoo

EPSILON = 1e-8

def add_special_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--prompt_mode', type=str, default=None, help="prompt mode should be in ['prefix-tune, prompt-tune']")
    parser.add_argument('--prompt_pool', type=int, default=None, help='size of prompt pool')
    parser.add_argument('--prompt_length', type=int, default=None, help='length of prompt')
    parser.add_argument('--topk', type=int, default=None, help='topk, default 5')
    parser.add_argument('--layer_ids', nargs='+', type=int, default=None, help='layers to apply prompt, e.t. [0, 1, 2, 3]')
    return parser

class L2P(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._prompt_mode = config.prompt_mode
        self._prompt_pool = config.prompt_pool
        self._prompt_length = config.prompt_length
        self._topk = config.topk
        self._layer_ids = config.layer_ids

        logger.info('Applying L2P (a class incremental method, test with {})'.format(self._incre_type))
    
    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = ViTZoo(self._logger, backbone_type=self._config.backbone, 
                pretrained=self._config.pretrained, pretrain_path=self._config.pretrain_path)
            prompt_module = Prompt(self._network.embed_dim, self._config.nb_tasks, prompt_pool=self._prompt_pool,
                                   prompt_length=self._prompt_length, prompt_mode=self._prompt_mode, top_k=self._topk,
                                   layer_ids=self._layer_ids, key_dim=self._network.feat_dim)
            self._network.set_prompt_module(prompt_module)
        
        self._network.update_fc(self._total_classes)
        
        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            if checkpoint['memory_class_means'] is not None and self._memory_bank is not None:
                self._memory_bank.set_class_means(checkpoint['memory_class_means'])
            self._logger.info("Loaded checkpoint model's state_dict !")
        if self._config.freeze_fe:
            self._network.freeze_FE()

        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        self._logger.info("Prompt's trainable params: {}".format(count_parameters(self._network.prompt, True)))
        self._network = self._network.cuda()

        for name, param in self._network.named_parameters():
            if param.requires_grad:
                self._logger.info('{} requre grad!'.format(name))

    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        losses = 0.
        ce_losses, prompt_losses = 0., 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, features, prompt_loss = model(inputs, train=True)
            
            loss = prompt_loss.sum()
            prompt_losses += loss.item()

            # ce with heuristic
            logits[:,:task_begin] = -float('inf')
            ce_loss = F.cross_entropy(logits, targets)
            loss += ce_loss
            ce_losses += ce_loss.item()

            preds = torch.max(logits[:,:task_end], dim=1)[1]
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        
        if scheduler != None:
            scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader), 'Loss_ce', ce_losses/len(train_loader), 'Loss_prompt', prompt_losses/len(train_loader)]
        return model, train_acc, train_loss
