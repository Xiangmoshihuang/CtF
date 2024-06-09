import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from argparse import ArgumentParser
from torchvision import datasets, transforms
from methods.multi_steps.finetune_il import Finetune_IL

from torch.cuda.amp import autocast, GradScaler

EPSILON = 1e-8

def add_special_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--T', type=float, default=None, help='tempreture apply to the output logits befor softmax')
    parser.add_argument('--lambda_fkd', type=float, default=None)
    parser.add_argument('--lambda_proto', type=float, default=None)
    parser.add_argument('--ssre_mode', type=str, default=None)
    return parser

class SSRE(Finetune_IL):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._protos = []
        self._old_network = None
        self._T = config.T
        self._lambda_fkd = config.lambda_fkd
        self._lambda_proto = config.lambda_proto
        
        logger.info('Applying SSRE (non-mem method, test with {})'.format(self._incre_type))

    def after_task(self):
        super().after_task()
        self._old_network = self._network.copy().freeze()
        # if hasattr(self._old_network,"module"):
        #     self.old_network_module_ptr = self._old_network.module
        # else:
        #     self.old_network_module_ptr = self._old_network
        self._logger.info("Model Compression!")
        
        self._network_compression()
        # self.save_checkpoint("{}_{}_{}".format(self.args["model_name"],self.args["init_cls"],self.args["increment"]))

    def prepare_task_data(self, data_manager):
        self.data_manager = data_manager
        if self._cur_task == -1:
            self._batch_size = 64
        else:
            self._batch_size = self._config.batch_size
        if self._cur_task == -1:
            data_manager.mode = 'ssre'
        super().prepare_task_data(data_manager)


    def prepare_model(self, checkpoint=None):
        super().prepare_model(checkpoint)
        if self._old_network is not None:
            self._old_network.cuda()
        self._logger.info("Model Expansion!")
        self._network_expansion()
        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        
    def incremental_train(self):
        super().incremental_train()
        self._network = self._network.cuda()
        self._build_protos()
        self._logger.info("Model Compression!")
        self._network_compression()

    def train(self):
        if self._cur_task > 0:
            self._network.eval()
            return
        self._network.train()

    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        self.train()
        scaler = GradScaler()
        losses = 0.
        losses_clf, losses_fkd, losses_proto = 0., 0., 0.
        correct, total = 0, 0

        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            with autocast():
                logits, loss_clf, loss_fkd, loss_proto = self._compute_ssre_loss(model, inputs, targets)
                loss = loss_clf + loss_fkd + loss_proto

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()
            losses += loss.item()
            losses_clf += loss_clf.item()
            losses_fkd += loss_fkd.item()
            losses_proto += loss_proto.item()
            _, preds = torch.max(logits, dim=1)
            correct += preds.eq(targets.expand_as(preds)).cpu().sum()
            total += len(targets)
        if scheduler != None:
            scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader), 'Loss_clf', losses_clf/len(train_loader), 'Loss_fkd', 
                        losses_fkd/len(train_loader),'Loss_proto', losses_proto/len(train_loader)]
        return model, train_acc, train_loss

    def _compute_ssre_loss(self, model, inputs, targets):
        if self._cur_task == 0:
            logits = model(inputs)[0]
            loss_clf = F.cross_entropy(logits/self._T, targets)
            return logits, loss_clf, torch.tensor(0.), torch.tensor(0.)
        
        features = model.feature_extractor(inputs) # N D
        
        with torch.no_grad():
            features_old = self._old_network.feature_extractor(inputs)
                    
        protos = torch.from_numpy(np.array(self._protos)).cuda() # C D
        with torch.no_grad():
            weights = F.normalize(features,p=2,dim=1,eps=1e-12) @ F.normalize(protos,p=2,dim=1,eps=1e-12).T
            weights = torch.max(weights,dim=1)[0]
            # mask = weights > self.args["threshold"]
            mask = weights
        logits = model(inputs)[0]
        loss_clf = F.cross_entropy(logits/self._T, targets, reduction="none")
        # loss_clf = torch.mean(loss_clf * ~mask)
        loss_clf =  torch.mean(loss_clf * (1-mask))
        
        loss_fkd = torch.norm(features - features_old, p=2, dim=1)
        loss_fkd = self._lambda_fkd * torch.sum(loss_fkd * mask)
        
        index = np.random.choice(range(self._known_classes),size=self._batch_size,replace=True)
        
        proto_features = np.array(self._protos)[index]
        proto_targets = index
        proto_features = proto_features
        proto_features = torch.from_numpy(proto_features).float().cuda()
        proto_targets = torch.from_numpy(proto_targets).cuda()
        
        
        proto_logits = model.fc(proto_features)
        loss_proto = self._lambda_proto * F.cross_entropy(proto_logits/self._T, proto_targets)
        return logits, loss_clf, loss_fkd, loss_proto
    
    def _network_expansion(self):
        if self._cur_task > 0:
            for p in self._network.feature_extractor.parameters():
                p.requires_grad = True
            for k, v in self._network.feature_extractor.named_parameters():
                if 'adapter' not in k:
                    v.requires_grad = False 
        # self._network.convnet.re_init_params() # do not use!
        self._network.feature_extractor.switch("parallel_adapters")       
        
    def _network_compression(self):
        
        model_dict = self._network.state_dict()
        for k, v in model_dict.items():
            if 'adapter' in k:
                k_conv3 = k.replace('adapter', 'conv')
                if 'weight' in k:
                    model_dict[k_conv3] = model_dict[k_conv3] + F.pad(v, [1, 1, 1, 1], 'constant', 0)
                    model_dict[k] = torch.zeros_like(v)
                elif 'bias' in k:
                    model_dict[k_conv3] = model_dict[k_conv3] + v
                    model_dict[k] = torch.zeros_like(v)
                else:
                    assert 0
        self._network.load_state_dict(model_dict)
        self._network.feature_extractor.switch("normal")
    
    def _build_protos(self):
        self._logger.info("Builiding protos......")
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                idx_dataset = self.data_manager.get_dataset(indices=np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=False)
                idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                self._protos.append(class_mean)
    
    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_features(_inputs.cuda())
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_features(_inputs.cuda())
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)