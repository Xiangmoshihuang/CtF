import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchtext import vocab
from torch import nn

from backbone.inc_net import IncrementalNet
from methods.single_step.finetune_normal import Finetune_normal
from utils.toolkit import count_parameters, tensor2numpy


EPSILON = 1e-8

'''
新方法命名规则: 
python文件(方法名小写) 
类名(方法名中词语字母大写)
'''

# base is finetune with or without memory_bank
class Multi_Modal(Finetune_normal):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._mode = self._config.mode.split('|') if self._config.mode is not None else ''
        self.T = self._config.T

        # glove options:
        # 'charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 
        # 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 
        # 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d', 
        # 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d'
        self.glove = vocab.pretrained_aliases["glove.840B.300d"]()
        self.vec_dim = 300
        if 'w_text_network' in self._mode:
            self._text_network = nn.Linear(self.vec_dim, self.vec_dim).cuda()
            self._text_optimizer = self._get_optimizer(filter(lambda p: p.requires_grad, self._text_network.parameters()), self._config)
            self._text_scheduler = self._get_scheduler(self._text_optimizer, self._config)
            self._logger.info('Created text encoder!')

        self._class_to_idx = None
        self._idx_to_class = None

        self.ground_truth = None

    def prepare_task_data(self, data_manager):
        # self._cur_task += 1
        # self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._cur_task = data_manager.nb_tasks - 1
        self._cur_classes = data_manager.get_task_size(0)

        self._total_classes = self._known_classes + self._cur_classes
        
        train_dataset = data_manager.get_dataset(source='train', mode='train', indices=np.arange(self._known_classes, self._total_classes))
        test_dataset = data_manager.get_dataset(source='test', mode='test', indices=np.arange(self._known_classes, self._total_classes))
        
        self._logger.info('Train dataset size: {}'.format(len(train_dataset)))
        self._logger.info('Test dataset size: {}'.format(len(test_dataset)))

        self._train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        self._test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

        if self._class_to_idx is None:
            self._class_to_idx = data_manager.class_to_idx
            self._idx_to_class = dict((value, key) for key, value in self._class_to_idx.items())
        
        # accumulate ground truth
        add_gt = []
        for i in range(self._known_classes, self._total_classes):
            add_gt.append(self.glove.vectors[self.glove.stoi[self._idx_to_class[i]]])
        if self.ground_truth is None:
            self.ground_truth = torch.stack(add_gt, dim=0).cuda()
        else:
            self.ground_truth = torch.cat([self.ground_truth, torch.stack(add_gt, dim=0).cuda()], dim=0)

    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = IncrementalNet(self._logger, self._config.backbone, self._config.pretrained, self._config.pretrain_path)
            self._network.update_fc(self.vec_dim)
        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            self._logger.info("Loaded checkpoint model's state_dict !")

        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        self._network = self._network.cuda()

    def _epoch_train(self, model, train_loader, optimizer, scheduler):
        correct, total = 0, 0
        losses = 0.

        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            # model forward has two mode which shoule be noticed before forward!
            img_vec, output_features = model(inputs)
            
            target_vec = self.map_targets_to_vec(targets).cuda()
            if 'w_text_network' in self._mode:
                text_vec = self._text_network(target_vec)   # [b, feature_dim]
            else:
                text_vec = target_vec

            ### compute supervised contrastive loss (begin) ###
            similarity_matrix = torch.div(torch.matmul(img_vec, text_vec.T), self.T)
            # for numerical stability
            logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
            logits = similarity_matrix - logits_max.detach()

            label = targets.contiguous().view(-1, 1)
            mask = torch.eq(label, label.T).float()

            logits_mask = 1- torch.eye(targets.shape[0], targets.shape[0]).cuda() # 单位矩阵取反，即对角线为0，其余为1
            mask = mask * logits_mask # 屏蔽掉自己与自己，将mask的对角线置为0
        
            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask # 屏蔽掉自己
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+EPSILON)

            # loss
            loss = - mean_log_prob_pos
            loss = loss.mean()
            ### compute supervised contrastive loss (end) ###
            if loss != loss:
                print('stop here')

            ### compute predict result (begin) ###
            test_vec = self.map_targets_to_vec(torch.arange(self._total_classes)).cuda()
            if 'w_text_network' in self._mode:
                test_vec = self._text_network(test_vec)   # [b, feature_dim]
            preds = torch.max(torch.matmul(img_vec, test_vec.T), dim=1)[1]
            ### compute predict result (end) ###

            optimizer.zero_grad()
            if 'w_text_network' in self._mode:
                self._text_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if 'w_text_network' in self._mode:
                self._text_optimizer.step()
            losses += loss.item()

            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        
        if scheduler != None:
            scheduler.step()
            if 'w_text_network' in self._mode:
                self._text_scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader)]
        return model, train_acc, train_loss
    
    def _epoch_test(self, model, test_loader, ret_pred_target=False):
        cnn_correct, total = 0, 0
        cnn_pred_all, target_all, features_all = [], [], []
        cnn_max_scores_all = []
        model.eval()
        for _, inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            img_vec, feature_outputs = model(inputs)

            target_vec = self.map_targets_to_vec(torch.arange(self._total_classes)).cuda()
            if 'w_text_network' in self._mode:
                text_vec = self._text_network(target_vec)   # [b, feature_dim]
            else:
                text_vec = target_vec
            
            cnn_max_scores, cnn_preds = torch.max(torch.matmul(img_vec, text_vec.T), dim=1)
            
            if ret_pred_target:
                cnn_pred_all.append(tensor2numpy(cnn_preds))
                target_all.append(tensor2numpy(targets))
                features_all.append(tensor2numpy(img_vec))
                cnn_max_scores_all.append(tensor2numpy(cnn_max_scores))
            else:
                cnn_correct += cnn_preds.eq(targets).cpu().sum()
                total += len(targets)
        
        if ret_pred_target:
            cnn_pred_all = np.concatenate(cnn_pred_all)
            target_all = np.concatenate(target_all)
            features_all = np.concatenate(features_all)
            cnn_max_scores_all = np.concatenate(cnn_max_scores_all)
            return cnn_pred_all, cnn_max_scores_all, target_all, features_all
        else:
            test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
            return test_acc
    
    def map_targets_to_vec(self, targets):
        target_vec = []
        for i in range(len(targets)):
            class_name = self._idx_to_class[targets[i].item()]
            class_vec = self.glove.vectors[self.glove.stoi[class_name]]
            target_vec.append(class_vec)
        return torch.stack(target_vec, dim=0)