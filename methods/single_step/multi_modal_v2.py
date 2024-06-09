import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import nn

from backbone.cnn_multi_modal_net import CNN_Multi_Modal_Net
from methods.single_step.finetune_normal import Finetune_normal
from utils.toolkit import count_parameters, tensor2numpy


EPSILON = 1e-8

'''
新方法命名规则: 
python文件(方法名小写) 
类名(方法名中词语字母大写)
'''

# base is finetune with or without memory_bank
class Multi_Modal_V2(Finetune_normal):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._mode = self._config.mode.split('|') if self._config.mode is not None else []
        self.T = self._config.T

        self._class_to_idx = None
        self._idx_to_class = None

        self._text_ground_truth = None

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
        

    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = CNN_Multi_Modal_Net(self._logger, self._config.backbone, self._config.pretrained, 
                                                self._config.pretrain_path, self._config.layer_names, self._mode)

        new_class_name = [self._idx_to_class[class_id] for class_id in range(self._known_classes, self._total_classes)]
        self._network.update_known_class_name(new_class_name)

        self._network = self._network.cuda()
        self._logger.info('Initializing task-{} adapter in network...'.format(self._cur_task))
        with torch.no_grad():
            self._network.eval()
            self._network.init_mode()
            self._network(torch.rand(1, 3, self._config.img_size, self._config.img_size).cuda())
            self._network.train_test_mode()

        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            self._logger.info("Loaded checkpoint model's state_dict !")

        self._text_ground_truth = self._network.get_text_vector()

        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        for layer_id in self._config.layer_names:
            controller_id = layer_id.replace('.', '_')+'_adapter_controller'
            if hasattr(self._network, controller_id):
                self._logger.info('{} params: {}'.format(controller_id, count_parameters(self._network.__getattr__(controller_id))))

    def _epoch_train(self, model, train_loader, optimizer, scheduler):
        correct, total = 0, 0
        losses = 0.

        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            # model forward has two mode which shoule be noticed before forward!
            img_features = model(inputs)
            img_features = F.normalize(img_features, dim=1)

            ### compute supervised contrastive loss (begin) ###
            similarity_matrix = torch.div(torch.matmul(img_features, self._text_ground_truth.T), self.T)
            # for numerical stability
            logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
            logits = similarity_matrix - logits_max.detach()

            label = targets.contiguous().view(-1, 1)
            mask = torch.eq(label, torch.arange(self._total_classes).view(1, -1).cuda()).float()
        
            # compute log_prob
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+EPSILON)

            # loss
            loss = - mean_log_prob_pos
            loss = loss.mean()

            if loss != loss:
                print('Stop!')
            ### compute supervised contrastive loss (end) ###

            losses += loss.item()
            
            preds = torch.argmax(torch.cosine_similarity(img_features.unsqueeze(1), self._text_ground_truth.unsqueeze(0), dim=-1),dim=1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        
        if scheduler != None:
            scheduler.step()
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
            img_features = model(inputs)

            cnn_max_scores, cnn_preds = torch.max(torch.cosine_similarity(img_features.unsqueeze(1), self._text_ground_truth.unsqueeze(0), dim=-1),dim=1)
            
            if ret_pred_target:
                cnn_pred_all.append(tensor2numpy(cnn_preds))
                target_all.append(tensor2numpy(targets))
                features_all.append(tensor2numpy(img_features))
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