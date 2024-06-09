import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim

from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import tensor2numpy

EPSILON = 1e-8

class Co2L(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._old_network = None
        self._T = config.T
        self._lamda = config.lamda
        self._epochs_finetune = config.epochs_finetune
        self._lrate_finetune = config.lrate_finetune
        self._milestones_finetune = config.milestones_finetune

        self._is_finetune = False

        if self._incre_type != 'cil':
            raise ValueError('Co2L is a class incremental method!')
    
    def prepare_task_data(self, data_manager):
        self._cur_task += 1
        self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_classes

        if self._cur_task > 0 and self._memory_bank != None:
            self._train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes), 
                    source='train', mode='train', appendent=self._memory_bank.get_memory(), two_view=True)
            self._normal_train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes), 
                    source='train', mode='train', appendent=self._memory_bank.get_memory())
        else:
            self._train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='train', two_view=True)
            self._normal_train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='train')
        
        self._test_dataset = data_manager.get_dataset(indices=np.arange(0, self._total_classes), source='test', mode='test')
        self._openset_test_dataset = data_manager.get_openset_dataset(known_indices=np.arange(0, self._total_classes), source='test', mode='test')

        self._logger.info('Train dataset size: {}'.format(len(self._train_dataset)))
        self._logger.info('Test dataset size: {}'.format(len(self._test_dataset)))

        self._train_loader = DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers, drop_last=True)
        self._test_loader = DataLoader(self._test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
        self._openset_test_loader = DataLoader(self._openset_test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

        self._sampler_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='test')

    def prepare_model(self, checkpoint=None):
        super().prepare_model(checkpoint)
        if self._old_network is not None:
            self._old_network.cuda()
    
    def incremental_train(self):
        self._logger.info('-'*10 + ' Learning on task {}: {}-{} '.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)
        self._is_finetuning = False
        self._network.activate_FE()
        optimizer = self._get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()), self._config, self._cur_task==0)
        scheduler = self._get_scheduler(optimizer, self._config, self._cur_task==0)
        if self._cur_task == 0:
            epochs = self._init_epochs
        else:
            epochs = self._epochs
        self._network = self._train_model(self._network, self._train_loader, self._test_loader, optimizer, scheduler,
            task_id=self._cur_task, epochs=epochs, note='stage1')

        self._logger.info('Finetune the network (classifier part) with the balanced dataset!')
        self._is_finetune = True
        self._network.freeze_FE()
        finetune_train_dataset = self._memory_bank.get_unified_sample_dataset(self._normal_train_dataset, self._network)
        finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=self._batch_size,
                                        shuffle=True, num_workers=self._num_workers)
        self._network.reset_fc_parameters()
        ft_optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.fc.parameters()), momentum=0.9, lr=self._lrate_finetune)
        ft_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=ft_optimizer, milestones=self._milestones_finetune, gamma=0.1)
        self._network = self._train_model(self._network, finetune_train_loader, self._test_loader, ft_optimizer, ft_scheduler,
            task_id=self._cur_task, epochs=self._epochs_finetune, note='stage2')
        self._is_finetune = False


    def after_task(self):
        super().after_task()
        self._old_network = self._network.copy().freeze()

    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        losses = 0.
        ce_losses, sup_contrast_losses, kd_losses = 0., 0., 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            if self._is_finetune:
                inputs, targets = inputs.cuda(), targets.cuda()
                logits, feature_outputs = model(inputs)
                loss = F.cross_entropy(logits[:,:task_end], targets)
                ce_losses += loss.item()
                
                preds = torch.max(logits[:,:task_end], dim=1)[1]
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)
            else:
                inputs = torch.cat([inputs[0], inputs[1]], dim=0).cuda()
                targets = targets.cuda()

                features = model(inputs)[1]['features']
                features = F.normalize(features, dim=1) # 对特征向量做归一化
                
                # Asym SupCon
                loss = self.sup_contrastive_loss_modified(features, targets, list(range(task_begin, task_end)))
                sup_contrast_losses += loss.item()

                # IRD
                if task_id > 0:
                    # current
                    features_sim1 = torch.div(torch.matmul(features, features.T), 0.2) # current_temp=0.2

                    logits_mask = (1 - torch.eye(*features_sim1.shape)).cuda()
                    features_sim1 = features_sim1 - torch.max(features_sim1*logits_mask, dim=1, keepdim=True)[0].detach()
                    row_size = features_sim1.size(0)
                    exp_logits1 = torch.exp(features_sim1[logits_mask.bool()].view(row_size, -1))
                    logits1 = exp_logits1 / exp_logits1.sum(dim=1, keepdim=True)

                    # past
                    with torch.no_grad():
                        features2_prev_task = self._old_network(inputs)[1]['features']
                        features2_prev_task = F.normalize(features2_prev_task, dim=1) # 对特征向量做归一化

                        features2_sim = torch.div(torch.matmul(features2_prev_task, features2_prev_task.T), 0.01) # past_temp=0.01
                        features2_sim = features2_sim - torch.max(features2_sim*logits_mask, dim=1, keepdim=True)[0].detach()
                        exp_logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1))
                        logits2 = exp_logits2 /  exp_logits2.sum(dim=1, keepdim=True)

                    loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()
                    kd_losses += loss_distill.item()
                    loss += self._lamda * loss_distill
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
        
        if scheduler != None:
            scheduler.step()
        
        if self._is_finetune:
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            train_loss = ['Loss', losses/len(train_loader), 'Loss_ce', ce_losses/len(train_loader)]
        else:
            train_acc = 0
            train_loss = ['Loss', losses/len(train_loader), 'Loss_supContrast', sup_contrast_losses/len(train_loader), 'Loss_IRD', kd_losses/len(train_loader)]

        return model, train_acc, train_loss


    def sup_contrastive_loss_modified(self, features, labels, target_labels):

        if labels.shape[0] != self._batch_size:
            raise ValueError('Num of labels does not match num of features')
        unsqueezed_labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(unsqueezed_labels, unsqueezed_labels.T).float()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.div(torch.matmul(features, features.T), self._T)
        # for numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        anchor_count, contrast_count = 2, 2 # 一般都是做两次增广，所以这里写死为2
        mask = mask.repeat(anchor_count, contrast_count)
        
        # mask-out self-contrast cases
        logits_mask = 1 - torch.eye(*mask.shape).cuda() # 相当于单位矩阵取反，即对角线为0，其余为1
        mask = mask * logits_mask # 屏蔽掉自己与自己，将mask的对角线置为0

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask # 屏蔽掉自己
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self._T / 0.07) * mean_log_prob_pos

        # modified part (begin) #
        curr_class_mask = torch.zeros_like(labels)
        for tl in target_labels:
            curr_class_mask += (labels == tl)
        curr_class_mask = curr_class_mask.view(-1).cuda()
        loss = curr_class_mask * loss.view(anchor_count, self._batch_size)
        # modified part (end) #

        # loss = loss.view(anchor_count, self.batch_size)
        # loss = loss.mean(0) # return batch wise loss
        
        loss = loss.mean()
        if loss != loss:
            print('stop!')

        return loss