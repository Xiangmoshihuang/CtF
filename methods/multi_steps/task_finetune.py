from os.path import join

import numpy as np
import torch
import torch.nn.functional as F

from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import accuracy, count_parameters, tensor2numpy, cal_bwf, mean_class_recall, cal_class_avg_acc, cal_avg_forgetting, cal_openset_test_metrics

EPSILON = 1e-8

'''
新方法命名规则: 
python文件(方法名小写) 
类名(方法名中词语字母大写)
'''

# base is finetune with or without memory_bank
class Task_Finetune(Finetune_IL):

    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        losses = 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            logits, feature_outputs = model(inputs)
            # task-specific loss
            loss = F.cross_entropy(logits[:, task_begin:task_end], targets-task_begin)
            preds = torch.max(logits[:, task_begin:task_end], dim=1)[1] + task_begin
    
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
    