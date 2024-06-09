import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import nn
from argparse import ArgumentParser

from methods.single_step.finetune_normal import Finetune_normal
from utils.toolkit import count_parameters, tensor2numpy, target2onehot


EPSILON = 1e-8

def add_special_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--prompt_length', type=int, default=None, help='length of prompt')
    return parser

# base is finetune with or without memory_bank
class CLIP_Text(Finetune_normal):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self.prompt_len = config.prompt_length

        self._class_to_idx = None
        self._idx_to_class = None

    def prepare_task_data(self, data_manager):
        # self._cur_task += 1
        # self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._cur_task = data_manager.nb_tasks - 1
        self._cur_classes = data_manager.get_task_size(0)

        self._total_classes = self._known_classes + self._cur_classes
        
        self._train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes), source='train', mode='train')
        self._test_dataset = data_manager.get_dataset(indices=np.arange(0, self._total_classes), source='test', mode='test')
        self._openset_test_dataset = data_manager.get_openset_dataset(known_indices=np.arange(0, self._total_classes), source='test', mode='test')

        self._logger.info('Train dataset size: {}'.format(len(self._train_dataset)))
        self._logger.info('Test dataset size: {}'.format(len(self._test_dataset)))

        self._train_loader = DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        self._test_loader = DataLoader(self._test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

        if self._class_to_idx is None:
            self._class_to_idx = data_manager.class_to_idx
            self._idx_to_class = dict((value, key) for key, value in self._class_to_idx.items())
        

    def prepare_model(self, checkpoint=None):
        if self._network == None:
            from backbone.clip_zoo import CLIPSZoo, PromptLearner

            self._network = CLIPSZoo(self._logger, self._config.backbone)
            prompt_module = PromptLearner(self._logger, self._network, self.prompt_len)
            self._network.set_prompt_module(prompt_module)

        new_class_name = [self._idx_to_class[class_id] for class_id in range(self._known_classes, self._total_classes)]
        self._network.update_known_class_name(new_class_name)

        if self._config.freeze_fe:
            self._network.freeze_FE()

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
            logits, features = model(inputs)

            loss = F.cross_entropy(logits, targets)
            
            losses += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            # preds = torch.argmax(logits[:, task_begin:task_end], dim=1) + task_begin
            
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
        cnn_pred_all, target_all = [], []
        img_features, text_features = [], []
        cnn_max_scores_all = []
        model.eval()
        for _, inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, features = model(inputs)

            cnn_max_scores, cnn_preds = torch.max(torch.softmax(logits, dim=-1), dim=-1)
            
            if ret_pred_target:
                cnn_pred_all.append(tensor2numpy(cnn_preds))
                target_all.append(tensor2numpy(targets))
                img_features.append(tensor2numpy(features['img_features']))
                text_features.append(tensor2numpy(features['text_features']))
                cnn_max_scores_all.append(tensor2numpy(cnn_max_scores))
            else:
                cnn_correct += cnn_preds.eq(targets).cpu().sum()
                total += len(targets)
        
        if ret_pred_target:
            cnn_pred_all = np.concatenate(cnn_pred_all)
            target_all = np.concatenate(target_all)
            cnn_max_scores_all = np.concatenate(cnn_max_scores_all)
            img_features = np.concatenate(img_features)
            text_features = np.concatenate(text_features)
            features_all = {'img_features': img_features, 'text_features':text_features}
            return cnn_pred_all, cnn_max_scores_all, target_all, features_all
        else:
            test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
            return test_acc