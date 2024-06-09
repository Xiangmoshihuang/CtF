import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import nn
from argparse import ArgumentParser

from backbone.inc_net import load_clip_to_cpu
from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import count_parameters, tensor2numpy, target2onehot


EPSILON = 1e-8

def add_special_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--context', type=str, default=None, help="context for continual clip testing. e.g. A photo of a{}.")
    return parser

# base is finetune with or without memory_bank
class Continual_CLIP(Finetune_IL):
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self._class_to_idx = None
        self._idx_to_class = None

        self._context = config.context

        self._text_inputs = None

    def prepare_task_data(self, data_manager):
        self._cur_task += 1
        self._cur_classes = data_manager.get_task_size(self._cur_task)
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
        from backbone.clip.clip import tokenize
        # import clip
        
        if self._network == None:
            self._network = load_clip_to_cpu(self._config.backbone)
            # model, preprocess = clip.load("ViT-B/32", device='cpu')

        new_class_name = [self._idx_to_class[class_id] for class_id in range(self._known_classes, self._total_classes)]
        text_inputs = torch.cat([tokenize(self._context.format(c)) for c in new_class_name]).cuda()
        self._text_inputs = text_inputs if self._text_inputs is None else torch.cat([self._text_inputs, text_inputs])

        if self._config.freeze_fe:
            self._network.freeze_FE()

        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            self._logger.info("Loaded checkpoint model's state_dict !")

        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        self._network = self._network.cuda()
    
    def incremental_train(self):
        pass
    
    def _epoch_test(self, model, test_loader, ret_task_acc=False, ret_pred_target=False, task_begin=None, task_end=None, task_id=None):
        cnn_correct, cnn_task_correct, total, task_total = 0, 0, 0, 0
        cnn_pred_all, target_all = [], []
        img_features_all, text_features_all = [], []
        cnn_max_scores_all = []
        model.eval()
        for _, inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                image_features = model.encode_image(inputs)
                text_features = model.encode_text(self._text_inputs)
            
            # normalized features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()

            cnn_max_scores, cnn_preds = torch.max(torch.softmax(logits_per_image, dim=-1), dim=-1)
            
            if ret_pred_target:
                cnn_pred_all.append(tensor2numpy(cnn_preds))
                target_all.append(tensor2numpy(targets))
                img_features_all.append(tensor2numpy(image_features))
                text_features_all.append(tensor2numpy(text_features))
                cnn_max_scores_all.append(tensor2numpy(cnn_max_scores))
            else:
                if ret_task_acc:
                    task_data_idxs = torch.argwhere(torch.logical_and(targets>=task_begin, targets<task_end))
                    cnn_task_correct += cnn_preds[task_data_idxs].eq(targets[task_data_idxs]).cpu().sum()
                    task_total += len(task_data_idxs)
                cnn_correct += cnn_preds.eq(targets).cpu().sum()
                total += len(targets)
        
        if ret_pred_target:
            cnn_pred_all = np.concatenate(cnn_pred_all)
            target_all = np.concatenate(target_all)
            cnn_max_scores_all = np.concatenate(cnn_max_scores_all)
            img_features_all = np.concatenate(img_features_all)
            text_features_all = np.concatenate(text_features_all)
            features_all = {'img_features': img_features_all, 'text_features':text_features_all}
            return cnn_pred_all, None, cnn_max_scores_all, None, target_all, features_all
        else:
            test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
            if ret_task_acc:
                test_task_acc = np.around(tensor2numpy(cnn_task_correct)*100 / task_total, decimals=2)
                return test_acc, test_task_acc
            else:
                return test_acc