'''
 * Based on vit from blip code base
 * https://github.com/salesforce/BLIP
'''

import torch
import torch.nn as nn
import copy

from backbone.inc_net import get_backbone

class ViTZoo(nn.Module):
    def __init__(self, logger, backbone_type, pretrained, pretrain_path):
        super(ViTZoo, self).__init__()
        self._logger = logger
        self.task_sizes = []
        self.output_features = {}
        
        # get feature encoder
        self.feat = get_backbone(self._logger, backbone_type, pretrained, pretrain_path)
        if hasattr(self.feat, 'head'):
            self.feat_dim = self.feat.head.in_features
            self.feat.head = nn.Identity()
            self.embed_dim = self.feat.embed_dim
        else:
            # for clip pretrained models
            self.feat_dim = self.feat.output_dim
            self.embed_dim = 768
        
        # create prompting module
        self.prompt = None

        # classifier
        self.fc = None
    
    def set_prompt_module(self, prompt_module):
        self.prompt = prompt_module

    # pen: get penultimate features    
    def forward(self, x, train=False):
        if not isinstance(self.fc, nn.Identity):
            with torch.no_grad():
                q = self.feat(x)
            out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=len(self.task_sizes)-1)
        else:
            out = self.feat(x)
        self.output_features['features'] = out
        
        out = self.fc(out)
        if (not isinstance(self.fc, nn.Identity)) and train:
            return out, self.output_features, prompt_loss
        else:
            return out, self.output_features
    
    def freeze_FE(self):
        for param in self.feat.parameters():
            param.requires_grad = False
        self.feat.eval()
        self._logger.info('Freezing feature extractor(requires_grad=False) ...')
        return self
    
    def update_fc(self, nb_classes):
        self.task_sizes.append(nb_classes - sum(self.task_sizes))
        if (not isinstance(self.fc, nn.Identity)) and len(self.task_sizes) > 1:
            self.prompt.process_task_count()
        fc = nn.Linear(self.feat_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
            self._logger.info('Updated classifier head output dim from {} to {}'.format(nb_output, nb_classes))
        else:
            self._logger.info('Created classifier head with output dim {}'.format(nb_classes))
        del self.fc
        self.fc = fc
