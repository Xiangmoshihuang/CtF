import torch
from torch import nn
import copy
from backbone.inc_net import get_backbone

from backbone.dynamic_er_net import DERNet

class FOSTERNet(DERNet):
    def __init__(self, logger, backbone_type, pretrained, pretrain_path=None):
        super().__init__(logger, backbone_type, pretrained, pretrain_path)
        self.fe_fc = None
        self.old_fc = None

    def forward(self, x):
        out_dict = {}
        features = [fe(x) for fe in self.feature_extractor]
        all_features = torch.cat(features, 1)

        out=self.fc(all_features) #{logics: self.fc(features)}
        fe_logits = self.fe_fc(features[-1])

        out_dict.update({"fe_logits": fe_logits, "features": all_features})

        if self.old_fc is not None:
            old_logits = self.old_fc(all_features[:, : -self._feature_dim])
            out_dict.update({"old_logits": old_logits})

        return out, out_dict
    
    def update_fc(self, nb_classes):
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        ft = get_backbone(self._logger, self.backbone_type, pretrained=self.pretrained, pretrain_path=self.pretrain_path)
        if ('resnet' in self.backbone_type or 'vit' in self.backbone_type) and 'clip' in self.backbone_type:
            feature_dim = ft.output_dim
        if 'resnet' in self.backbone_type:
            feature_dim = ft.fc.in_features
            ft.fc = nn.Identity()
        elif 'efficientnet' in self.backbone_type:
            feature_dim = ft.classifier[1].in_features
            ft.classifier = nn.Dropout(p=0.4, inplace=True)
        elif 'mobilenet' in self.backbone_type:
            feature_dim = ft.classifier[-1].in_features
            ft.classifier = nn.Dropout(p=0.2, inplace=False)
        elif 'vit_base_patch16_224' in self.backbone_type:
            feature_dim = ft.head.in_features
            ft.head = nn.Identity()
        else:
            raise ValueError('{} did not support yet!'.format(self.backbone_type))

        if len(self.feature_extractor)==0:
            self.feature_extractor.append(ft)
            self._feature_dim = feature_dim
        else:
            self.feature_extractor.append(ft)
            self.feature_extractor[-1].load_state_dict(self.feature_extractor[-2].state_dict())
            
        fc = self.generate_fc(self._feature_dim*len(self.feature_extractor), nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output,:-self._feature_dim] = weight
            fc.bias.data[:nb_output] = bias

        self.old_fc = self.fc
        self.fc = fc
        self.fe_fc = self.generate_fc(self._feature_dim, nb_classes)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        self._logger.info("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma