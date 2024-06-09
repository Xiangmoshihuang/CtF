from backbone.inc_net import IncrementalNet
from typing import Callable, Iterable
from torch import nn
import torch
import torch.nn.functional as F
import math

class CNN_Experts_v1(nn.Module):
    def __init__(self, mode, expert_per_task:int, fm_size:int, in_planes:int, kernel_num:int, kernel_size:int=3, cnn_prompt_num:int=0):
        super().__init__()
        self.mode = mode
        self.expert_per_task = expert_per_task
        self.in_planes = in_planes
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.cnn_prompt_num = cnn_prompt_num

        self.padding = torch.nn.ZeroPad2d(int((self.kernel_size-1)/2))
        self.conv_special_param = nn.Parameter(torch.empty(self.expert_per_task, self.kernel_num, self.in_planes, self.kernel_size, self.kernel_size))
        self.conv_channel_param = nn.Parameter(torch.empty(self.expert_per_task, self.in_planes, self.in_planes+self.kernel_num+self.cnn_prompt_num, 1, 1))
        if self.cnn_prompt_num > 0:
            self.cnn_prompt_param = nn.Parameter(torch.empty(self.expert_per_task, self.cnn_prompt_num, fm_size, fm_size))
            torch.nn.init.kaiming_uniform_(self.cnn_prompt_param, a=math.sqrt(5))

        torch.nn.init.kaiming_uniform_(self.conv_special_param, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.conv_channel_param, a=math.sqrt(5))
    
    def forward(self, x, gate):
        origin_h, origin_w = x.shape[-2:]
        padded_x = self.padding(x)
        b, c, h, w = padded_x.shape
        out = []
        for i in range(gate.shape[1]):
            param_idxs = gate[:, i]
            special_param = self.conv_special_param[param_idxs] # b, self.kernel_num, c, self.kernel_size, self.kernel_size
            new_features = F.conv2d(padded_x.reshape(b*c, h, w), special_param.reshape(-1, c, self.kernel_size, self.kernel_size), groups=b)
            new_features = new_features.reshape(b, self.kernel_num, origin_h, origin_w)

            if self.cnn_prompt_num > 0:
                cnn_prompt = self.cnn_prompt_param[param_idxs]
                new_features = torch.cat([new_features, cnn_prompt], dim=1)

            cated_features = torch.cat([x, new_features], dim=1) # batch_size, c+kernel_num, h, w
            channel_param = self.conv_channel_param[param_idxs] # batch_size, c, c+kernel_num, 1, 1
            channel_out = F.conv2d(cated_features.reshape(-1, origin_h, origin_w), channel_param.reshape(-1, c+self.kernel_num, 1, 1), groups=b)
            out.append(channel_out.reshape(b, c, origin_h, origin_w))
            
        return torch.stack(out, dim=1) # b, topk, c, h, w

class CNN_Gate_v1(nn.Module):
    def __init__(self, mode, expert_per_task:int, fm_size:int, in_planes:int, kernel_num:int, **kwargs):
        super().__init__()
        self.mode = mode
        self.fm_size = fm_size
        self.in_planes = in_planes
        self.expert_per_task = expert_per_task

        if 'conv1x1_fm_compress' in self.mode:
            self.conv_channel = nn.Parameter(torch.empty(1, self.in_planes, 1, 1))
            torch.nn.init.kaiming_uniform_(self.conv_channel, a=math.sqrt(5))
        
        self.prototype = nn.Parameter(torch.empty(self.expert_per_task, fm_size*fm_size))
        torch.nn.init.kaiming_uniform_(self.prototype, a=math.sqrt(5))
    
    def forward(self, x):
        if 'conv1x1_fm_compress' in self.mode:
            query_fm = torch.conv2d(x, self.conv_channel).reshape(x.shape[0], -1)
        elif 'avg_fm_compress' in self.mode:
            query_fm = (torch.sum(x, dim=1) / x.shape[1]).reshape(x.shape[0], -1)
        else:
            raise ValueError('Unknown feature map compress mode: {}'.format(self.mode))
        
        normed_query_fm = F.normalize(query_fm, dim=1)
        dists = torch.cdist(normed_query_fm, self.prototype, p=2)
        
        return torch.softmax(dists, dim=-1), query_fm # [b, expert_num], [b, h, w]

class CNN_Gate_v2(nn.Module):
    def __init__(self, mode, expert_per_task:int, fm_size:int, in_planes:int, kernel_num:int=1, kernel_size:int=3):
        super().__init__()
        self.mode = mode
        self.fm_size = fm_size
        self.in_planes = in_planes
        self.expert_per_task = expert_per_task
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size

        self.width_conv = nn.Parameter(torch.empty(self.kernel_num, self.in_planes, self.fm_size, self.kernel_size))
        torch.nn.init.kaiming_uniform_(self.width_conv, a=math.sqrt(5))

        self.high_conv = nn.Parameter(torch.empty(self.kernel_num, self.in_planes, self.kernel_size, self.fm_size))
        torch.nn.init.kaiming_uniform_(self.high_conv, a=math.sqrt(5))
        
        self.width_prototype = nn.Parameter(torch.empty(self.expert_per_task, fm_size-self.kernel_size+1))
        torch.nn.init.kaiming_uniform_(self.width_prototype, a=math.sqrt(5))

        self.high_prototype = nn.Parameter(torch.empty(self.expert_per_task, fm_size-self.kernel_size+1))
        torch.nn.init.kaiming_uniform_(self.high_prototype, a=math.sqrt(5))
    
    def forward(self, x):
        query_width = torch.conv2d(x, self.width_conv).reshape(x.shape[0], -1) # [b, kernel_num, fm_size-kernel_size+1]
        query_high = torch.conv2d(x, self.high_conv).reshape(x.shape[0], -1) # [b, kernel_num, fm_size-kernel_size+1]
        
        normed_query_width = F.normalize(query_width, dim=1)
        dists_width = torch.cdist(normed_query_width, self.width_prototype, p=2)

        normed_query_high = F.normalize(query_high, dim=1)
        dists_high = torch.cdist(normed_query_high, self.high_prototype, p=2)
        
        # [b, expert_num], [b, 2, kernel_num, fm_size-kernel_size+1]
        return torch.softmax(dists_width+dists_high, dim=-1), torch.stack([query_width, query_high], dim=1)

class CNN_MoE_v1(nn.Module):
    def __init__(self, mode, fm_size:int, in_planes:int, kernel_num:int, expert_per_task:int, topk:int):
        super().__init__()
        self.mode = mode
        self.fm_size = fm_size
        self.in_planes = in_planes
        self.kernel_num = kernel_num
        self.expert_per_task = expert_per_task
        self.topk = topk

        self.experts = CNN_Experts_v1(self.mode, self.expert_per_task, self.fm_size, self.in_planes, kernel_num=self.kernel_num)
        if 'gate_v2' in self.mode:
            self.gate = CNN_Gate_v2(self.mode, self.expert_per_task, self.fm_size, self.in_planes, kernel_num=self.kernel_num)
        else:
            self.gate = CNN_Gate_v1(self.mode, self.expert_per_task, self.fm_size, self.in_planes, kernel_num=self.kernel_num)
    
    def forward(self, pretrained_x, x):
        features = {}
        b, c, h, w = x.shape
        
        if 'gate_w_pretrain_fm' in self.mode:
            scores, query_fm = self.gate(pretrained_x)
        else:
            scores, query_fm = self.gate(x)

        max_scores, idxs = torch.topk(scores, self.topk, dim=-1) # b, self.topk
        experts_out = self.experts(x, idxs) # b, topk, c, h, w
        out = torch.sum(max_scores.reshape(b, self.topk, 1, 1, 1) * experts_out, dim=1)
        
        features['gate_scores'] = scores # b, expert_per_task
        features['expert_idxs'] = idxs # b, topk
        features['query_fm'] = query_fm # b, h*w
        features['experts_out'] = experts_out # b, c, h, w

        return out, features


class CNN_MoE_Net_V1(IncrementalNet):
    def __init__(self, logger, backbone_type, pretrained, pretrain_path=None, layer_names:Iterable[str]=[], mode=None, expert_kernel_num=1, expert_per_task=3, topk=2):
        '''
        layers_name can be ['conv1','layer1','layer2','layer3','layer4'] for resnet18
        '''
        super().__init__(logger, backbone_type, pretrained, pretrain_path)
        self.mode = mode
        self.expert_kernel_num = expert_kernel_num
        self.expert_per_task = expert_per_task
        self.topk = topk
        self.layer_names = [] if layer_names is None else layer_names

        self.forward_batch_size = None

        model_dict = dict([*self.feature_extractor.named_modules()]) 
        for layer_id in self.layer_names:
            moe_id = layer_id.replace('.', '_')+'_MoEs'
            self.register_module(moe_id, nn.ModuleList([]))
            layer = model_dict[layer_id]
            layer.register_forward_pre_hook(self.apply_adapters(moe_id))
        
    def apply_adapters(self, moe_id: str) -> Callable:
        def hook(module, input):
            if isinstance(input, tuple):
                input = input[0]
            b, c, h, w = input.shape
            
            if len(getattr(self, moe_id)) < len(self.task_sizes):
                if 'moe_v1' in self.mode:
                    getattr(self, moe_id).append(CNN_MoE_v1(self.mode, h, c, self.expert_kernel_num, self.expert_per_task, self.topk).cuda())
                    self._logger.info('Created MoE v1 before layer-{}, feature_map_shape = ({}, {}, {})'.format(moe_id, c, h, w))
                else:
                    raise ValueError('Unknown MoE type in mode: {}'.format(self.mode))
            
            pretrained_fm = input[:self.forward_batch_size]
            fm = input[self.forward_batch_size:]

            moe_out, moe_features = getattr(self, moe_id)[-1](pretrained_fm, fm)
            out = torch.cat([pretrained_fm, moe_out+fm], dim=0)

            self.output_features[moe_id] = moe_features

            return (out,)
            
        return hook
    
    def forward(self, x):
        self.forward_batch_size = x.shape[0]
        features = self.feature_extractor(x.repeat(2, 1, 1, 1))[self.forward_batch_size:]
        out = self.fc(features)

        self.output_features['features'] = features
        return out, self.output_features

    def freeze_FE(self):
        for name, param in self.feature_extractor.named_parameters():
            if 'activate_bn' in self.mode and 'bn' in name:
                continue
            
            param.requires_grad = False
            
        self.feature_extractor.eval()
        self._logger.info('Freezing feature extractor(requires_grad=False) ...')
        return self
