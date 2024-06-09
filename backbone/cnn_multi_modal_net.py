from backbone.inc_net import IncrementalNet
from torch.nn import functional as F
from typing import Callable, Iterable
from torch import nn
import torch
import clip


class CNN_Multi_Modal_Net(IncrementalNet):
    def __init__(self, logger, backbone_type, pretrained, pretrain_path=None, layer_names:Iterable[str]=[], mode=None):
        '''
        layers_name can be ['conv1','layer1','layer2','layer3','layer4']
        '''
        super().__init__(logger, backbone_type, pretrained, pretrain_path)
        
        self._text_encoder, _ = clip.load("ViT-B/32", "cuda")
        self.mode = mode
        self.layer_names = [] if layer_names is None else layer_names
        self.task_sizes = []
        self._known_class_names = []
        self._known_class_embeding = []
        self._known_class_text_vector = []

        self._training_mode = 'unknown_mode'

        self._img_text_param = None

        self._prompt_keys = None
        self._old_prompt_keys = None

        model_dict = dict([*self.feature_extractor.named_modules()])
        for layer_id in self.layer_names:
            adapter_id = layer_id.replace('.', '_')+'_adapter'
            layer = model_dict[layer_id]
            layer.register_forward_pre_hook(self.apply_adapters(adapter_id))

    def init_mode(self):
        self._training_mode = 'init_mode'
        # self._logger.info("Training mode 'init_mode' is set !")
    
    def train_test_mode(self):
        self._training_mode = 'train_test_mode'
        # self._logger.info("Training mode 'train_test_mode' is set !")
    
    def skip_adapters_mode(self):
        self._training_mode = 'skip_adapters_mode'
        # self._logger.info("Training mode 'skip_adapters_mode' is set !")

    def apply_adapters(self, adapter_id: str) -> Callable:
        def hook(module, input):
            if isinstance(input, tuple):
                input = input[0]
            b, c, h, w = input.shape
            if self._training_mode == 'skip_adapters_mode':
                return (input,)
            elif self._training_mode == 'init_mode':
                if 'text_img_attention' in self.mode:
                    input_dim = self.feature_dim
                elif 'text_img_cat' in self.mode:
                    input_dim = self.feature_dim*2

                if 'MLP_controller' in self.mode:
                    self.register_module(adapter_id+'_controller', nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(),
                                                       nn.Linear(input_dim, c*c+c)).cuda())
                    self._logger.info('Created {} MLP controller: {}=>{}=>{}'.format(adapter_id, input_dim, input_dim, c*c+c))
                elif 'FC_controller' in self.mode:
                    self.register_module(adapter_id+'_controller', nn.Linear(input_dim, c*c+c).cuda())
                    self._logger.info('Created {} FC controller: {}=>{}'.format(adapter_id, input_dim, c*c+c))
                return (input,)
            
            elif self._training_mode == 'train_test_mode':
                param = getattr(self, adapter_id+'_controller')(self._img_text_param)

                # more efficient version of inplementation!
                weight = param[:, :c*c].reshape(b*c, c, 1, 1)
                bias = param[:, -c:].reshape(b*c)
                out_features = F.conv2d(input.reshape(b*c, h, w), weight, bias, groups=b).reshape(b, c, h, w)
                return (out_features + torch.relu(out_features),)

                # very slow version of inplementation!
                # out_features = []
                # for i in range(b):
                #     weight = param[i][:getattr(self, adapter_id+'_weight_dim')*getattr(self, adapter_id+'_weight_dim')]
                #     weight = weight.reshape(c, c, 1, 1)
                #     bias = param[i][-getattr(self, adapter_id+'_weight_dim'):]
                #     out_features.append(F.conv2d(input[i], weight, bias))
                # out_features = torch.stack(out_features, dim=0)
                # return (input + torch.relu(out_features),)
            else:
                raise ValueError('Unknown training mode in mode: {}'.format(self.mode))

        return hook
    
    def forward(self, x):
        if self._training_mode == 'init_mode':
            return self.feature_extractor(x)
        else:
            self.skip_adapters_mode()
            pretrain_img_features = self.feature_extractor(x)   # [b, feature_dim]

            if self._old_prompt_keys is None or 'freeze_old_prompt_keys' not in self.mode:
                prompt_weights = F.cosine_similarity(pretrain_img_features.unsqueeze(1), self._prompt_keys.unsqueeze(0), dim=-1) # [b, class_num]
            else:
                old_prompt_weights = F.cosine_similarity(pretrain_img_features.unsqueeze(1), self._old_prompt_keys.unsqueeze(0), dim=-1)
                new_prompt_weights = F.cosine_similarity(pretrain_img_features.unsqueeze(1), self._prompt_keys.unsqueeze(0), dim=-1)
                prompt_weights = torch.cat([old_prompt_weights, new_prompt_weights], dim=-1)
            text_features = torch.matmul(prompt_weights, self._known_class_text_vector.float()) # [b, text_feature_dim]

            self.train_test_mode()
            if 'text_img_cat' in self.mode:
                self._img_text_param = torch.cat([pretrain_img_features, text_features], dim=-1)
            features = self.feature_extractor(x)

            return features

    def update_known_class_name(self, new_class_names):
        self._known_class_names.extend(new_class_names)

        new_class_embeding = torch.cat([clip.tokenize(f'A photo of a {item}') for item in new_class_names]).cuda()
        self._known_class_embeding = new_class_embeding if len(self._known_class_embeding) == 0 else torch.cat([self._known_class_embeding, new_class_embeding])

        with torch.no_grad():
            new_class_text_vector = self._text_encoder.encode_text(new_class_embeding).float()
            new_class_text_vector = F.normalize(new_class_text_vector, dim=1)
        self._known_class_text_vector = new_class_text_vector if len(self._known_class_text_vector) == 0 else torch.cat([self._known_class_text_vector, new_class_text_vector])
        
        if self._prompt_keys is None:
            self._prompt_keys = nn.Parameter(torch.rand((new_class_embeding.shape[0], self._feature_dim))).cuda()
            self._logger.info('Created prompt keys: {}'.format(self._prompt_keys.shape))
        else:
            if 'freeze_old_prompt_keys' in self.mode:
                self._old_prompt_keys = self._prompt_keys.data
                self._prompt_keys = nn.Parameter(torch.rand((new_class_embeding.shape[0], self._feature_dim))).cuda()
                self._logger.info('Freezed old prompt keys and created new prompt keys: {}'.format(self._prompt_keys.shape))
            else:
                prompt_keys = nn.Parameter(torch.rand((self._known_class_embeding.shape[0], self._feature_dim))).cuda()
                prompt_keys.data[:self._prompt_keys.data.shape[0], :] = self._prompt_keys.data
                self._logger.info('Updated prompt keys: {} => {}'.format(self._prompt_keys.shape, prompt_keys.shape))
                self._prompt_keys = prompt_keys

    def get_text_vector(self, indices=None):
        if indices is None:
            indices = torch.arange(len(self._known_class_text_vector))
        return self._known_class_text_vector[indices]

