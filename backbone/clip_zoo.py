import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Callable, Iterable

from backbone.inc_net import load_clip_to_cpu
from backbone.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from backbone.clip import tokenize

_tokenizer = _Tokenizer()

class COOP_Prompt(nn.Module):
    def __init__(self, logger, clip_model, prompt_len, prompt_mode='class_specific', class_token_position='end'):
        super().__init__()
        self._logger = logger
        self.dtype = clip_model.dtype
        self.embed_dim = clip_model.embed_dim
        self.prompt_len = prompt_len
        self.prompt_mode = prompt_mode
        self.class_token_position = class_token_position
        self.token_embedding = clip_model.token_embedding

        # self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = []

        self.new_prompt_tokens = None
        self.old_prompt_tokens = None
    
    def update_new_classes(self, new_class_names):
        self._logger.info("Initializing new class prompts")
        class_names = [name.replace("_", " ") for name in new_class_names]
        name_lens = [len(_tokenizer.encode(name)) for name in class_names]
        self.name_lens.extend(name_lens)

        prompt_prefix = " ".join(["X"] * self.prompt_len) # "X" will be replaced by learnable prompt afterwards
        prompts = [prompt_prefix + " " + name + "." for name in class_names]
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.token_embedding.weight.device)
        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts).type(self.dtype)

        if self.prompt_mode == 'class_specific':
            new_prompt_tokens = torch.empty(len(new_class_names), self.prompt_len, self.embed_dim, dtype=self.dtype)
            nn.init.normal_(new_prompt_tokens, std=0.02)
        elif self.prompt_mode == 'task_specific':
            new_prompt_tokens = torch.empty(self.prompt_len, self.embed_dim, dtype=self.dtype)

        if self.new_prompt_tokens is None:
            self.new_prompt_tokens = nn.Parameter(new_prompt_tokens, requires_grad=True)  # to be optimized
            
            self.register_buffer('tokenized_prompts', tokenized_prompts)
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + self.prompt_len :, :])  # CLS, EOS

        elif self.prompt_mode == 'class_specific': 
            # only class_specific prompt can be incrementally added.(Origin COOP do not support incremental train)
            old_prompt_tokens = self.new_prompt_tokens.data if self.old_prompt_tokens is None else torch.cat([self.old_prompt_tokens, self.new_prompt_tokens.data])
            self.old_prompt_tokens = nn.Parameter(old_prompt_tokens, requires_grad=False)
            self.new_prompt_tokens = nn.Parameter(new_prompt_tokens, requires_grad=True)

            self.tokenized_prompts = torch.cat([self.tokenized_prompts, tokenized_prompts.to(self.tokenized_prompts.device)])
            self.token_prefix = torch.cat([self.token_prefix, embedding[:, :1, :].to(self.token_prefix.device)])
            self.token_suffix = torch.cat([self.token_suffix, embedding[:, 1 + self.prompt_len :, :].to(self.token_suffix.device)])

    def forward(self):
        prompt_tokens = self.new_prompt_tokens if self.old_prompt_tokens is None else torch.cat([self.old_prompt_tokens, self.new_prompt_tokens])
        
        if self.class_token_position == 'end':
            prompts = torch.cat(
                [
                    self.token_prefix,  # (n_cls, 1, dim)
                    prompt_tokens,     # (n_cls, n_ctx, dim)
                    self.token_suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        elif self.class_token_position == 'middle':
            half_prompt_len = self.prompt_len // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = self.token_prefix[i : i + 1, :, :]
                class_i = self.token_suffix[i : i + 1, :name_len, :]
                suffix_i = self.token_suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = prompt_tokens[i : i + 1, :half_prompt_len, :]
                ctx_i_half2 = prompt_tokens[i : i + 1, half_prompt_len:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = self.token_prefix[i : i + 1, :, :]
                class_i = self.token_suffix[i : i + 1, :name_len, :]
                suffix_i = self.token_suffix[i : i + 1, name_len:, :]
                ctx_i = prompt_tokens[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError
        return self.tokenized_prompts, prompts


class AttriCLIP_Prompt(nn.Module):
    def __init__(self, logger, clip_model, prompt_len, ):
        super().__init__()
        self._logger = logger
        self.dtype = clip_model.dtype
        self.embed_dim = clip_model.embed_dim
        self.prompt_len = prompt_len
        self.token_embedding = clip_model.token_embedding

        # self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = []

        self.new_prompt_tokens = None
        self.old_prompt_tokens = None
    
    def update_new_classes(self, new_class_names):
        self._logger.info("Initializing new class prompts")
        class_names = [name.replace("_", " ") for name in new_class_names]
        name_lens = [len(_tokenizer.encode(name)) for name in class_names]

        prompt_prefix = " ".join(["X"] * self.prompt_len) # "X" will be replaced by learnable prompt afterwards
        prompts = [prompt_prefix + " " + name + "." for name in class_names]
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.token_embedding.weight.device)
        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts).type(self.dtype)

        new_prompt_tokens = torch.empty(len(new_class_names), self.prompt_len, self.embed_dim, dtype=self.dtype)
        nn.init.normal_(new_prompt_tokens, std=0.02)

        if self.new_prompt_tokens is None:
            self.new_prompt_tokens = nn.Parameter(new_prompt_tokens, requires_grad=True)  # to be optimized
            
            self.register_buffer('tokenized_prompts', tokenized_prompts)
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + self.prompt_len :, :])  # CLS, EOS
        else:
            old_prompt_tokens = self.new_prompt_tokens.data if self.old_prompt_tokens is None else torch.cat([self.old_prompt_tokens, self.new_prompt_tokens.data])
            self.old_prompt_tokens = nn.Parameter(old_prompt_tokens, requires_grad=False)
            self.new_prompt_tokens = nn.Parameter(new_prompt_tokens, requires_grad=True)

            self.tokenized_prompts = torch.cat([self.tokenized_prompts, tokenized_prompts.to(self.tokenized_prompts.device)])
            self.token_prefix = torch.cat([self.token_prefix, embedding[:, :1, :].to(self.token_prefix.device)])
            self.token_suffix = torch.cat([self.token_suffix, embedding[:, 1 + self.prompt_len :, :].to(self.token_suffix.device)])

    def forward(self):
        prompt_tokens = self.new_prompt_tokens if self.old_prompt_tokens is None else torch.cat([self.old_prompt_tokens, self.new_prompt_tokens])
        # class tokens are put into the end of the sequence
        prompts = torch.cat(
            [
                self.token_prefix,  # (n_cls, 1, dim)
                prompt_tokens,     # (n_cls, n_ctx, dim)
                self.token_suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return self.tokenized_prompts, prompts

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.embed_dim = clip_model.transformer.width
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, tokenized_prompts, prompts=None):
        if prompts is not None:
            x = prompts + self.positional_embedding.type(self.dtype)
        else:
            x = tokenized_prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class CLIPSZoo(nn.Module):
    def __init__(self, logger, backbone_type, layer_names: Iterable[str]=[]):
        super().__init__()
        self._logger = logger
        self._layer_names = layer_names

        if backbone_type == 'resnet50_clip':
            clip_model = load_clip_to_cpu('RN50')
        elif backbone_type == 'resnet101_clip':
            clip_model = load_clip_to_cpu('RN101')
        elif backbone_type == 'vit_base_patch16_224_clip':
            clip_model = load_clip_to_cpu('ViT-B/16')
        elif backbone_type == 'vit_base_patch32_224_clip':
            clip_model = load_clip_to_cpu('ViT-B/32')

        self.prompt_learner = None
        self.image_encoder = clip_model.visual
        self.token_embedding = clip_model.token_embedding
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.feat_dim = self.image_encoder.output_dim
        self.embed_dim = self.text_encoder.embed_dim

        self.output_features = {}
        self.task_sizes = []
    
    def set_prompt_module(self, prompt_module):
        self.prompt_learner = prompt_module

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        tokenized_prompts, prompts = self.prompt_learner() # 含可学习参数
        text_features = self.text_encoder(tokenized_prompts, prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.output_features['img_features'] = image_features
        self.output_features['text_features'] = text_features

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, self.output_features

    def update_known_class_name(self, class_names):
        self.task_sizes.append(len(class_names))
        self.prompt_learner.update_new_classes(class_names)
    
    def freeze_FE(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.image_encoder.eval()

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()

        self.logit_scale.requires_grad = False

        self._logger.info("Freezing clip's feature extractor(requires_grad=False) ...")
        return self

