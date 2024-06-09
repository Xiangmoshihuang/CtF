import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict

from backbone.inc_net import load_clip_to_cpu
from backbone.clip import tokenize
from backbone.clip.model import MultiheadAttention, QuickGELU
from backbone.clip_zoo import TextEncoder


class ITFusionBlock(nn.Module):
    def __init__(self, embed_dim, n_head=8, hiden_dim=16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.hidden_dim = hiden_dim

        self.att = MultiheadAttention(embed_dim=embed_dim, num_heads=n_head, is_cross_attention=False)
        self.ln_1 = nn.LayerNorm(embed_dim)

        self.cross_att = MultiheadAttention(embed_dim=embed_dim, num_heads=n_head, is_cross_attention=True)
        self.ln_2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(embed_dim, hiden_dim)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(hiden_dim, embed_dim))
        ]))

    def forward(self, img_inputs, text_inputs=None):
        it_out = img_inputs + self.att(img_inputs)
        it_out = self.ln_1(it_out)

        if text_inputs is not None:
            it_out = text_inputs + self.cross_att(it_out, text_inputs, text_inputs)
            it_out = self.ln_2(it_out)
        
        it_out = it_out + self.mlp(it_out)

        return it_out

class ITFusionLayer(nn.Module):
    def __init__(self, n_blocks, embed_dim, n_head=8, hiden_dim=16) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.hidden_dim = hiden_dim
    
        self.it_blocks = nn.ModuleList([ITFusionBlock(embed_dim=embed_dim, n_head=n_head, hiden_dim=hiden_dim) for i in range(n_blocks)])


class FusionNet(nn.Module):
    def __init__(self, logger, backbone_type, context:str, mode:str):
        super().__init__()
        self._logger = logger
        self._context = context
        self._mode = mode

        if backbone_type == 'resnet50_clip':
            clip_model = load_clip_to_cpu('RN50')
        elif backbone_type == 'resnet101_clip':
            clip_model = load_clip_to_cpu('RN101')
        elif backbone_type == 'vit_base_patch16_224_clip':
            clip_model = load_clip_to_cpu('ViT-B/16')
        elif backbone_type == 'vit_base_patch32_224_clip':
            clip_model = load_clip_to_cpu('ViT-B/32')

        self.image_encoder = clip_model.visual
        self.token_embedding = clip_model.token_embedding
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.feat_dim = self.image_encoder.output_dim
        self.embed_dim = self.text_encoder.embed_dim

        self._itm_heads = nn.ModuleList([])
        self._task_heads = nn.ModuleList([])
        self._fusion_modules = nn.ModuleList([])

        self.task_sizes = []
        self.known_class_names = []

        self._tokenized_inputs = None
        self._text_inputs = None
        self.output_features = {}    

    def forward_clip(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        text_features = self.text_encoder(self._tokenized_inputs, self._text_inputs)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, self.output_features

    def update_new_class_name(self, new_class_names):
        self.task_sizes.append(len(new_class_names))
        self.known_class_names.extend(new_class_names)

        class_names = [name.replace("_", " ") for name in new_class_names]
        prompts = [self._context.format(name) for name in class_names]
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.token_embedding.weight.device)
        self._tokenized_inputs = tokenized_prompts if self._tokenized_inputs is None else torch.cat([self._tokenized_inputs, tokenized_prompts])

        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts).type(self.dtype)
        self._text_inputs = embedding if self._text_inputs is None else torch.cat([self._text_inputs, embedding])

        self._itm_heads.append(nn.Linear(self.embed_dim, 2))
        self._task_heads.append(nn.Linear(self.embed_dim, len(new_class_names)))
        self._fusion_modules.append()
    
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

