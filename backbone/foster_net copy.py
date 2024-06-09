import torch
from torch import nn

class FOSTERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(FOSTERNet, self).__init__()
        self.backbone_type = args["backbone_type"]
        self.backbones = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.fe_fc = None
        self.task_sizes = []
        self.oldfc = None
        self.args = args

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.backbones)

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        out = self.fc(features)
        fe_logits = self.fe_fc(features[:, -self.out_dim :])["logits"]

        out.update({"fe_logits": fe_logits, "features": features})

        if self.oldfc is not None:
            old_logits = self.oldfc(features[:, : -self.out_dim])["logits"]
            out.update({"old_logits": old_logits})

        out.update({"eval_logits": out["logits"]})
        return out

    def update_fc(self, nb_classes):
        self.backbones.append(get_backbone(self.args, self.pretrained))
        if self.out_dim is None:
            self.out_dim = self.backbones[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
            self.backbones[-1].load_state_dict(self.backbones[-2].state_dict())

        self.oldfc = self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.fe_fc = self.generate_fc(self.out_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_backbone(self):
        for param in self.backbones.parameters():
            param.requires_grad = False
        self.backbones.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        logging.info("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma
    
    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
                args["dataset"],
                args["seed"],
                args["backbone_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.backbones) == 1
        self.backbones[0].load_state_dict(model_infos['backbone'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc