from torch import nn
import copy
from backbone.inc_net import IncrementalNet


class SuperClassNet(IncrementalNet):

    def generate_fc(self, in_dim, out_dim):
        return nn.Linear(in_dim, out_dim, bias=False)

    def update_fc(self, nb_classes):
        self.task_sizes.append(nb_classes - sum(self.task_sizes))
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if not isinstance(self.fc, nn.Identity):
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.weight.data[:nb_output] = weight
            self._logger.info('Updated classifier head output dim from {} to {}'.format(nb_output, nb_classes))
        else:
            self._logger.info('Created classifier head with output dim {}'.format(nb_classes))
        del self.fc
        self.fc = fc
