import numpy as np
import os
import copy
import torch
from argparse import ArgumentParser
import torch.nn.functional as F
from scipy.spatial.distance import cdist

from methods.multi_steps.finetune_il import Finetune_IL
from backbone.super_class_net import SuperClassNet
from utils.toolkit import count_parameters, tensor2numpy

EPSILON = 1e-8

def add_special_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--T', type=float, default=None, help='tempreture apply to the output logits befor softmax')
    parser.add_argument('--theta', type=float, default=None, help='hyper-parameter to control distance between clusters')
    parser.add_argument('--cluster_metric', type=str, default=None, help='metirc to evaluate distance between clusters')
    return parser

class SuperClass(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._old_network = None
        self._T = config.T
        self._theta = config.theta
        self._cluster_metric = config.cluster_metric

        self._class_prototypes = None
        self._cluster_prototype, self._cluster_contain = [], []
        self._theshold = None
            
    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = SuperClassNet(self._logger, self._config.backbone, self._config.pretrained, self._config.pretrain_path)
        
        self._network.freeze_FE()

        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        self._network = self._network.cuda()
    
    def after_task(self):
        super().after_task()
        self._old_network = self._network.copy().freeze()
        for cluster_id, cluster in enumerate(self._cluster_contain):
            self._logger.info('Cluster {}: {}'.format(cluster_id, cluster))

    def incremental_train(self):
        self._logger.info('-'*10 + ' Learning on task {}: {}-{} '.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)
        optimizer = self._get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()), self._config, self._cur_task==0)
        scheduler = self._get_scheduler(optimizer, self._config, self._cur_task==0)
        if self._cur_task == 0:
            epochs = self._init_epochs
        else:
            epochs = self._epochs
        self._network = self._train_model(self._network, self._train_loader, self._test_loader, optimizer, scheduler, task_id=self._cur_task, epochs=epochs)

        new_class_prototypes = self._class_prototypes[self._known_classes:self._total_classes].cpu().numpy()
        if self.cur_taskID == 0:
            self._cluster_prototype, self._cluster_contain, self._theshold = self.init_clusters(new_class_prototypes, np.arange(self._total_classes), self._theta, self._cluster_metric)
            class_idxes = np.delete(np.arange(self._total_classes), [self._cluster_contain[0][0], self._cluster_contain[1][0]])
            self._cluster_prototype, self._cluster_contain = self.clustering(self._cluster_prototype, self._cluster_contain, new_class_prototypes[class_idxes], self._theshold, class_idxes, self._cluster_metric)
        else:
            self._cluster_prototype, self._cluster_contain = self.clustering(self._cluster_prototype, self._cluster_contain, new_class_prototypes, self._theshold, np.arange(self._known_classes, self._total_classes), self._cluster_metric)

    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        correct, total = 0, 0
        model.train()

        cnn_logits_all = []
        target_all = []

        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, feature_outputs = model(inputs)
            
            cnn_logits_all.append(logits)
            target_all.append(targets)

            total += len(targets)

        cnn_logits_all = torch.cat(cnn_logits_all, dim=0)
        target_all = torch.cat(target_all, dim=0)

        normed_vectors = F.normalize(cnn_logits_all, dim=1)
        class_means = [self._class_prototypes] if self._class_prototypes is not None else []
        for class_id in range(task_begin, task_end):
            class_data_idx = torch.argwhere(target_all == class_id).squeeze()
            class_mean = normed_vectors[class_data_idx].mean(dim=0, keepdim=True)
            class_means.append(F.normalize(class_mean, dim=1))
        
        self._class_prototypes = torch.cat(class_means, dim=0)

        # 仅计算task内部的acc
        dists = torch.cdist(normed_vectors, self._class_prototypes[task_begin:task_end], p=2)
        cnn_predict = torch.argmin(dists, dim=1) + task_begin
        correct = (cnn_predict == target_all).sum()
        
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', 0.0]
        return model, train_acc, train_loss

    def _epoch_test(self, model, test_loader, ret_task_acc=False, ret_pred_target=False, task_begin=None, task_end=None, task_id=None):
        cnn_correct, cnn_task_correct, total, task_total = 0, 0, 0, 0
        cnn_pred_all, nme_pred_all, target_all, features_all = [], [], [], []
        cnn_max_scores_all, nme_max_scores_all = [], []
        model.eval()
        
        if ret_pred_target:
            cluster_prototype = np.stack(self._cluster_prototype, axis=0)

        for _, inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, feature_outputs = model(inputs)
                
            if ret_pred_target:
                # 返回预测结果时，将返回划分簇的结果，同时修改类别标签为簇标签
                features = feature_outputs['features'].detach().cpu().numpy()
                # scores = features.dot(cluster_prototype.T)
                scores = cdist(features, cluster_prototype, metric=self._cluster_metric)
                cnn_max_scores = np.min(scores, axis=1)
                cnn_preds = np.argmin(scores, axis=1)
                cnn_pred_all.append(cnn_preds)
                target_all.append(tensor2numpy(targets)) # 待重新映射为簇标签
                features_all.append(tensor2numpy(feature_outputs['features']))
                cnn_max_scores_all.append(cnn_max_scores)
            else:
                normed_vectors = F.normalize(logits, dim=1)
                dists = torch.cdist(normed_vectors, self._class_prototypes, p=2)
                # cnn_max_scores, cnn_preds = torch.max(dists, dim=-1)
                cnn_max_scores, cnn_preds = torch.min(dists, dim=-1)
                cnn_correct += cnn_preds.eq(targets).cpu().sum()
                total += len(targets)
        
        if ret_pred_target:
            cnn_pred_all = np.concatenate(cnn_pred_all)
            cnn_max_scores_all = np.concatenate(cnn_max_scores_all)
            target_all = np.concatenate(target_all)
            features_all = np.concatenate(features_all)
            
            cluster_target = copy.deepcopy(target_all)
            for cluster_id, cluster in enumerate(self._cluster_contain):
                for class_id in cluster:
                    class_data_idx = np.where(target_all==class_id)
                    cluster_target[class_data_idx] = cluster_id

            return cnn_pred_all, None, cnn_max_scores_all, None, cluster_target, features_all
        else:
            test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
            if ret_task_acc:
                # test_task_acc = np.around(tensor2numpy(cnn_task_correct)*100 / task_total, decimals=2)
                # return test_acc, test_task_acc
                return test_acc, 0.0
            else:
                return test_acc

    def init_clusters(self, data, class_names, theta, metric):
        cluster_prototype, cluster_contain = [], []
        sim = cdist(data, data, metric=metric)
        # sim = data.dot(data.T)
        # sim = 1 - np.exp(sim) / np.sum(np.exp(sim), axis=1, keepdims=True)
        theshold = np.max(sim) * theta
        z1, z2 = divmod(np.argmax(sim), sim.shape[0])
        cluster_prototype.append(data[z1])
        cluster_prototype.append(data[z2])
        cluster_contain.append([class_names[z1]])
        cluster_contain.append([class_names[z2]])
        return cluster_prototype, cluster_contain, theshold

    def clustering(self, cluster_prototype, cluster_contain, data, theshold, class_names, metric):
        idxs = np.arange(data.shape[0])
        while len(idxs) > 0:
            clusters = np.stack(cluster_prototype, axis=0)
            # 计算样本与已知簇中心的距离
            sim = cdist(data[idxs], clusters, metric=metric)  # [len(idxs), len(clusters)]
            # sim = data[idxs].dot(clusters.T)
            # sim = 1 - np.exp(sim) / np.sum(np.exp(sim), axis=1, keepdims=True)
            # 对每个样本，选出距离其最近的簇，并获知其最小的值
            min_sim = np.min(sim, axis=1)   # [len(idxs)]
            min_sim_idxs = np.argmin(sim, axis=1)  # [len(idxs)]
            # 从这些最小值中，选出最大的
            max_sim = np.max(min_sim)   # [1]
            max_sim_idxs = np.argmax(min_sim)   # [1]
            # 若这个余弦相似度最小值大于阈值，则将其定义为新的簇中心，否则将其加入已有簇中
            if max_sim > theshold:
                cluster_prototype.append(data[idxs][max_sim_idxs])
                cluster_contain.append([class_names[idxs][max_sim_idxs]])
            elif class_names[idxs][max_sim_idxs] not in cluster_contain[min_sim_idxs[max_sim_idxs]]:
                # 通过平均的方式动态更新簇中心
                old_cluster_size = len(cluster_contain[min_sim_idxs[max_sim_idxs]])
                cluster_prototype[min_sim_idxs[max_sim_idxs]] = (cluster_prototype[min_sim_idxs[max_sim_idxs]] * old_cluster_size + data[idxs][max_sim_idxs]) / (old_cluster_size + 1)
                cluster_contain[min_sim_idxs[max_sim_idxs]].append(class_names[idxs][max_sim_idxs])
            
            idxs = np.delete(idxs, max_sim_idxs)

        return cluster_prototype, cluster_contain

    def save_predict_records(self, cnn_pred, cnn_pred_scores, nme_pred, nme_pred_scores, targets, features):
        record_dict = {}
        record_dict['cnn_pred'] = cnn_pred
        record_dict['cnn_pred_scores'] = cnn_pred_scores
        record_dict['nme_pred'] = nme_pred
        record_dict['nme_pred_scores'] = nme_pred_scores
        record_dict['targets'] = targets
        record_dict['features'] = features
        record_dict['class_prototypes'] = self._class_prototypes
        record_dict['cluster_contain'] = self._cluster_contain
        record_dict['cluster_prototype'] = self._cluster_prototype

        filename = 'pred_record_seed{}_task{}.npy'.format(self._seed, self._cur_task)
        np.save(os.path.join(self._logdir, filename), record_dict)
    
