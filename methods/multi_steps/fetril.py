import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from sklearn.svm import LinearSVC

from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import target2onehot, tensor2numpy

EPSILON = 1e-8
# 需要拿一半的类别进行初始化，与其他方法的设置不一致

class FeTrIL(Finetune_IL):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self.args = config
        self._means = []
        self._svm_accs = []  # svm_acc :

    def prepare_task_data(self, data_manager):
        self.data_manager = data_manager
        super().prepare_task_data(data_manager)

    def prepare_model(self, checkpoint=None):
        super().prepare_model(checkpoint)
        if self._cur_task > 0:
            self._network.freeze_FE()
        
    def incremental_train(self):
        self._logger.info('-'*10 + ' Learning on task {}: {}-{} '.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)
        
        if self._cur_task == 0:
            optimizer = self._get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()), self._config, self._cur_task==0)
            scheduler = self._get_scheduler(optimizer, self._config, self._cur_task==0)
            epochs = self._init_epochs
            self._network = self._train_model(self._network, self._train_loader, self._test_loader, optimizer, scheduler, task_id=self._cur_task, epochs=epochs)
            self._compute_means()
            self._build_feature_set()
        else:
            epochs = self._epochs
            self._compute_means()
            self._compute_relations()
            self._build_feature_set()
            train_loader = DataLoader(self._feature_trainset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers, pin_memory=True)
            optimizer = self._get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()), self._config, self._cur_task==0)
            scheduler = self._get_scheduler(optimizer, self._config, self._cur_task==0)
            self._network = self._train_model(self._network, train_loader, self._test_loader, optimizer, scheduler, task_id=self._cur_task, epochs=epochs)
        self._train_svm(self._feature_trainset, self._feature_testset)
            
    # 计算类的prototype  
    def _compute_means(self):
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes): 
                idx_dataset = self.data_manager.get_dataset(source='train', mode='test', 
                                                            indices = np.arange(class_idx, class_idx+1), ret_data=False)
                idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                self._means.append(class_mean)        
            
    def _compute_relations(self):
        old_means = np.array(self._means[:self._known_classes])
        new_means = np.array(self._means[self._known_classes:])
        self._relations=np.argmax((old_means/np.linalg.norm(old_means,axis=1)[:,None])@(new_means/np.linalg.norm(new_means,axis=1)[:,None]).T,axis=1)+self._known_classes

    def _build_feature_set(self):
        self.vectors_train = []
        self.labels_train = []
        for class_idx in range(self._known_classes, self._total_classes):
            idx_dataset = self.data_manager.get_dataset(source='train',mode='test', 
                                                        indices= np.arange(class_idx, class_idx+1), ret_data=False)
            idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            self.vectors_train.append(vectors)
            self.labels_train.append([class_idx]*len(vectors))
        for class_idx in range(0,self._known_classes):
            new_idx = self._relations[class_idx]
            self.vectors_train.append(self.vectors_train[new_idx-self._known_classes]-self._means[new_idx]+self._means[class_idx])
            self.labels_train.append([class_idx]*len(self.vectors_train[-1]))
        
        self.vectors_train = np.concatenate(self.vectors_train)
        self.labels_train = np.concatenate(self.labels_train)
        self._feature_trainset = FeatureDataset(self.vectors_train,self.labels_train)
        
        self.vectors_test = []
        self.labels_test = []
        for class_idx in range(0, self._total_classes):
            idx_dataset = self.data_manager.get_dataset(source='test',mode='test', 
                                                        indices = np.arange(class_idx, class_idx+1), ret_data=False)
            idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            self.vectors_test.append(vectors)
            self.labels_test.append([class_idx]*len(vectors))
        self.vectors_test = np.concatenate(self.vectors_test)
        self.labels_test = np.concatenate(self.labels_test)

        self._feature_testset = FeatureDataset(self.vectors_test,self.labels_test)

    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        if self._cur_task == 0:
            model.train()
        else:
            model.eval()
        losses = 0.
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            if self._cur_task ==0:
                logits = model(inputs)[0]
            else:
                logits = model.fc(inputs)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct += preds.eq(targets.expand_as(preds)).cpu().sum()
            total += len(targets)
        if scheduler != None:
            scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader)]
        return model, train_acc, train_loss

    def _train_svm(self, train_set, test_set):
        train_features = train_set.features.numpy()
        train_labels = train_set.labels.numpy()
        test_features = test_set.features.numpy()
        test_labels = test_set.labels.numpy()
        train_features = train_features/np.linalg.norm(train_features,axis=1)[:,None]
        test_features = test_features/np.linalg.norm(test_features,axis=1)[:,None]
        svm_classifier = LinearSVC(random_state=42)
        svm_classifier.fit(train_features,train_labels)
        self._logger.info("svm train: acc: {}".format(np.around(svm_classifier.score(train_features,train_labels)*100,decimals=2)))
        acc = svm_classifier.score(test_features,test_labels)
        self._svm_accs.append(np.around(acc*100,decimals=2))
        self._logger.info("svm evaluation: acc_list: {}".format(self._svm_accs))

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_features(_inputs.cuda())
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_features(_inputs.cuda())
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)
    

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        assert len(features) == len(labels), "Data size error!"
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return idx, feature, label
