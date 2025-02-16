import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client, load_item, save_item
from sklearn.preprocessing import label_binarize
from sklearn import metrics


class clientMRL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.sub_feature_dim = args.sub_feature_dim
        
        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            proj = nn.Linear(args.feature_dim + args.sub_feature_dim, args.feature_dim).to(self.device)
            save_item(proj, self.role, 'proj', self.save_folder_name)

    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        proj = load_item(self.role, 'proj', self.save_folder_name)
        global_model = load_item('Server', 'global_model', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        optimizer_p = torch.optim.SGD(proj.parameters(), lr=self.learning_rate)
        optimizer_g = torch.optim.SGD(global_model.parameters(), lr=self.learning_rate)
        # model.to(self.device)
        model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep_g = global_model.base(x)
                rep = model.base(x)
                rep_concat = torch.concat((rep_g, rep), dim=1)
                rep_new = proj(rep_concat)
                output_g = global_model.head(rep_new[:, :self.sub_feature_dim])
                output = model.head(rep_new)
                loss = self.loss(output, y) + self.loss(output_g, y)

                optimizer.zero_grad()
                optimizer_p.zero_grad()
                optimizer_g.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_p.step()
                optimizer_g.step()

        # model.cpu()
        save_item(model, self.role, 'model', self.save_folder_name)
        save_item(global_model, self.role, 'global_model', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def test_metrics(self):
        testloaderfull = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        proj = load_item(self.role, 'proj', self.save_folder_name)
        global_model = load_item('Server', 'global_model', self.save_folder_name)
        # model.to(self.device)
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep_g = global_model.base(x)
                rep = model.base(x)
                rep_concat = torch.concat((rep_g, rep), dim=1)
                rep_new = proj(rep_concat)
                output = model.head(rep_new)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc
    
    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        proj = load_item(self.role, 'proj', self.save_folder_name)
        global_model = load_item('Server', 'global_model', self.save_folder_name)
        # model.to(self.device)
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)
                rep_g = global_model.base(x)
                rep = model.base(x)
                rep_cat = torch.concat((rep_g, rep), dim=1)
                rep_new = proj(rep_cat)
                output = model.head(rep_new)
                output_g = global_model.head(rep_new[:, :self.sub_feature_dim])
                loss = self.loss(output, y) + self.loss(output_g, y)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num