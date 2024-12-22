import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from torch.utils.data import DataLoader


class clientKTL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mu = args.mu
        self.ETF_dim = args.num_classes

        self.m = 0.5
        self.s = 64

        self.classes_ids_tensor = torch.tensor(list(range(self.num_classes)), 
                                               dtype=torch.int64, device=self.device)
        self.MSEloss = nn.MSELoss()


    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        ETF = load_item('Server', 'ETF', self.save_folder_name)
        ETF = F.normalize(ETF.T)

        data_generated = load_item('Server', 'data_generated', self.save_folder_name)
        if data_generated is not None:
            gen_loader = DataLoader(data_generated, self.batch_size, drop_last=False, shuffle=True)
            gen_iter = iter(gen_loader)
        proj_fc = load_item('Server', 'proj_fc', self.save_folder_name)

        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        opt_proj_fc = torch.optim.SGD(proj_fc.parameters(), lr=self.learning_rate)
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
                proj = model(x)
                proj = F.normalize(proj)
                cosine = F.linear(proj, ETF)

                # ArcFace loss
                one_hot = F.one_hot(y, self.num_classes).to(self.device)
                arccos = torch.acos(cosine)
                cosine_new = torch.cos(arccos + self.m)
                cosine = one_hot * cosine_new + (1 - one_hot) * cosine
                cosine = cosine * self.s
                loss = self.loss(cosine, y)

                # knowledge transfer
                if data_generated is not None:
                    try:
                        (x_G, y_G) = next(gen_iter)
                    except StopIteration:
                        gen_iter = iter(gen_loader)
                        (x_G, y_G) = next(gen_iter)
            
                    if type(x_G) == type([]):
                        x_G[0] = x_G[0].to(self.device)
                    else:
                        x_G = x_G.to(self.device)
                    y_G = y_G.to(self.device)

                    rep_G = model.base(x_G)
                    proj_G = proj_fc(rep_G)
                    
                    loss += self.MSEloss(proj_G, y_G) * self.mu

                optimizer.zero_grad()
                opt_proj_fc.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
                torch.nn.utils.clip_grad_norm_(proj_fc.parameters(), 100)
                optimizer.step()
                opt_proj_fc.step()

        save_item(model, self.role, 'model', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        ETF = load_item('Server', 'ETF', self.save_folder_name)
        ETF = F.normalize(ETF.T)
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
                proj = model(x)
                cosine = F.linear(F.normalize(proj), ETF)

                test_acc += (torch.sum(torch.argmax(cosine, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(cosine.detach().cpu().numpy())
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

    def collect_protos(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)

        model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                proj = model(x)
                proj = F.normalize(proj)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(proj[i, :].detach().data)

        save_item(agg_func(protos), self.role, 'protos', self.save_folder_name)


def agg_func(protos):
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = F.normalize(proto / len(proto_list), dim=0)
        else:
            protos[label] = F.normalize(proto_list[0], dim=0)

    return protos