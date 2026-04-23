import torch
import torch.nn as nn
import numpy as np
import time
import higher
from flcore.clients.clientbase import Client, load_item, save_item
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data
from collections import defaultdict
from torch.backends.cuda import sdp_kernel


class clientL2G(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        self.meta_study_batches = args.meta_study_batches
        self.meta_quiz_batches = args.meta_quiz_batches
        self.num_quiz = args.meta_quiz_batches * self.batch_size

    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        guiding_vec = load_item('Server', 'guiding_vec', self.save_folder_name)
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
                feat = model.base(x)
                output = model.head(feat)
                ce_loss = self.loss(output, y) 
                guiding_loss = self.loss_mse(feat, guiding_vec(y))
                loss = ce_loss + guiding_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
                optimizer.step()

        save_item(model, self.role, 'model', self.save_folder_name)
        
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)[self.num_quiz:]
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)
        
    def meta_study(self):
        total_train_data = read_client_data(self.dataset, self.id, is_train=True)
        meta_study_loader = DataLoader(
            total_train_data[self.num_quiz:], self.batch_size, drop_last=True, shuffle=True)
        meta_quiz_loader = DataLoader(
            total_train_data[:self.num_quiz], self.batch_size, drop_last=True, shuffle=True)
        model = load_item(self.role, 'model', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        guiding_vec = load_item('Server', 'guiding_vec', self.save_folder_name)
        opt_vec = torch.optim.SGD(guiding_vec.parameters(), lr=0)
        # model.to(self.device)
        model.train()

        with sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
            with higher.innerloop_ctx(model, optimizer) as (fmodel, diffopt):
                pseudo_loss = 0
                meta_loss = 0
                for i, (x, y) in enumerate(meta_study_loader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if i < self.meta_study_batches:
                        # with torch.backends.cudnn.flags(enabled=False):
                        #     feat = fmodel.base(x)
                        #     output = fmodel.head(feat)
                        feat = fmodel.base(x)
                        output = fmodel.head(feat)
                        ce_loss = self.loss(output, y) 
                        guiding_loss = self.loss_mse(feat, guiding_vec(y))
                        pseudo_loss = pseudo_loss + ce_loss + guiding_loss
                    else:
                        break
                diffopt.step(pseudo_loss/self.meta_study_batches)
                for i, (x, y) in enumerate(meta_quiz_loader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if i < self.meta_quiz_batches:
                        # with torch.backends.cudnn.flags(enabled=False):
                        #     output = fmodel(x)
                        output = fmodel(x)
                        meta_loss = meta_loss + self.loss(output, y)
                    else:
                        break
                
                opt_vec.zero_grad()
                grads_ = torch.autograd.grad(
                    meta_loss/self.meta_quiz_batches, 
                    guiding_vec.parameters(), 
                )
                grads = defaultdict(list)
                for class_id, grad in enumerate([*grads_[0]]):
                    if torch.sum(abs(grad)) > 0:
                        grads[class_id] = grad
                save_item(grads, self.role, 'grads', self.save_folder_name)
