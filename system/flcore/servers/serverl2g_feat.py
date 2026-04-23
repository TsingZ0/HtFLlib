import time
import random
import torch
import torch.nn as nn
import numpy as np
from flcore.clients.clientl2g_feat import clientL2G
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from collections import defaultdict


class FedL2G(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientL2G)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.warm_up_rounds = args.warm_up_rounds

        # use nn.Embedding for convenient indexing
        self.guiding_vec = nn.Embedding(args.num_classes, args.feature_dim).to(self.device)
        save_item(self.guiding_vec, self.role, 'guiding_vec', self.save_folder_name)
        self.optimizer = torch.optim.SGD(
            self.guiding_vec.parameters(), lr=args.server_learning_rate)
        self.feature_dim = args.feature_dim
        
        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i%self.eval_gap == 0 and i >= self.warm_up_rounds:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()

            for client in self.selected_clients:
                if i >= self.warm_up_rounds:
                    client.train()
                client.meta_study()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]
                
            self.receive_grads()
            self.aggregate_grads()

            if i >= self.warm_up_rounds:
                self.Budget.append(time.time() - s_t)
                print('-'*25, 'time cost', '-'*25, self.Budget[-1])

                if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                    break


        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()


    def receive_grads(self):
        assert (len(self.selected_clients) > 0)
        print('-'*50, 'Receiving client grads ...')

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_gs = []
        for client in active_clients:
            self.uploaded_ids.append(client.id)
            grads = load_item(client.role, 'grads', self.save_folder_name)
            self.uploaded_gs.append(grads)

    def aggregate_grads(self):
        assert (len(self.uploaded_ids) > 0)
        print('-'*50, 'Aggregating client grads ...')

        grads_agg_dict = grads_aggregate(self.uploaded_gs)
        grads_agg = []
        missing_classes = []
        for class_id in range(self.num_classes):
            if len(grads_agg_dict[class_id]) > 0:
                grads_agg.append(grads_agg_dict[class_id])
            else:
                grads_agg.append(torch.zeros(self.feature_dim, device=self.device))
                missing_classes.append(class_id)
        print('Missing classes:', missing_classes)

        self.optimizer.zero_grad()
        self.guiding_vec.weight.grad = torch.stack(grads_agg).detach().clone()
        torch.nn.utils.clip_grad_norm_(self.guiding_vec.parameters(), 100)
        self.optimizer.step()
        
        save_item(self.guiding_vec, self.role, 'guiding_vec', self.save_folder_name)


def grads_aggregate(grads_dict_list):
    grads_agg_dict = defaultdict(list)
    for grads_dict in grads_dict_list:
        for k in grads_dict.keys():
            grads_agg_dict[k].append(grads_dict[k])

    for k in grads_agg_dict.keys():
        grads = torch.stack(grads_agg_dict[k])
        grads_agg_dict[k] = torch.mean(grads, dim=0).detach()
    return grads_agg_dict