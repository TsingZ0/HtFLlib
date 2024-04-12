import copy
import random
import time

import numpy as np
from flcore.clients.clientkd import clientKD, recover, decomposition
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from flcore.trainmodel.models import BaseHeadSplit


class FedKD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            global_model = BaseHeadSplit(args, 0).to(args.device)            
            save_item(global_model, self.role, 'global_model', self.save_folder_name)
        
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientKD)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.T_start = args.T_start
        self.T_end = args.T_end
        self.energy = self.T_start


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_ids()
            self.aggregate_parameters()

            self.send_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

            self.energy = self.T_start + ((1 + i) / self.global_rounds) * (self.T_end - self.T_start)
            for client in self.clients:
                client.energy = self.energy

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()

        
    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)

        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        global_param = {name: param.detach().cpu().numpy() 
                        for name, param in global_model.named_parameters()}
        for k in global_param.keys():
            global_param[k] = np.zeros_like(global_param[k])
            
        for cid in self.uploaded_ids:
            client = self.clients[cid]
            compressed_param = load_item(client.role, 'compressed_param', client.save_folder_name)
            client_param = recover(compressed_param)
            for server_k, client_k in zip(global_param.keys(), client_param.keys()):
                global_param[server_k] += client_param[client_k] * 1/len(self.uploaded_ids)

        compressed_param = decomposition(global_param.items(), self.energy)
        save_item(compressed_param, self.role, 'compressed_param', self.save_folder_name)