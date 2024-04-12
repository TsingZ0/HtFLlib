import copy
import random
import time
from flcore.clients.clientfml import clientFML
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from flcore.trainmodel.models import BaseHeadSplit


class FML(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            global_model = BaseHeadSplit(args, 0).to(args.device)            
            save_item(global_model, self.role, 'global_model', self.save_folder_name)
        
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFML)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


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
        
        
    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)

        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        for param in global_model.parameters():
            param.data.zero_()
            
        for cid in self.uploaded_ids:
            client = self.clients[cid]
            client_model = load_item(client.role, 'global_model', client.save_folder_name)
            for server_param, client_param in zip(global_model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * 1/len(self.uploaded_ids)

        save_item(global_model, self.role, 'global_model', self.save_folder_name)
