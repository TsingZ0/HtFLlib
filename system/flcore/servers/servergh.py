import time
import torch
import torch.nn as nn
from flcore.clients.clientgh import clientGH
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from torch.utils.data import DataLoader


class FedGH(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientGH)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.CEloss = nn.CrossEntropyLoss()
        self.server_learning_rate = args.server_learning_rate
        self.server_epochs = args.server_epochs

        head = load_item(self.clients[0].role, 'model', self.clients[0].save_folder_name).head
        save_item(head, 'Server', 'head', self.save_folder_name)


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_parameters()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()
                client.collect_protos()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_protos()
            self.train_head()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()


    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            protos = load_item(client.role, 'protos', client.save_folder_name)
            for cc in protos.keys():
                y = torch.tensor(cc, dtype=torch.int64, device=self.device)
                uploaded_protos.append((protos[cc], y))
            
        save_item(uploaded_protos, self.role, 'uploaded_protos', self.save_folder_name)
    
    def train_head(self):
        uploaded_protos = load_item(self.role, 'uploaded_protos', self.save_folder_name)
        proto_loader = DataLoader(uploaded_protos, self.batch_size, drop_last=False, shuffle=True)
        head = load_item('Server', 'head', self.save_folder_name)
        
        opt_h = torch.optim.SGD(head.parameters(), lr=self.server_learning_rate)

        for _ in range(self.server_epochs):
            for p, y in proto_loader:
                out = head(p)
                loss = self.CEloss(out, y)
                opt_h.zero_grad()
                loss.backward()
                opt_h.step()

        save_item(head, 'Server', 'head', self.save_folder_name)
