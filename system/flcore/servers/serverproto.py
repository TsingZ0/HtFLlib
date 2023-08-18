import time
import numpy as np
from flcore.clients.clientproto import clientProto
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from utils.data_utils import read_client_data
from threading import Thread
from collections import defaultdict


class FedProto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientProto)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        self.rs_test_acc_cla = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_protos()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc, self.rs_test_acc_cla], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(max(self.rs_test_acc_cla))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            protos = load_item(client.role, 'protos', client.save_folder_name)
            uploaded_protos.append(protos)
            
        global_protos = proto_aggregation(uploaded_protos)
        save_item(global_protos, self.role, 'global_protos', self.save_folder_name)

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_correct1 = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct[0]*1.0)
            tot_correct1.append(ct[1]*1.0)
            print(f'Client {c.id}: Acc: {ct[0]*1.0/ns}, AUC: {auc}')
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, (tot_correct, tot_correct1), tot_auc

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        # stats_train = self.train_metrics()

        test_acc = sum(stats[2][0])*1.0 / sum(stats[1])
        test_acc1 = sum(stats[2][1])*1.0 / sum(stats[1])
        # train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2][0], stats[1])]
        accs1 = [a / n for a, n in zip(stats[2][1], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
            self.rs_test_acc_cla.append(test_acc1)
        else:
            acc.append(test_acc)
        
        # if loss == None:
        #     self.rs_train_loss.append(train_loss)
        # else:
        #     loss.append(train_loss)

        # print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test Accurancy of Classifier: {:.4f}".format(test_acc1))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test Accurancy of Classifier: {:.4f}".format(np.std(accs1)))
    

# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221
def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label