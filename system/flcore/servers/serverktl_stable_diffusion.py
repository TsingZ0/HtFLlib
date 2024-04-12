import pickle
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flcore.clients.clientktl import clientKTL
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from collections import defaultdict
from torch.utils.data import DataLoader


class FedKTL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientKTL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.feature_dim = args.feature_dim
        self.server_learning_rate = args.server_learning_rate
        self.gen_batch_size = args.gen_batch_size
        self.server_batch_size = args.server_batch_size
        self.server_epochs = args.server_epochs
        self.lamda = args.lamda
        self.ETF_dim = args.num_classes
        self.classes_ids_tensor = torch.tensor(list(range(self.num_classes)), 
                                               dtype=torch.int64, device=self.device)
        
        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            trainloader = self.clients[0].load_train_data()
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                self.img_shape = x[0].shape
                break

            from diffusers import StableDiffusionPipeline
            pipe = StableDiffusionPipeline.from_pretrained(
                args.generator_path, 
                custom_pipeline='./stable-diffusion/pipelines/', 
                # torch_dtype=torch.float16, 
            )
            pipe.set_progress_bar_config(disable=True)
            pipe = pipe.to(self.device)

            self.pipe_num_inference_steps = 50
            self.pipe_num_images_per_prompt = 1 # set 1 to reduce GPU memory
            self.pipe_height = pipe.unet.config.sample_size * pipe.vae_scale_factor
            self.pipe_width = pipe.unet.config.sample_size * pipe.vae_scale_factor
            self.pipe_guidance_scale = 7.5
            self.pipe_negative_prompt = None
            self.pipe_eta = 0.0
            self.pipe_generator = None
            self.pipe_latents = None
            self.pipe_prompt_embeds = None
            self.pipe_negative_prompt_embeds = None
            self.pipe_output_type = "pil"
            self.pipe_return_dict = True
            self.pipe_callback = None
            self.pipe_callback_steps = 1
            self.pipe_cross_attention_kwargs = None
            self.pipe_guidance_rescale = 0.0
            self.pipe_device = pipe._execution_device
            self.pipe_do_classifier_free_guidance = self.pipe_guidance_scale > 1.0
            self.pipe_text_encoder_lora_scale = (
                self.pipe_cross_attention_kwargs.get("scale", None) if self.pipe_cross_attention_kwargs is not None else None
            )
            self.pipe_num_channels_latents = pipe.unet.config.in_channels
            self.pipe_prompt = args.stable_diffusion_prompt

            self.pipe_prompt_embeds, self.pipe_negative_prompt_embeds = pipe.encode_prompt(
                self.pipe_prompt, 
                self.pipe_device,
                self.pipe_num_images_per_prompt,
                self.pipe_do_classifier_free_guidance,
                self.pipe_negative_prompt,
                prompt_embeds=self.pipe_prompt_embeds,
                negative_prompt_embeds=self.pipe_negative_prompt_embeds,
                lora_scale=self.pipe_text_encoder_lora_scale,
            )

            latents = pipe.prepare_latents(
                        self.pipe_num_images_per_prompt,
                        self.pipe_num_channels_latents,
                        self.pipe_height,
                        self.pipe_width,
                        self.pipe_prompt_embeds.dtype,
                        self.pipe_device,
                        self.pipe_generator,
                        None,
                    )
            print('latents shape', latents.shape)
            self.pipe_latent_shape = latents.shape
            pipe.latent_dim = latents.view(-1).shape[0]
            print('latents dim', pipe.latent_dim)
            save_item(pipe, self.role, 'pipe', self.save_folder_name)

            F = Feature_Transformer(self.ETF_dim, pipe.latent_dim).to(self.device)
            save_item(F, self.role, 'F', self.save_folder_name)
            print('Feature_Transformer', F)

            Centroids = nn.Embedding(self.num_classes, pipe.latent_dim).to(self.device)
            save_item(Centroids, self.role, 'Centroids', self.save_folder_name)
            print('Centroids', Centroids)

            while True:
                try:
                    P = generate_random_orthogonal_matrix(self.ETF_dim, self.num_classes)
                    I = torch.eye(self.num_classes)
                    one = torch.ones(self.num_classes, self.num_classes)
                    F = np.sqrt(self.num_classes / (self.num_classes-1)) * torch.matmul(P, I-((1/self.num_classes) * one))
                    ETF = F.requires_grad_(False).to(self.device)
                    save_item(ETF, self.role, 'ETF', self.save_folder_name)
                    break
                except AssertionError:
                    pass
            
            clientprocess = transforms.Compose(
                [transforms.Resize(size=self.img_shape[-1]), 
                transforms.CenterCrop(size=(self.img_shape[-1], self.img_shape[-1])), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            save_item(clientprocess, self.role, 'clientprocess', self.save_folder_name)
            print('clientprocess', clientprocess)

            proj_fc = nn.Linear(self.feature_dim, pipe.latent_dim).to(self.device)
            save_item(proj_fc, self.role, 'proj_fc', self.save_folder_name)
        
        self.MSEloss = nn.MSELoss()


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
                client.collect_protos()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_protos()
            self.align()
            self.generate_images(i)

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
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

    @torch.no_grad()
    def set_Centroids(self, uploaded_protos, F, Centroids): # set Centroids to the centroids of latent vectors
        proto_loader = DataLoader(uploaded_protos, self.server_batch_size, drop_last=False, shuffle=True)
        protos = defaultdict(list)
        F.eval()
        for P, y in proto_loader:
            Q = F(P).detach()
            for i, yy in enumerate(y):
                y_c = yy.item()
                protos[y_c].append(Q[i, :].data)
        protos = avg_func(protos)
        for i, weight in enumerate(Centroids.weight):
            if type(protos[i]) != type([]):
                weight.data = protos[i].data.clone()

    def align(self):
        uploaded_protos = load_item(self.role, 'uploaded_protos', self.save_folder_name)
        pipe = load_item(self.role, 'pipe', self.save_folder_name)
        F = load_item(self.role, 'F', self.save_folder_name)
        Centroids = load_item(self.role, 'Centroids', self.save_folder_name)
        self.set_Centroids(uploaded_protos, F, Centroids)

        opt_F = torch.optim.Adam(F.parameters(), 
                                 lr=self.server_learning_rate, 
                                 betas=(0.9, 0.999),
                                 eps=1e-08, 
                                 weight_decay=0, 
                                 amsgrad=False)
        opt_Centroids = torch.optim.Adam(Centroids.parameters(), 
                                    lr=self.server_learning_rate, 
                                    betas=(0.9, 0.999),
                                    eps=1e-08, 
                                    weight_decay=0, 
                                    amsgrad=False)

        print('\n----Server aligning ...----\n')
        F.train()
        Centroids.train()
        for _ in range(self.server_epochs):
            proto_loader = DataLoader(uploaded_protos, self.server_batch_size, drop_last=False, shuffle=True)
            for P, y in proto_loader:
                Q = F(P)
                latents = pipe.prepare_latents(
                            Q.shape[0],
                            self.pipe_num_channels_latents,
                            self.pipe_height,
                            self.pipe_width,
                            self.pipe_prompt_embeds.dtype,
                            self.pipe_device,
                            self.pipe_generator,
                            None,
                        ).view(Q.shape[0], -1)
                loss = MMD(Q, latents, 'rbf', self.device)

                centroids = Centroids(y) # approximate transformed class centroids to reduce computational cost
                loss += self.MSEloss(Q, centroids) * self.lamda

                opt_F.zero_grad()
                opt_Centroids.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(F.parameters(), 100)
                torch.nn.utils.clip_grad_norm_(Centroids.parameters(), 100)
                opt_F.step()
                opt_Centroids.step()

        self.set_Centroids(uploaded_protos, F, Centroids)
        latent_centroids_vec = Centroids(self.classes_ids_tensor).detach().data
        latent_centroids = [torch.squeeze(latent.view(self.pipe_latent_shape)) for latent in latent_centroids_vec]

        save_item(F, self.role, 'F', self.save_folder_name)
        save_item(Centroids, self.role, 'Centroids', self.save_folder_name)
        save_item(latent_centroids, self.role, 'latent_centroids', self.save_folder_name)

    @torch.no_grad()
    def generate_images(self, R):
        print('\n----Server generating ...----\n')
        pipe = load_item(self.role, 'pipe', self.save_folder_name)
        clientprocess = load_item(self.role, 'clientprocess', self.save_folder_name)
        data_generated = []

        latent_centroids = load_item(self.role, 'latent_centroids', self.save_folder_name)
        latents_loader = DataLoader(latent_centroids, self.gen_batch_size, drop_last=False, shuffle=False)
        PIL_transform = transforms.Compose([transforms.PILToTensor()])
        for latents in latents_loader:
            PIL_images = pipe(
                latents=latents, 
                prompt_embeds=torch.tile(self.pipe_prompt_embeds, (latents.shape[0], 1, 1)), 
                negative_prompt_embeds=torch.tile(self.pipe_negative_prompt_embeds, (latents.shape[0], 1, 1)), 
                num_inference_steps=self.pipe_num_inference_steps, 
                num_images_per_prompt=self.pipe_num_images_per_prompt
            ).images
            raw_images = torch.stack([PIL_transform(img) for img in PIL_images]).clamp(0, 255) / 255
            images = clientprocess(raw_images)
            data = [(xx, yy.view(-1)) for xx, yy in zip(images, latents)]
            data_generated.extend(data)

        save_item(data_generated, self.role, 'data_generated', self.save_folder_name)


# https://github.com/NeuralCollapseApplications/ImbalancedLearning/blob/main/models/resnet.py#L347
def generate_random_orthogonal_matrix(feat_in, num_classes):
    a = np.random.random(size=(feat_in, num_classes))
    P, _ = np.linalg.qr(a)
    P = torch.tensor(P).float()
    assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-07), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
    return P


def avg_func(protos):
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos


def MMD(x, y, kernel, device='cpu'):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
    return torch.mean(XX + YY - 2. * XY)


class Feature_Transformer(nn.Module):
    def __init__(self, in_features, out_features, num_layers=2):
        super().__init__()

        layers = []
        for i in range(num_layers-1):
            layers.append(nn.Linear(in_features, in_features))
            layers.append(nn.BatchNorm1d(in_features))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features, out_features))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        out = self.mlp(x)
        return out
    