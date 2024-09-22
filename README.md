# Heterogeneous Federated Learning Library (HtFLlib)
Standard federated learning, e.g., [FedAvg](http://proceedings.mlr.press/v54/mcmahan17a.html), assumes that all the participating clients build their local models with the same architecture, which limits its utility in real-world scenarios. In practice, clients can build their models with ***heterogeneous model architectures*** for specific local tasks. When faced with **data heterogeneity**, **model heterogeneity**, **communication overhead**, and **intellectual property (IP) protection**, Heterogeneous Federated Learning (HtFL) emerges. 

- ***9 data-free HtFL algorithms and 21 heterogeneous model architectures.***
- [PFLlib](https://github.com/TsingZ0/PFLlib) compatible.

## Environments
Install [CUDA v11.6](https://developer.nvidia.com/cuda-11-6-0-download-archive). 

Install [conda latest](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) and activate conda. 

```bash
conda env create -f env_cuda_latest.yaml # You may need to downgrade the torch using pip to match the CUDA version
```

## Scenarios and datasets

Here, we only show the MNIST dataset in the ***label skew*** scenario generated via Dirichlet distribution for example. Please refer to my other repository [PFLlib](https://github.com/TsingZ0/PFLlib) for more help. 

*You can also modify codes in PFLlib to support model heterogeneity scenarios, but it requires much effort. In this repository, you only need to configure `system/main.py` to support model heterogeneity scenarios.*

**Note**: you may need to manually clean checkpoint files in the `temp/` folder via `system/clean_temp_files.py` if your program crashes accidentally. You can also set a checkpoint folder by yourself to prevent automatic deletion using the `-sfn` argument in the command line. 

## Data-free algorithms with code (updating)
Here, "data-free" refers to the absence of any additional dataset beyond the clients' private data. We only consider data-free algorithms here, as they have fewer restrictions and assumptions, making them more valuable and easily extendable to other scenarios, such as the existence of public server data. 

- **Local** — Each client trains its model locally without federation.
- **FedDistill (FD)** — [Communication-Efficient On-Device Machine Learning: Federated Distillation and Augmentation under Non-IID Private Data](https://arxiv.org/pdf/1811.11479.pdf) *2018*
- **FML** — [Federated Mutual Learning](https://arxiv.org/abs/2006.16765) *2020*
- **LG-FedAvg** — [Think Locally, Act Globally: Federated Learning with Local and Global Representations](https://arxiv.org/abs/2001.01523) *2020*
- **FedGen** — [Data-Free Knowledge Distillation for Heterogeneous Federated Learning](http://proceedings.mlr.press/v139/zhu21b.html) *ICML 2021*
- **FedProto** — [FedProto: Federated Prototype Learning across Heterogeneous Clients](https://ojs.aaai.org/index.php/AAAI/article/view/20819) *AAAI 2022* 
- **FedKD** — [Communication-efficient federated learning via knowledge distillation](https://www.nature.com/articles/s41467-022-29763-x) *Nature Communications 2022*
- **FedGH** — [FedGH: Heterogeneous Federated Learning with Generalized Global Header](https://dl.acm.org/doi/10.1145/3581783.3611781) *ACM MM 2023*
- **FedTGP** — [FedTGP: Trainable Global Prototypes with Adaptive-Margin-Enhanced Contrastive Learning for Data and Model Heterogeneity in Federated Learning](https://arxiv.org/abs/2401.03230) *AAAI 2024*
- **FedKTL** — [An Upload-Efficient Scheme for Transferring Knowledge From a Server-Side Pre-trained Generator to Clients in Heterogeneous Federated Learning](https://arxiv.org/abs/2403.15760) *CVPR 2024* *(Note: FedKTL requires pre-trained generators to run, please refer to its [project page](https://github.com/TsingZ0/FedKTL) for download links.)*

## Experimental results

You can run `total.sh` with *pre-tuned hyperparameters* to obtain some results, like
  ```bash
  cd ./system
  sh total.sh
  ```

Or you can find some results in our accepted FL paper (i.e., [FedTGP](https://github.com/TsingZ0/FedTGP) and [FedKTL](https://github.com/TsingZ0/FedKTL)). *Please note that this developing project may not be able to reproduce the results on these papers, since some basic settings may change due to the requests of the community.* 
