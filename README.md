# General Federated Learning (GFL)
Standard federated learning (FL), e.g., [FedAvg](http://proceedings.mlr.press/v54/mcmahan17a.html), assumes that all the participating clients build their local models with the same architecture, which limits its utility in real-world scenarios. In practice, each client can build its own model with a specific model architecture for specific local tasks. 

## Scenarios and datasets

Here, we only show the mnist dataset in the ***label skew*** scenario generated via Dirichlet distribution for example. Please refer to my another repository [PFL-Non-IID](https://github.com/TsingZ0/PFL-Non-IID) for more help. 

You can also modify codes in PFL-Non-IID to support model heterogeneity scenarios, but it tasks much effort. In this repository, you only need to configure `system/main.py` to supports model heterogeneity scenarios. 

## Data-free algorithms with code (updating)
  
- **Local** — Each client trains its model locally without federation.

- **FedDistill** — [Federated Knowledge Distillation](https://www.cambridge.org/core/books/abs/machine-learning-and-wireless-communications/federated-knowledge-distillation/F679266F85493319EB83635D2B17C2BD#access-block) *2020*

- **LG-FedAvg** — [Think Locally, Act Globally: Federated Learning with Local and Global Representations](https://arxiv.org/abs/2001.01523) *2020*

- **FedGen** — [Data-Free Knowledge Distillation for Heterogeneous Federated Learning](http://proceedings.mlr.press/v139/zhu21b.html) *ICML 2021*

- **FedProto** — [FedProto: Federated Prototype Learning across Heterogeneous Clients](https://ojs.aaai.org/index.php/AAAI/article/view/20819) *AAAI 2022*

