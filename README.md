# Heterogeneous Federated Learning (HtFL)
Standard federated learning, e.g., [FedAvg](http://proceedings.mlr.press/v54/mcmahan17a.html), assumes that all the participating clients build their local models with the same architecture, which limits its utility in real-world scenarios. In practice, each client can build its model with a specific model architecture for a specific local task. 

## Scenarios and datasets

Here, we only show the MNIST dataset in the ***label skew*** scenario generated via Dirichlet distribution for example. Please refer to my other repository [PFL-Non-IID](https://github.com/TsingZ0/PFL-Non-IID) for more help. 

*You can also modify codes in PFL-Non-IID to support model heterogeneity scenarios, but it requires much effort. In this repository, you only need to configure `system/main.py` to support model heterogeneity scenarios.*

**Note**: you may need to manually clean checkpoint files in the `temp/` folder via `system/clean_temp_files.py` if your program crashes accidentally. You can also set a checkpoint folder by yourself to prevent automatic deletion using the `-sfn` argument in the command line. 

## Data-free algorithms with code (updating)
  
- **Local** — Each client trains its model locally without federation.

- **FedDistill** — [Federated Knowledge Distillation](https://www.cambridge.org/core/books/abs/machine-learning-and-wireless-communications/federated-knowledge-distillation/F679266F85493319EB83635D2B17C2BD#access-block) *2020*

- **FML** — [Federated Mutual Learning](https://arxiv.org/abs/2006.16765) *2020*

- **LG-FedAvg** — [Think Locally, Act Globally: Federated Learning with Local and Global Representations](https://arxiv.org/abs/2001.01523) *2020*

- **FedGen** — [Data-Free Knowledge Distillation for Heterogeneous Federated Learning](http://proceedings.mlr.press/v139/zhu21b.html) *ICML 2021*

- **FedProto** — [FedProto: Federated Prototype Learning across Heterogeneous Clients](https://ojs.aaai.org/index.php/AAAI/article/view/20819) *AAAI 2022* 

- **FedKD** — [Communication-efficient federated learning via knowledge distillation](https://www.nature.com/articles/s41467-022-29763-x) *Nature Communications 2022*

- **FedGH** — [FedGH: Heterogeneous Federated Learning with Generalized Global Header](https://dl.acm.org/doi/10.1145/3581783.3611781) *ACM MM 2023*

