# Graph Node Classification Model with Universal Domain Adaptation

This repository contains the PyTorch implementation of my novel node classification model based on Graph Neural Networks (GNNs). This model excels in tasks involving graph node classification within the context of universal domain adaptation, where label sets between source and target domains may not be identical. It is also designed to resist overfitting, even in the presence of label noise.

## Key Features

- **Dual GCN Components**: Integrates local and global graph information for comprehensive feature extraction.
- **Attention Mechanisms**: Generates unified node representations across different graphs.
- **Scoring and Separation Strategies**: Differentiates common class samples from target private class samples.
- **Robust to Label Noise**: Utilizes [parameter and neuron pruning](https://proceedings.neurips.cc/paper_files/paper/2023/file/a4316bb210a59fb7aafeca5dd21c2703-Paper-Conference.pdf), and a [dual-teacher framework](https://papers.nips.cc/paper_files/paper/2023/file/7eeb42802d3750ca59e8a0523068e9e6-Paper-Conference.pdf) based on the ensemble method.
- **Effective Knowledge Transfer**: Employs various loss functions to promote knowledge transfer between domains.


## Dependencies

- Python (>=3.6)
- Torch  (>=1.2.0)
- numpy (>=1.16.4)
- torch_scatter (>= 1.3.0)
- torch_geometric (>= 1.3.0)

## Datasets
The `data` folder contains different domain data. The preprocessed data can be found on [Google Drive](https://drive.google.com/file/d/1DzQ3QN9yjQxU4vtYkXyCiJKFw7oCCPSM/view?usp=sharing).

The original datasets can be found [here](https://www.aminer.cn/citation).

## Usage
 - Place the datasets in `data/`
 - Change the `dataset` in `UniDA_GNN.py` .
 - Training/Testing:
 ```bash
 python UniDA_GNN.py
 ```
