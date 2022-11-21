### Brain Tumor Segmentation Based on the Fusion of Deep Semantics and Edge Information in Multimodal MRI

-----------

The semantic segmentation part consists of the improved Swin Transformer, and the edge detection part consists of convolutional layers and ESAB modules.The fusion part uses the method of graph convolution to fuse semantic features and edge features.

#### DataSets

----------

The dataset used in this paper is BraTS2018，2019 and 2020. The datasets can be found in kaggle, using the training set part.

#### Requirement

-------------------

This article is implemented by Pytorch.

1. PyTorch 1.9.0
2. Some other libraries

#### Cite

----------------------------------------------

@article{*ZHU2023376*,

title = {Brain tumor segmentation based on the fusion of deep semantics and edge information in multimodal MRI},

journal = {Information Fusion},

volume = {91},

pages = {376-387},

year = {2023},

issn = {1566-2535},

doi = {https://doi.org/10.1016/j.inffus.2022.10.022},

url = {https://www.sciencedirect.com/science/article/pii/S1566253522001981},

author = {Zhiqin Zhu and Xianyu He and Guanqiu Qi and Yuanyuan Li and Baisen Cong and Yu Liu},

}
