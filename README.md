# Multi-label augmentation transformer hashing for cross-modal retrieval (MATH)
Pytorch implementation of paper 'Multi-label augmentation transformer hashing for cross-modal retrieval'.

## Abstract
Deep cross-modal hashing (DCMH) has become a prominent research direction in multimedia information retrieval by integrating deep learning with hashing techniques. Existing methods primarily focus on capturing pairwise similarities across modalities while often underutilizing the rich semantic information embedded in multi-label annotations. Although some approaches leverage multi-labels to supervise hashing function learning, their effectiveness is limited by the sparsity of the multi-label feature space. To address these limitations, we propose a multi-label augmentation transformer hashing (MATH) method. Specifically, MATH introduces a label-modality feature fusion module based on attention mechanisms, which effectively integrates important semantic features from image and text modalities into the multi-label space. This fusion enhances the representational capacity of multi-label features, thereby improving their ability to guide cross-modal hashing learning. Additionally, we define a multi-label cross-modal contrastive alignment loss, which unifies image, text, and multi-label information in a contrastive learning framework to achieve more precise semantic alignment across modalities. Extensive experiments on three benchmark datasets demonstrate that the proposed MATH method achieves state-of-the-art performance in cross-modal hashing retrieval tasks.

### Dependencies 
You need to install these packages to run
- python 3.7.16
- pytorch 1.9.1
- torchvision 0.10.1
- numpy
- scipy
- tqdm
- pillow
- einops
- ftfy
- regex
- ...

### Preparation
1.This code is based on the "ViT-B/32". Please download "ViT-B/32" and place it in the ./cache directory. You can obtain it from the following link:

link：https://pan.baidu.com/s/1o031yK_SwX32Q1vD5YefFA password：oszl

2.The cleaned datasets (MIRFLICKR25K, NUSWIDE, and MSCOCO) used in our experiments are available at the following link:

link：https://pan.baidu.com/s/1o031yK_SwX32Q1vD5YefFA
password：oszl

(1)Update the variable `img_root_path` in the script `make_DatasetName.py` to the absolute path of the directory.

(2)Execute the script `make_DatasetName.py` to generate the corresponding `.mat` files. Then, use these `.mat` files to carry out the experiment.

### How to run
``` 
python main.py --is-train --dataset flickr25k --query-num 2000 --train-num 10000 --result-name "RESULT_MATH_FLICKR" --k-bits 16
```

More scripts for training and testing can be found in `./run_MATH.sh`. 

If you have any problems, please feel free to contact Zhiran Yu (yuzhiran45@163.com).
