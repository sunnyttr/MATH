# Multi-label augmentation transformer hashing for cross-modal retrieval (MATH)
Pytorch implementation of paper 'Multi-label augmentation transformer hashing for cross-modal retrieval'.

## Abstract
Deep cross-modal hashing (DCMH) has emerged as a significant research direction in multimedia information retrieval by integrating deep learning with hashing techniques. Current approaches predominantly focus on analyzing cross-modal pairwise similarities while often overlooking the valuable contribution of multi-label information. Although some methods utilize multi-labels to supervise the learning of hashing functions, their performance remains limited due to the inherent sparsity of the multi-label feature space.To address these limitations, we propose a multi-label augmentation transformer hashing (MATH) method. Specifically, MATH incorporates a novel label modality feature fusion module that employs attention mechanisms to integrate semantically important features from both image and text modalities into the multi-label modality. This integration generates more comprehensive multi-label feature representations that effectively guide the learning of cross-modal hashing functions. Furthermore, we introduce a multi-label cross-modal contrastive alignment loss that incorporates image, multi-label, and text information into a unified contrastive learning framework, thereby enhancing semantic alignment across modalities with greater precision. Experimental results on three benchmark datasets demonstrate that our proposed MATH method achieves state-of-the-art performance in cross-modal hashing retrieval tasks.

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
