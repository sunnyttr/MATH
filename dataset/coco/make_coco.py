# generate mat data for MSCOCO
import os
import scipy.io as scio
import numpy as np
import random
import torch

seed = 1
torch.random.manual_seed(seed=seed)
np.random.seed(seed=seed)
random.seed(seed)

img_path = 'all_imgs.txt'
txt_path = 'all_txts.txt'
lab_path = 'all_labels.txt'
label_t_path = 'labels_annotation.txt'

img_root_path = "/share/home/u2315363106/upload/coco_images/"

'''
####################### sentences #######################
'''
def sentence2str(f_name):
    with open(f_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    res = {}
    for i, line in enumerate(lines):
        line = line.strip()
        sentences_5 = line.split(';')
        use_cap = sentences_5[random.randint(0, len(sentences_5) - 1)]  # use random one
        use_cap = use_cap.split(".")[0]
        res[i] = use_cap
    return res
    
sentences_strs_dict = sentence2str(txt_path)


'''
####################### labels #######################
'''
labels = np.loadtxt(lab_path, dtype=np.float32)  # ndarray: (N, 80)
labels_dict = {}
for i in range(labels.shape[0]):
    labels_dict[i] = labels[i]


'''
####################### image paths #######################
'''
img_abs_paths_dict = {}
with open(img_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        img_abs_paths_dict[i] = os.path.join(img_root_path, line.strip().split("/")[-1])


'''
####################### remove empty label (all 0) #######################
all_labels.txt contains 1069 empty labels (all 0).
'''
used_keys = []  # list
for key in labels_dict:
    if sum(labels_dict[key]) != 0:
        used_keys.append(key)

sentences_strs_list = []
img_abs_paths_list = []
labels_list = []
for key in used_keys:  # 122218 instances remained
    sentences_strs_list.append(sentences_strs_dict[key])
    img_abs_paths_list.append(img_abs_paths_dict[key])
    labels_list.append(labels_dict[key])


txt = np.asarray(sentences_strs_list)
img = np.asarray(img_abs_paths_list)
lab = np.asarray(labels_list)

# generate label_t. data
def label2str(label_filename):
    """
    # return ndarray : (N, ), each one is a str of label, i.e., 'person knife cake sink'
    """
    with open(label_filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    res = []
    for i, line in enumerate(lines):
        line = line.strip()

        if line == "":
            continue

        tags = line.split(',')
        sss = " ".join(tags)
        res.append(sss)

    return np.array(res)

label_str = label2str(label_t_path)  # ndarray: (N,)

# save path
root_dir = "."

scio.savemat(os.path.join(root_dir, "index.mat"), {'index': img})
scio.savemat(os.path.join(root_dir, "caption.mat"), {'caption': txt})
scio.savemat(os.path.join(root_dir, "label.mat"), {'label': lab})
scio.savemat(os.path.join(root_dir, "label_t.mat"), {'label_t': label_str})	
print("Save all *.mat")
