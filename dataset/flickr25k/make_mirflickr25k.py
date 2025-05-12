# generate mat data for flickr25k
import os
import scipy.io as scio
import numpy as np

img_path = 'all_imgs.txt'
txt_path = 'all_tags.txt'
lab_path = 'all_labels.txt'
label_t_path = 'labels_annotation.txt'

img_root_path = "/share/home/u2315363106/upload/mirflickr/"

'''
####################### tags #######################
'''
def tag2str(tag_filename):
    """
    # return ndarray : (N, ), each one is a str of tags, i.e., 'cigarette tattoos smoke red dress sunglasses'
    """
    with open(tag_filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    res = []
    for i, line in enumerate(lines):
        line = line.strip()

        if line == "":
            print("Error: Empty tags...")  # no empty in flickr
            exit()

        tags = line.split(',')
        sss = " ".join(tags)
        res.append(sss)

    return np.array(res)

tags_strs = tag2str(txt_path)  # ndarray: (N,)


'''
####################### labels #######################
'''
labels = np.loadtxt(lab_path, dtype=np.float32)  # ndarray: (N, 24)


'''
####################### image paths #######################
'''
img_abs_paths = []
with open(img_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

    for line in lines:
        ttt = line.strip()
        img_abs_paths.append(os.path.join(img_root_path, ttt.split("/")[1]))

img_abs_paths = np.array(img_abs_paths)  # ndarray: (N,)

# generate label_t. data
def label2str(label_filename):
    """
    # return ndarray : (N, ), each one is a str of label, i.e., 'female people portrait sky structures'
    """
    with open(label_filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    res = []
    for i, line in enumerate(lines):
        line = line.strip()

        if line == "":
            print("Error: Empty label...")
            exit()

        tags = line.split(',')
        sss = " ".join(tags)
        res.append(sss)

    return np.array(res)

label_str = label2str(label_t_path)  # ndarray: (N,)

# save path
root_dir = "."

scio.savemat(os.path.join(root_dir, "index.mat"), {'index': img_abs_paths}) 
scio.savemat(os.path.join(root_dir, "caption.mat"), {'caption': tags_strs}) 
scio.savemat(os.path.join(root_dir, "label.mat"), {'label': labels})
scio.savemat(os.path.join(root_dir, "label_t.mat"), {'label_t': label_str})	 
print("Save all *.mat")
