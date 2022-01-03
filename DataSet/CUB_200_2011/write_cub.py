import torch
import os.path as osp
import os
from PIL import Image


def list2dict(list):
    dict = {}
    for l in list:
        s = l.split(' ')
        id = int(s[0])
        cls = s[1]
        if id not in dict.keys():
            dict[id] = cls
        else:
            raise EOFError('The same ID can only appear once')
    return dict

def text_read(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            lines[i] = line.strip('\n')
    return lines

image_file = './images.txt'
split_file = './train_test_split.txt'
class_file = './image_class_labels.txt'

id2image = list2dict(text_read(image_file))
id2train = list2dict(text_read(split_file))  # 1: train images; 0: test iamges
id2class = list2dict(text_read(class_file))

train_idx = []
test_idx = []
for k in sorted(id2train.keys()):
    if id2train[k] == '1':
        train_idx.append(k)
    else:
        test_idx.append(k)

for id in train_idx:
    class_name = id2class[id]
    image = id2image[id]
    image_path = osp.join('./images', image)

    image_file = Image.open(image_path).convert('RGB')

    target_path_root = './train'
    if not osp.exists(target_path_root):
        os.mkdir(target_path_root)

    target_path = osp.join(target_path_root, image)
    target_path_formerpart, name = osp.split(target_path)

    if not osp.exists(target_path_formerpart):
        os.mkdir(target_path_formerpart)
    os.system(f'cp {image_path} {target_path}')

for id in test_idx:
    class_name = id2class[id]
    image = id2image[id]
    image_path = osp.join('./images', image)

    target_path_root = './test'
    if not osp.exists(target_path_root):
        os.makedirs(target_path_root)

    target_path = osp.join(target_path_root, image)
    target_path_formerpart, name = osp.split(target_path)
    if not osp.exists(target_path_formerpart):
        os.makedirs(target_path_formerpart)
    os.system(f'cp {image_path} {target_path}')





