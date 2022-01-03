import torch
import numpy
import os.path as osp
import os


csv_map = {'train': './train.csv', 'test': './test.csv'}

IMAGE_PATH = './images'


for key, value in csv_map.items():
    csv_path = value
    type = key
    lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
    wnids = []

    for l in lines:
        name, wnid = l.split(',')
        image_path = osp.join(IMAGE_PATH, name)
        if wnid not in wnids:
            current_path = f'./{type}/{wnid}'
            os.makedirs(current_path)
            wnids.append(wnid)
        current_img_path = osp.join(current_path, name)
        os.system(f'cp {image_path} {current_img_path}')

