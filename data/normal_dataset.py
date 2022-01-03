import sys
import os
from os import path as osp
from collections import defaultdict

import torch
from copy import deepcopy
import numpy as np
from torch.utils import data as data
from PIL import Image, ImageEnhance
from torchvision import transforms

class TransformLoader:
    def __init__(self, transformer_opt):
        self.transformer_opt = transformer_opt

    def parse_transform(self, transform_type, transform_opt):
        method = getattr(transforms, transform_type)
        method = method(**transform_opt)
        return method

    def get_composed_transform(self):
        transform_funcs = []
        for transform_opt in self.transformer_opt:
            transform_type = transform_opt.pop('type')
            transform_funcs.append(self.parse_transform(transform_type, transform_opt))
        transform = transforms.Compose(transform_funcs)
        return transform

class NormalDataset(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
          dataroot (str): Data root path
          selected_classes: The list of classes using for training the tasks
          aug (bool): Whether proceeding data augmentation
    """

    def __init__(self, opt):
        super(NormalDataset, self).__init__()
        self.opt = opt
        self.dataroot = opt['dataroot']
        self.all_classes = opt['all_classes']
        self.selected_classes = opt['selected_classes']

        self.opt_transformer_agu = deepcopy(self.opt.get('transformer_agu'))
        self.opt_transformer = deepcopy(self.opt.get('transformer'))

        self.user_defined = opt.get('user_defined', False)

        classes, class_to_idx = self._find_classes(self.dataroot)

        if self.user_defined:
            # task_id = self.opt['task_id']
            # TODO

            # aquatic mammals, fish, insects, reptiles, small mammals, large carnivores
            all_list = ['beaver', 'dolphin', 'otter', 'seal', 'whale',
                               'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
                               'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
                               'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
                        'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
                        'bear', 'leopard', 'lion', 'tiger', 'wolf',

                        'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor',
                        'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
                        'bottle', 'bowl', 'can', 'cup', 'plate',
                        'bed', 'chair', 'couch', 'table', 'wardrobe'
                        ]

            # all_list = ['beaver', 'dolphin', 'otter', 'seal', 'whale',
            #                    'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
            #                    'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
            #                    'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
            #
            #                    'bed', 'chair', 'couch', 'table', 'wardrobe',
            #                    'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',

            #             'baby', 'boy', 'girl', 'man', 'woman',
            #             'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor',
            #             'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
            #             'bottle', 'bowl', 'can', 'cup', 'plate'
            #             ]

            # all_list1 = ['beaver', 'dolphin', 'otter',
            #             'aquarium_fish', 'flatfish', 'ray',
            #             'bee', 'beetle', 'butterfly',
            #             'crocodile', 'dinosaur', 'lizard',
            #             'bed', 'chair', 'couch',
            #             'hamster', 'mouse', 'rabbit',
            #             'baby', 'boy', 'girl',
            #             'lawn_mower', 'rocket', 'streetcar',
            #             'orchid', 'poppy', 'rose',
            #             'bottle', 'bowl', 'can',
            #             ]

            # all_list2 = ['seal', 'whale',
            # 'shark', 'trout',
            # 'caterpillar', 'cockroach',
            # 'snake', 'turtle',
            # 'table', 'wardrobe',
            # 'shrew', 'squirrel',
            # 'man', 'woman',
            # 'tank', 'tractor'
            # 'sunflower', 'tulip'
            # 'cup', 'plate']

            # all_list2 = ['maple', 'oak', 'palm', 'pine', 'willow',
            #              'apples', 'mushrooms', 'oranges', 'pears', 'sweet_peppers',
            #              'clock', 'computer_keyboard', 'lamp', 'telephone', 'television',
            #              'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train']

            # all_list = all_list1 + all_list2

            # if task_id == 0:
            #     waited_list = ['beaver', 'dolphin', 'otter', 'seal', 'whale',
            #                    'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
            #                    'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
            #                    'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
            #                    'bed', 'chair', 'couch', 'table', 'wardrobe',
            #                    'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']
            # else:
            #     waited_list = ['baby', 'boy', 'girl', 'man', 'woman',
            #                    'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor',
            #                    'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
            #                    'bottles', 'bowls', 'cans', 'cups', 'plates']

            waited_list = [all_list[pos] for pos in self.selected_classes]

            class_to_idx = {k: v for k, v in class_to_idx.items() if k in waited_list}

            self._num_classes = len(class_to_idx.keys())
            image_path_list, label_list = self._make_dataset(class_to_idx)

            self.image_path_list = image_path_list

            self.relative_label_dict = {}
            for k, v in class_to_idx.items():
                self.relative_label_dict.update({v: self.all_classes.tolist().index(all_list.index(k))})
            self.softmax_label_list = [self.relative_label_dict[l] for l in label_list]

            self.pathl_to_partl_dict = {}
            for k, v in class_to_idx.items():
                self.pathl_to_partl_dict.update({v: all_list.index(k)})
            self.label_list = [self.pathl_to_partl_dict[l] for l in label_list]

        else:
            class_to_idx = {k: v for k, v in class_to_idx.items() if v in self.selected_classes}

            self._num_classes = len(class_to_idx.keys())
            image_path_list, label_list = self._make_dataset(class_to_idx)

            self.image_path_list = image_path_list
            self.label_list = label_list
            self.relative_label_dict = {}
            for v in class_to_idx.values():
                self.relative_label_dict.update({v: self.all_classes.tolist().index(v)})
            self.softmax_label_list = [self.relative_label_dict[l] for l in self.label_list]

        # set the pre-load to false
        self.pre_load = opt['pre_load']
        self.to_cuda = opt.get('to_cuda', False)
        self.feat_list = None

        # (keys, values) = (label, relative index of selected_class)

        if len(self.image_path_list) != len(self.label_list):
            raise ValueError('Some labels of images in Dataset are missing')

        self.transformer_agu = None
        self.transformer_test = None
        # define the transform function
        if self.opt_transformer_agu is not None:
            tf_aug = TransformLoader(self.opt_transformer_agu)
            self.transformer_agu = tf_aug.get_composed_transform()
        if self.opt_transformer is not None:
            tf = TransformLoader(self.opt_transformer)
            self.transformer_test = tf.get_composed_transform()

        self.set_aug(opt['aug'])

        # pre-load the images
        if opt['pre_load']:
            self.image_list = [Image.open(path).convert('RGB') for path in self.image_path_list]
            if self.to_cuda:
                self.image_list = [self.transform(img).cuda() for img in self.image_list]
        else:
            self.image_list = None

    def set_aug(self, aug):
        self._aug = aug
        if self._aug:
            self.transform = self.transformer_agu
        else:
            self.transform = self.transformer_test

    def get_aug(self):
        return self._aug

    # def set_buffer(self, feat_list, label_list):
    #     """
    #     set the features pre-calculated in advance
    #     :param feat_list:
    #     :param label_list:
    #     :return:
    #     """
    #     self.pre_load = True
    #     self.feat_list = feat_list.tolist()
    #     self.label_list = label_list.tolist()

    def get_num_classes(self):
        return self._num_classes

    def sample_the_training_data(self, task_id, shots):
        if task_id > 0:
            self._sample_data(shots)

    def sample_the_buffer_data(self, num_samples_per_class):
        self._sample_data(num_samples_per_class)

    def sample_the_buffer_data_with_index(self, index_list):
        self.image_path_list = [self.image_path_list[index] for index in index_list]
        self.label_list = [self.label_list[index] for index in index_list]
        self.softmax_label_list = [self.softmax_label_list[index] for index in index_list]
        if self.image_list is not None:
            self.image_list = [self.image_list[index] for index in index_list]

    def sample_all_data(self, agu=True, agu_times=1):
        if agu and agu_times > 1:
            index_list = list(range(len(self.image_path_list))) * agu_times
        else:
            index_list = range(len(self.image_path_list))
        return self.obtainImageWithIndexList(index_list, agu=agu)

    def combine_another_dataset(self, dataset):
        self.selected_classes = np.append(self.selected_classes, dataset.selected_classes)
        self.image_path_list = self.image_path_list + dataset.image_path_list
        self.label_list = self.label_list + dataset.label_list
        self.softmax_label_list += dataset.softmax_label_list
        self._num_classes = len(np.unique(self.label_list))
        self.relative_label_dict = {**self.relative_label_dict, **dataset.relative_label_dict}
        if self.image_list is not None and dataset.image_list is not None:
            self.image_list = self.image_list + dataset.image_list
        elif self.image_list is None and dataset.image_list is None:
            return
        else:
            raise ValueError('The origin dataset and the dataset to be combined should all have or have not image_list')

    def combine_crossDomain_dataset(self, dataset):
        label_list = [l + len(self.all_classes) for l in dataset.label_list]
        softmax_label_list = [l + len(self.all_classes) for l in dataset.softmax_label_list]
        self.selected_classes = np.append(self.selected_classes, dataset.selected_classes + len(self.all_classes))

        self.all_classes = np.append(self.all_classes, dataset.all_classes + len(self.all_classes))
        self.image_path_list = self.image_path_list + dataset.image_path_list
        self.label_list = self.label_list + label_list
        self.softmax_label_list += softmax_label_list
        self._num_classes = len(np.unique(self.label_list))
        if self.image_list is not None and dataset.image_list is not None:
            self.image_list = self.image_list + dataset.image_list
        elif self.image_list is None and dataset.image_list is None:
            return
        else:
            raise ValueError('The origin dataset and the dataset to be combined should all have or have not image_list')

    def _sample_data(self, num_samples_per_class):

        training_data_index = self.sample_data_index(num_samples_per_class)

        self.sample_the_buffer_data_with_index(training_data_index)

    def sample_data_index(self, num_samples_per_class):
        if self._num_classes * num_samples_per_class > len(self.label_list):
            raise ValueError(f'Can not sample data from the dataset '
                             f'which only contains {len(self.label_list)} samples')

        training_data_index = []
        index_dic = defaultdict(list)
        for index, label in enumerate(self.label_list):
            index_dic[label].append(index)
        for label, index_list in index_dic.items():
            pos = torch.randperm(len(index_list))[:num_samples_per_class].tolist()
            training_data_index.extend([index_list[p] for p in pos])

        return training_data_index

    def _make_dataset(self, class_to_idx):
        """
        generate the image list and label list
        """
        image_path_list = []
        label_list = []
        dir = osp.expanduser(self.dataroot)
        for target in sorted(class_to_idx.keys()):
            d = osp.join(dir, target)
            if not osp.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    image_path_list.append(path)
                    label_list.append(class_to_idx[target])

        return image_path_list, label_list

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()

        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __obtainImage(self, index):
        path = self.image_path_list[index]
        if self.image_list is not None:
            image = self.image_list[index]
        else:
            image = Image.open(path).convert('RGB')

        image = self.transform(image)
        return image

    def obtainImageWithIndexList(self, index_list, agu=False):
        if self.image_list is not None:
            images = [self.image_list[index] for index in index_list]
        else:
            images = [Image.open(self.image_path_list[index]).convert('RGB') for index in index_list]

        images = self.transform_image_list(images, agu = agu)
        labels = [self.label_list[index] for index in index_list]
        label_softmax = [self.softmax_label_list[index] for index in index_list]
        labels = torch.tensor(labels)
        label_softmax = torch.tensor(label_softmax)

        return (images, labels, label_softmax)

    def transform_image_list(self, images, agu=False):
        if agu:
            images = [self.transformer_agu(image) for image in images]
        else:
            images = [self.transformer_test(image) for image in images]

        images = torch.stack(images, dim=0)

        return images

    def __getitem__(self, index):
        label = self.label_list[index]
        label_softmax = self.softmax_label_list[index]

        image = self.__obtainImage(index)
        return (image, label, label_softmax)

    def __len__(self):
        return len(self.image_path_list)
