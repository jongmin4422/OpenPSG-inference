import os
import torch
import numpy as np
import pickle
import xml.etree.ElementTree as Et

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection

year_folder = {
    '2007': 'VOC2007',
    '2012': 'VOC2012'
}

VOC_CLASS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
             'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


class VOCDataset(Dataset):
    def __init__(self, root, img_transform, year='2007', split='trainval'):
        self.root = root
        self.img_transform = img_transform
        self.year = year
        self.split = split

        self.img_dir = os.path.join(root, year_folder[year], 'JPEGImages')
        self.ann_dir = os.path.join(root, year_folder[year], 'Annotations')

        self.load_img_list()

    def load_img_list(self):
        if self.split not in ['train', 'val', 'trainval', 'test']:
            assert (
                'Unknown data split: [{}] not in [train, val, trainval, test]')

        split_file = os.path.join(
            self.root, year_folder[self.year], 'ImageSets/Main', self.split + '.txt')
        self.img_name_list = np.loadtxt(split_file, dtype=np.int32)

    def load_ann_file(self, ann_path):
        gt = []
        ann = Et.parse(ann_path).getroot()
        for obj in ann.findall('object'):
            # gt.append(obj[0].text)
            gt.append(VOC_CLASS.index(obj[0].text))

        return np.array(gt)

    def decode_int_filename(self, int_name):
        if self.year == '2007':
            return '{:06d}'.format(int_name)

        elif self.year == '2012':
            s = str(int(int_name))
            return s[:4] + '_' + s[4:]

        else:
            NotImplementedError

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, index):
        name_int = self.img_name_list[index]
        name_str = self.decode_int_filename(name_int)

        img = Image.open(os.path.join(self.img_dir, name_str + '.jpg'))
        gt = self.load_ann_file(os.path.join(self.ann_dir, name_str + '.xml'))

        transformed_img = self.img_transform(img)

        return {'idx': index, 'name': name_str, 'original_imgs': np.array(img), 'transformed_imgs': transformed_img, 'gt': gt}
        return {'idx': index, 'name': name_str, 'transformed_imgs': transformed_img, 'gt': gt}


class VOCProposalDataset(VOCDataset):
    def __init__(self, root, region_proposal_path, img_transform, year='2007', split='trainval', num_proposals=100):
        super().__init__(root, img_transform, year, split)
        self.num_proposals = num_proposals
        self.region_proposal_path = region_proposal_path
        self.load_proposals()

    def load_proposals(self):
        with open(self.region_proposal_path, 'rb') as f:
            proposals = pickle.load(f)
        self.proposals = proposals

    def __getitem__(self, index):
        name_int = self.img_name_list[index]
        name_str = self.decode_int_filename(name_int)

        img = Image.open(os.path.join(self.img_dir, name_str + '.jpg'))
        gt = self.load_ann_file(os.path.join(self.ann_dir, name_str + '.xml'))
        proposals = self.proposals[index][0][:self.num_proposals].copy()

        transformed_imgs = []
        for box in proposals:
            crop_img = img.crop(tuple(box))
            transformed_imgs.append(self.img_transform(crop_img))
        transformed_imgs = torch.stack(transformed_imgs, dim=0)

        W, H = img.size
        proposals[:, [0, 2]] /= W
        proposals[:, [1, 3]] /= H

        return {'idx': index, 'name': name_str, 'original_imgs': np.array(img), 'transformed_imgs': transformed_imgs, 'normalized_box': proposals, 'gt': gt}


class COCODataset(Dataset):
    def __init__(self, root, coco_dict_path, img_transform, year='2017', split='val'):
        self.root = root
        self.coco_dict_path = coco_dict_path
        self.img_transform = img_transform
        self.year = year
        self.split = split

        self.img_dir = os.path.join(root, split+year)
        self.ann_dir = os.path.join(
            root, 'annotations', 'instances_{}{}.json'.format(split, year))

        self.load_dataset()
        self.load_coco_dict()

    def load_dataset(self):
        self.dataset = CocoDetection(root=self.img_dir, annFile=self.ann_dir)

    def load_coco_dict(self):
        with open(self.coco_dict_path, 'rb') as f:
            coco_dict = pickle.load(f)
        self.coco_dict = coco_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img = self.dataset[index][0]
        ann = self.dataset[index][1]
        cls2idx = self.coco_dict['cls2idx']

        gt = []
        for obj in ann:
            gt.append(cls2idx[obj['category_id']])
        gt = torch.tensor(gt)

        transformed_imgs = self.img_transform(img)

        return {'idx': index, 'original_imgs': np.array(img), 'transformed_imgs': transformed_imgs, 'gt': gt}


class COCOProposalDataset(COCODataset):
    def __init__(self, root, region_proposal_path, coco_dict_path, img_transform, year='2017', split='val', num_proposals=100):
        super().__init__(root, coco_dict_path, img_transform, year, split)
        self.region_proposal_path = region_proposal_path
        self.num_proposals = num_proposals
        self.load_proposals()

    def load_proposals(self):
        with open(self.region_proposal_path, 'rb') as f:
            proposals = pickle.load(f)
        self.proposals = proposals

    def __getitem__(self, index):
        img = self.dataset[index][0]
        ann = self.dataset[index][1]
        cls2idx = self.coco_dict['cls2idx']

        gt = []
        for obj in ann:
            gt.append(cls2idx[obj['category_id']])
        gt = torch.tensor(gt)
        proposals = self.proposals[index][0][:self.num_proposals].copy()

        transformed_imgs = []
        for box in proposals:
            crop_img = img.crop(tuple(box))
            transformed_imgs.append(self.img_transform(crop_img))
        transformed_imgs = torch.stack(transformed_imgs, dim=0)

        W, H = img.size
        proposals[:, [0, 2]] /= W
        proposals[:, [1, 3]] /= H

        return {'idx': index, 'original_imgs': np.array(img), 'transformed_imgs': transformed_imgs, 'normalized_box': proposals, 'gt': gt}
