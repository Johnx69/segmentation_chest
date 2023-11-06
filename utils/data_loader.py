import numpy as np
import imgaug.augmenters as iaa
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import cv2
import torch.nn.functional as F


class Covid(Dataset):
    def __init__(self, rootpath, cropsize=(256, 256), mode="train", *args, **kwargs):
        super(Covid, self).__init__(*args, **kwargs)
        assert mode in ("train", "val", "test")
        self.mode = mode
        self.cropsize = cropsize

        self.rootpth = rootpath
        self.imgs = []

        class_dir = os.listdir(
            os.path.join(self.rootpth, self.mode.capitalize())
        )  # class
        self.class_cvt = {"Normal": 0, "COVID-19": 1, "Non-COVID": 2}

        # print(class_dir)
        for _class in class_dir:
            self.class_path = os.path.join(self.rootpth, self.mode.capitalize())
            self.img_path = os.path.join(self.class_path, _class, "images")
            #             print(self.img_path)
            img_lst = os.listdir(self.img_path)
            #             print(img_lst)
            #             infect_imgs = os.listdir(self.img_path.replace('Lung Segmentation Data','Infection Segmentation Data',2))
            img_lst.sort()
            for img in img_lst:
                self.imgs.append((img, self.class_cvt[_class]))
        #         print(self.imgs)

        #         #  pre-processing
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.trans_color = iaa.Sequential(
            [
                iaa.LinearContrast((0.4, 1.6)),
                #           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                #           iaa.Add((-40, 40), per_channel=0.5, name="color-jitter")
            ]
        )
        self.trans_train = iaa.Sequential(
            [
                #           iaa.Resize(self.cropsize),
                iaa.Fliplr(0.5),
                #           iaa.Affine(rotate=(-45, 45),
                #                     translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)}),
            ]
        )

    def __getitem__(self, idx):
        impth = self.imgs[idx]
        #         print(impth)
        self.img_path = os.path.join(
            self.class_path, list(self.class_cvt.keys())[impth[1]], "images"
        )
        img = cv2.imread(os.path.join(self.img_path, impth[0]))
        img = cv2.resize(img, self.cropsize, cv2.INTER_LINEAR)

        #         label = torch.tensor(impth[1])
        label = F.one_hot(torch.tensor(impth[1]), num_classes=3)
        #         print(label)

        mask_path = self.img_path.replace("images", "lung masks")

        lung_mask = cv2.imread(os.path.join(mask_path, impth[0]), 0)
        lung_mask = cv2.resize(lung_mask, self.cropsize, cv2.INTER_NEAREST).astype(
            np.int64
        )
        #         lung_mask = np.where(lung_mask==0, 0 , 1)

        infect_path = self.img_path.replace("images", "infection masks")
        infect_mask = cv2.imread(os.path.join(infect_path, impth[0]), 0)
        infect_mask = cv2.resize(infect_mask, self.cropsize, cv2.INTER_NEAREST).astype(
            np.int64
        )
        #         infect_mask = np.where(infect==0, 0 , 1)

        # if self.mode == 'test':
        # return img, label,lung_mask, infect_mask

        if self.mode == "train":
            #             color = self.trans_color.to_deterministic()

            #             img = color.augment_image(img)
            det_tf = self.trans_train.to_deterministic()
            img = det_tf.augment_image(img)
            lung_mask = det_tf.augment_image(lung_mask)
            infect_mask = det_tf.augment_image(infect_mask)

        img = self.to_tensor(img)
        #         img = img.permute(2, 0, 1)

        lung_mask = np.where(lung_mask != 0, 1.0, 0.0)
        lung_mask = torch.from_numpy(lung_mask.astype(np.float32)).clone()
        lung_mask = F.one_hot(lung_mask.long(), num_classes=2)
        lung_mask = lung_mask.to(torch.float32)
        lung_mask = lung_mask.permute(2, 0, 1)

        infect_mask = np.where(infect_mask != 0, 1.0, 0.0)
        infect_mask = torch.from_numpy(infect_mask.astype(np.float32)).clone()
        infect_mask = F.one_hot(infect_mask.long(), num_classes=2)
        infect_mask = infect_mask.to(torch.float32)
        infect_mask = infect_mask.permute(2, 0, 1)

        return img, label, lung_mask, infect_mask

    def __len__(self):
        return len(self.imgs)
