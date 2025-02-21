'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2024-01-26

The code is designed for self-blending method (SBI, CVPR 2024).
'''
import os
import random
import shutil
import sys
sys.path.append('.')

import cv2
import yaml
import torch
import numpy as np
from copy import deepcopy
import albumentations as A
from training.dataset.albu import IsotropicResize
from training.dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from training.dataset.sbi_poisson_api import SBI_API

# if os.path.exists('./sbi_output_poisson'):
#     shutil.rmtree('./sbi_output_poisson')
# os.makedirs('./sbi_output_poisson', exist_ok=True)

class SBIPoissonDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)
        
        # Get real lists
        # Fix the label of real images to be 0
        self.real_imglist = [(img, label) for img, label in zip(self.image_list, self.label_list) if label == 0]

        # Init SBI
        self.sbi = SBI_API(phase=mode,image_size=config['resolution'])

        # Init data augmentation method
        self.transform = self.init_data_aug_method()

    def __getitem__(self, index):
        # print("index=", index)
        # Get the real image paths and labels
        real_image_path, real_label = self.real_imglist[index]

        # Get the landmark paths for real images
        real_landmark_path = real_image_path.replace('frames', 'landmarks').replace('.png', '.npy')
        landmark = self.load_landmark(real_landmark_path).astype(np.int32)

        # Load the real images
        real_image = self.load_rgb(real_image_path)
        # print("real_image_path=", real_image_path)
        real_path_name = real_image_path.replace('\\', '_').replace('.png', '')
        real_path_name = '_'.join(real_path_name.split('_')[-2:])
        # print("real_path_name=", real_path_name)
        real_image = np.array(real_image)  # Convert to numpy array

        # Generate the corresponding SBI sample
        fake_image, real_image, fake_poisson_image = self.sbi(real_image, landmark)
        # fake_image, real_image = self.sbi(real_image, landmark)
        assert fake_poisson_image is not None
        if fake_image is None:
            fake_image = deepcopy(real_image)
            fake_poisson_image = deepcopy(real_image)
            fake_label = 0
        else:
            fake_label = 1

        # # ----remove from here -----

        # # Convert RGB to BGR
        # fake_image_bgr = cv2.cvtColor(fake_image, cv2.COLOR_RGB2BGR)
        # real_image_bgr = cv2.cvtColor(real_image, cv2.COLOR_RGB2BGR)
        # fake_poisson_image_bgr = cv2.cvtColor(fake_poisson_image, cv2.COLOR_RGB2BGR)

        # # if index == 604:
        # #     print("idx, landmark shape: ", index, landmark.shape)
        # #     print("landmark=", landmark)

        # # # Draw landmark points on the images
        # # for point in landmark:
        # #     cv2.circle(fake_image_bgr, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
        # #     cv2.circle(real_image_bgr, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)

        # # Save images using cv2.imwrite
        # randid = np.random.randint(1, 1000) 
        # # os.makedirs('./sbi_output_poisson', exist_ok=True)
        # cv2.imwrite(f'./sbi_output_poisson/{index}_fake_{real_path_name}.png', fake_image_bgr)
        # cv2.imwrite(f'./sbi_output_poisson/{index}_real_{real_path_name}.png', real_image_bgr)
        # cv2.imwrite(f'./sbi_output_poisson/{index}_fake_poisson_{real_path_name}.png', fake_poisson_image_bgr)

        # # ---end of remove----

        # To tensor and normalize for fake and real images
        fake_image_trans = self.normalize(self.to_tensor(fake_image))
        real_image_trans = self.normalize(self.to_tensor(real_image))
        fake_poisson_image_trans = self.normalize(self.to_tensor(fake_poisson_image))

        if random.random() < 0.8:
            return {"fake": (fake_image_trans, fake_label), 
                "real": (real_image_trans, real_label)}
        else:
            return {"fake": (fake_poisson_image_trans, fake_label), 
                "real": (real_image_trans, real_label)}

    def __len__(self):
        return len(self.real_imglist)

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor and label tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors for fake and real data
        fake_images, fake_labels = zip(*[data["fake"] for data in batch])
        real_images, real_labels = zip(*[data["real"] for data in batch])

        # Stack the image, label, landmark, and mask tensors for fake and real data
        fake_images = torch.stack(fake_images, dim=0)
        fake_labels = torch.LongTensor(fake_labels)
        real_images = torch.stack(real_images, dim=0)
        real_labels = torch.LongTensor(real_labels)

        # Combine the fake and real tensors and create a dictionary of the tensors
        images = torch.cat([real_images, fake_images], dim=0)
        labels = torch.cat([real_labels, fake_labels], dim=0)
        
        data_dict = {
            'image': images,
            'label': labels,
            'landmark': None,
            'mask': None,
        }
        return data_dict

    def init_data_aug_method(self):
        trans = A.Compose([           
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
            A.OneOf([                
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p = 0 if self.config['with_landmark'] else 1),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'], contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'], quality_upper=self.config['data_aug']['quality_upper'], p=0.5)
        ], 
            additional_targets={'real': 'sbi'},
        )
        return trans


if __name__ == '__main__':
    with open('/data/home/zhiyuanyan/DeepfakeBench/training/config/detector/sbi.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_set = SBIDataset(config=config, mode='train')
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True, 
            num_workers=0,
            collate_fn=train_set.collate_fn,
        )
    from tqdm import tqdm
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        print(iteration)
        if iteration > 10:
            break