'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2024-01-26

The code is designed for self-blending method (SBI, CVPR 2024).
'''
import os
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
from training.dataset.sbi_api import SBI_API
from .utils.p2p_models import *
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

cuda = True if torch.cuda.is_available() else False
# if os.path.exists('./sbi_p2p_imgs'):
#     shutil.rmtree('./sbi_p2p_imgs')
# os.makedirs('./sbi_p2p_imgs', exist_ok=True)

class SBIP2PDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)
        
        # Get real lists
        # Fix the label of real images to be 0
        self.real_imglist = [(img, label) for img, label in zip(self.image_list, self.label_list) if label == 0]

        # Init SBI
        self.sbi = SBI_API(phase=mode,image_size=config['resolution'])

        # Init data augmentation method
        self.transform = self.init_data_aug_method()

        # Define the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("device=", self.device)
        # self.Tensor = torch.FloatTensor if self.device == torch.device("cpu") else torch.cuda.FloatTensor
        # self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.Tensor = torch.FloatTensor

        self.generator = GeneratorUNet()
        self.discriminator = Discriminator()

        # self.generator = self.generator.cuda()
        # self.discriminator = self.discriminator.cuda()

        self.generator.load_state_dict(torch.load("/raid/dtle/deepfake/PyTorch-GAN/implementations/pix2pix/saved_models/facades/generator_190.pth"))
        self.discriminator.load_state_dict(torch.load("/raid/dtle/deepfake/PyTorch-GAN/implementations/pix2pix/saved_models/facades/discriminator_190.pth"))
        

    def pix2pix_format(self, img_f, img_r, hr_shape=(256,256)):
        # dd = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("dd=", dd)
        
        # print("img_types=", type(img_f), type(img_r))
        img_f_pil = Image.fromarray(img_f)
        img_r_pil = Image.fromarray(img_r)

        # print("img_f_pil=", img_f_pil)
        hr_transform = transforms.Compose(
            [
                transforms.Resize(hr_shape, Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        img_lr = hr_transform(img_f_pil)
        img_hr = hr_transform(img_r_pil)

        # print("img_lr_shape=", img_lr.shape)
        # print("img_lr=", img_lr)
        # print("is_cuda fake image:", img_lr.is_cuda)
        # img_lr = self.normalize(self.to_tensor(img_f))
        # img_hr = self.normalize(self.to_tensor(img_r))
        # print("shape=", img_lr.shape, img_hr.shape)
        # img_lr = img_lr.to('cuda')
        # img_hr = img_hr.to(self.device)
        return {"lr": img_lr, "hr": img_hr}

    def pix2pix_process(self, img_f, img_r, generator, discriminator):
        resize_380 = transforms.Resize((380, 380), interpolation=Image.BICUBIC)
        # print("img_f_shape=", img_f.shape)
        # print("img_f=", img_f)
        imgs = self.pix2pix_format(img_f, img_r)
        real_A = imgs["lr"].type(self.Tensor)
        real_B = imgs["hr"].type(self.Tensor)

        # real_A = Variable(imgs["lr"]).type(torch.cuda.FloatTensor)
        # real_B = Variable(imgs["hr"]).type(torch.cuda.FloatTensor)
        real_A = real_A.unsqueeze(0)
        real_B = real_B.unsqueeze(0)
        
        fake_B = generator(real_A)

        real_A = real_A.squeeze(0)
        real_B = real_B.squeeze(0)
        fake_B = fake_B.squeeze(0)

        # print("real_A_before_shape=", real_A.shape)
        # print("real_A_before=", real_A)

        real_A = resize_380(real_A)
        real_B = resize_380(real_B)
        fake_B = resize_380(fake_B)
        # print("real_A_shape=", real_A.shape)
        # print("real_A=", real_A)

        # print("shape_p2p_return=", real_A.shape, real_B.shape, fake_B.shape)

        return {"sbi": real_A, "real": real_B, "p2p": fake_B}

    # def denormalize(self, tensors):
    #     """ Denormalizes image tensors using mean and std """
    #     for c in range(3):
    #         tensors[:, c].mul_(std[c]).add_(mean[c])
    #     # return torch.clamp(tensors, 0, 1)
    #     print("tensors=", tensors)
    #     return tensors
    def denormalize(self, tensors):
        # Ensure mean and std are torch tensors
        mean_t = torch.tensor(mean).view(3, 1, 1)  # Shape (C, 1, 1)
        std_t = torch.tensor(std).view(3, 1, 1)  # Shape (C, 1, 1)
        
        # Reverse normalization
        tensors = tensors * std_t + mean_t  # Correct broadcasting
        return torch.clamp(tensors, 0, 1)
        # print("tensors=", tensors)
        return tensors

    def __getitem__(self, index):
        # Get the real image paths and labels
        real_image_path, real_label = self.real_imglist[index]

        # Get the landmark paths for real images
        real_landmark_path = real_image_path.replace('frames', 'landmarks').replace('.png', '.npy')
        landmark = self.load_landmark(real_landmark_path).astype(np.int32)

        # Load the real images
        real_image = self.load_rgb(real_image_path)
        real_image = np.array(real_image)  # Convert to numpy array


        # Generate the corresponding SBI sample
        fake_image, real_image = self.sbi(real_image, landmark)
        # print("fake_image_shape=", fake_image.shape)
        # print("fake_image=", fake_image)
        # print("shape_real_image=", real_image.shape)
        if fake_image is None:
            fake_image = deepcopy(real_image)
            fake_label = 0
        else:
            fake_label = 1

        #-------remove from here---------
        # gan_imgs = self.pix2pix_process(fake_image, real_image, self.generator, self.discriminator)
        # fake_image = gan_imgs['sbi']
        # real_image = gan_imgs['real']
        # p2p_image = gan_imgs['p2p']
        
        # # print("fake_image_shape=", fake_image.shape)
        # # print("fake_image=", fake_image)
        
        # fake_image = self.denormalize(fake_image)
        # real_image = self.denormalize(real_image)
        # p2p_image = self.denormalize(p2p_image)

        # # Convert PyTorch tensors to NumPy arrays and transpose dimensions
        # fake_image = fake_image.permute(1, 2, 0).cpu().detach().numpy()
        # real_image = real_image.permute(1, 2, 0).cpu().detach().numpy()
        # p2p_image = p2p_image.permute(1, 2, 0).cpu().detach().numpy()

        # fake_image = (fake_image * 255).astype(np.uint8)
        # real_image = (real_image * 255).astype(np.uint8)
        # p2p_image = (p2p_image * 255).astype(np.uint8)
        #-------end of remove---------

        # # ----remove from here -----

        # Convert RGB to BGR
        # fake_image_bgr = cv2.cvtColor(fake_image, cv2.COLOR_RGB2BGR)
        # real_image_bgr = cv2.cvtColor(real_image, cv2.COLOR_RGB2BGR)
        # p2p_image_bgr = cv2.cvtColor(p2p_image, cv2.COLOR_RGB2BGR)

        # # # if index == 604:
        # # #     print("idx, landmark shape: ", index, landmark.shape)
        # # #     print("landmark=", landmark)

        # # # Draw landmark points on the images
        # # for point in landmark:
        # #     cv2.circle(fake_image_bgr, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
        # #     cv2.circle(real_image_bgr, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)

        # # # Save images using cv2.imwrite
    
        # cv2.imwrite(f'./sbi_p2p_imgs/{index}_fake.png', fake_image_bgr)
        # cv2.imwrite(f'./sbi_p2p_imgs/{index}_real.png', real_image_bgr)
        # cv2.imwrite(f'./sbi_p2p_imgs/{index}_p2p.png', p2p_image_bgr)

        # # ---end of remove----

        # To tensor and normalize for fake and real images
        fake_image_trans = self.normalize(self.to_tensor(fake_image))
        real_image_trans = self.normalize(self.to_tensor(real_image))
        # p2p_image_trans = self.normalize(self.to_tensor(p2p_image))

        # random = np.random.rand()
        # if random < 0.5:
        #     return {"fake": (fake_image_trans, fake_label), 
        #         "real": (real_image_trans, real_label),
        #         "p2p": (p2p_image_trans, fake_label)}
        # else: 
        #     return {"fake": (p2p_image_trans, fake_label), 
        #         "real": (real_image_trans, real_label),
        #         "p2p": (fake_image_trans, fake_label)}

        return {"fake": (fake_image_trans, fake_label), 
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