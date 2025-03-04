import os
import cv2
import numpy as np

def crop(image, crop_percent, resolution = (256, 256)):
        """
        Randomly crops an image by a given percentage from an already enlarged image (1.3x the original).

        :param image: Input image (numpy array) assumed to be 1.3x the original size.
        :param crop_percent: Fraction of the original size to keep (e.g., 0.2 means cropping 20%).
        :return: Cropped image.
        """
        H_curr, W_curr = image.shape[:2]  # Get current size

        # Compute new crop size
        scale_factor = (1 + crop_percent) / 1.3  # Example: (1.2 / 1.3) for 20% crop
        H_new, W_new = int(H_curr * scale_factor), int(W_curr * scale_factor)

        # Compute the amount to crop from each side
        crop_top = (H_curr - H_new) // 2
        crop_bottom = H_curr - H_new - crop_top
        crop_left = (W_curr - W_new) // 2
        crop_right = W_curr - W_new - crop_left

        # Perform cropping
        cropped_image = image[crop_top:H_curr - crop_bottom, crop_left:W_curr - crop_right]
        cropped_image = cv2.resize(cropped_image, resolution, interpolation=cv2.INTER_CUBIC) # hard coding 380 for now

        return cropped_image

dataset_folder = '/raid/dtle/deepfake/DeepfakeBench/datasets/rgb/Celeb-DF-v2'
dataset_name = dataset_folder.split('/')[-1]
# recursively iterate each frame
for root, dirs, files in os.walk(dataset_folder):
    
    for file in files:
        if file.endswith('.png'):
            # read image
            image = cv2.imread(os.path.join(root, file))
            # crop image
            cropped_image = crop(image, 0.2)
            # save image
            save_path = os.path.join(root, file).replace(dataset_name, f'{dataset_name}-cropped0125')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, cropped_image)
            # print("save_path: ", save_path)