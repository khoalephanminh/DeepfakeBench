import sys
import cv2
import os
from skimage.metrics import structural_similarity as ssim

# def calculate_ssim(image1, image2):
#     # Convert images to grayscale
#     gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#     gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

#     # Compute SSIM between the two images
#     score, _ = ssim(gray_image1, gray_image2, full=True)
#     return score

def calculate_ssim(image1, image2):
    # Ensure the images have 3 channels
    assert image1.shape == image2.shape and image1.shape[2] == 3, "Images must have the same shape and be 3-channel"

    # Split the images into their respective channels
    channels1 = cv2.split(image1)
    channels2 = cv2.split(image2)

    # Compute SSIM for each channel and average the results
    ssim_scores = [ssim(ch1, ch2, full=True)[0] for ch1, ch2 in zip(channels1, channels2)]
    average_ssim = sum(ssim_scores) / len(ssim_scores)

    return average_ssim

def resize_image(image, size):
    return cv2.resize(image, size)

def find_best_match(sample_image_path, folder_path):
    # Read the sample image
    sample_image = cv2.imread(sample_image_path)

    if sample_image is None:
        print(f"Error: Could not open sample image {sample_image_path}")
        return

    # Resize the sample image to a fixed size (for example, 256x256)
    target_size = (256, 256)
    sample_image = resize_image(sample_image, target_size)

    best_score = -1
    best_match = None
    best_match_path = None
    best_match_folder = None

    cnt = 0
    # Iterate through all files and subdirectories in the folder
    for root, dirs, files in os.walk(folder_path):
        cnt += 1
        sys.stdout.write(f'\rcnt={cnt} best_so_far={best_score} best_match={best_match_folder}')
        for filename in files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            
            # Full path to the current image
            image_path = os.path.join(root, filename)

            # Read the current image
            current_image = cv2.imread(image_path)

            if current_image is None:
                print(f"\nWarning: Could not open image {filename}")
                continue

            # Resize the current image to the same size as the sample image
            current_image = resize_image(current_image, target_size)

            # Calculate SSIM
            score = calculate_ssim(sample_image, current_image)

            # Update the best match if this image has a higher score
            if score > best_score:
                best_score = score
                best_match = filename
                best_match_path = image_path
                best_match_folder = os.path.basename(root)

    if best_match:
        print(f"\nBest match: {best_match} in {best_match_path} with SSIM score of {best_score}")
    else:
        print("\nNo matching images found.")

# Example usage
folder_path = '/raid/dtle/deepfake/DeepfakeBench/datasets/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/frames'
sample_image_paths = ['./figures/sbi_grad_1.jpg', './figures/sbi_grad_2.jpg', './figures/sbi_grad_3.jpg', './figures/sbi_grad_4.jpg']
for sample_image_path in sample_image_paths:
    find_best_match(sample_image_path, folder_path)
