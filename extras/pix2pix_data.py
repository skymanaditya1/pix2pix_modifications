# Transforms the data into the data format required by the pix2pix model 
# Resizes the image to the dimension 256x256 and concatenates the images along the width dimension
import os.path as osp
from glob import glob 
import os
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt

root_dir = '/ssd_scratch/cvit/aditya1'

perturbed_image_dir = osp.join(root_dir, 'CelebAPerturbed')
original_image_dir = osp.join(root_dir, 'CelebA-HQ-img')

output_image_dir = osp.join(root_dir, 'CelebAPaired')
os.makedirs(output_image_dir, exist_ok=True)

perturbed_images = glob(perturbed_image_dir + '/*.jpg')
print(f'Starting generating paired data')

for perturbed_image in tqdm(perturbed_images):
    # Find the corresponding original image
    original_image = osp.join(original_image_dir, osp.basename(perturbed_image))
    # print(f'Original image : {original_image}, {os.path.isfile(original_image)}')
    perturbed_img = cv2.cvtColor(cv2.imread(perturbed_image), cv2.COLOR_BGR2RGB)
    original_img = cv2.cvtColor(cv2.imread(original_image), cv2.COLOR_BGR2RGB)

    # print(f'Original image : {original_img.shape}, perturbed image : {perturbed_img.shape}')

    perturbed_img_resized = cv2.resize(perturbed_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC) # dimension 256x256
    original_img_resized = cv2.resize(original_img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC) # dimension 256x256

    combined_img = cv2.hconcat([perturbed_img_resized, original_img_resized])
    # cv2.imwrite(osp.join(output_image_dir, osp.basename(perturbed_image)), combined_img)
    plt.imsave(osp.join(output_image_dir, osp.basename(perturbed_image)), combined_img)