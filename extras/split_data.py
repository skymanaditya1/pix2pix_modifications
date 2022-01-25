# Split the dataset into train, test, and val 
from glob import glob
import random
import os
import os.path as osp
import shutil

from sklearn.model_selection import train_test_split

BASE_DIR = '/ssd_scratch/cvit/aditya1/CelebAPaired'

TRAIN_DIR = osp.join(BASE_DIR, 'train')
VAL_DIR = osp.join(BASE_DIR, 'val')
TEST_DIR = osp.join(BASE_DIR, 'test')

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# move files in file_list to destination directory
def move_files(file_list, dest_dir):
    for file in file_list:
        shutil.move(file, dest_dir)

def split_dataset():
    images = glob(BASE_DIR + '/*.jpg')
    train, test = train_test_split(images, test_size=0.01, shuffle=True, random_state=8)
    train, val = train_test_split(train, test_size=0.01, shuffle=True, random_state=8)

    move_files(train, TRAIN_DIR)
    move_files(val, VAL_DIR)
    move_files(test, TEST_DIR)

if __name__ == '__main__':
    split_dataset()