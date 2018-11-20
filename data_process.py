import numpy as np
import os, os.path, random
from PIL import Image


def crop_image():
    img_path = 'Summer 2018 Pics'
    print (os.listdir(img_path))

    for subdir in os.listdir(img_path):
        subdir_path = os.path.join(img_path, subdir)
        if subdir == 'A':
            for f in os.listdir(subdir_path):
                ext = os.path.splitext(f)[1]
                if ext == '.jpg':
                    new_name = os.path.splitext(f)[0] + '_cropped' + '.png'
                    save_path = os.path.join(subdir_path, new_name)
                    image = Image.open(os.path.join(subdir_path, f))
                    box = (0, 217, 945, 625)
                    crop = image.crop(box)
                    crop.save(save_path)

        elif subdir == 'B':
            for f in os.listdir(subdir_path):
                ext = os.path.splitext(f)[1]
                if ext == '.jpg':
                    new_name = os.path.splitext(f)[0] + '_cropped' + '.png'
                    save_path = os.path.join(subdir_path, new_name)
                    image = Image.open(os.path.join(subdir_path, f))
                    box = (0, 217, 945, 625)
                    crop = image.crop(box)
                    crop.save(save_path)
        elif subdir == 'C':
            for f in os.listdir(subdir_path):
                ext = os.path.splitext(f)[1]
                if ext == '.jpg':
                    new_name = os.path.splitext(f)[0] + '_cropped' + '.png'
                    save_path = os.path.join(subdir_path, new_name)
                    image = Image.open(os.path.join(subdir_path, f))
                    box = (0, 217, 945, 625)
                    crop = image.crop(box)
                    crop.save(save_path)



crop_image()