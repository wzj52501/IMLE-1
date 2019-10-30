
import os
from tqdm import tqdm
import cv2
import numpy as np

def rgb_float(image_content):
    return image_content / 127.5 - 1

def float_rgb(image_content):
    return (image_content + 1) * 127.5

def load_image(self,eval = False):
    if(eval):
        image_dir_name = os.path.join('data','eval_faces')
    else:
        image_dir_name = os.path.join('data','random_faces')
    image_list = os.listdir(image_dir_name)
    image_content = []
    image_name = []
    image_n = []
    image_z = []
    for name in tqdm(image_list):
        cell_content = cv2.imread(os.path.join(image_dir_name,name)).astype(np.float32)
        cell_content = cv2.resize(cell_content,(self.image_height,self.image_weight))
        image_content.append(cell_content)
        image_name.append(name)

    return rgb_float(np.array(image_content)), image_name

def make_image(input,name_list):
    write_dir = 'eval'
    image_content = float_rgb(input).astype(np.uint8)
    index = 0
    for cell in image_content:
        print(name_list[index].decode())
        cv2.imwrite(os.path.join(write_dir,name_list[index].decode()), cell)
        index += 1

