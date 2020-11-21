# Computes grid feature at downsampled point

import torch
import json
import numpy as np
import math
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lower', type=int)
parser.add_argument('--higher', type=int)
args = parser.parse_args()

with open('', 'r') as f:
     dataset = json.load(f)

grid_dir = './gridfeats/'
size_dir = './img_sizes/'
out_dir = './gridfeats_pt'

img_ids = list(vqamb.keys())

cnt = 0
for img_id in img_ids[args.lower:args.higher]:

     cnt += 1
     if cnt % 100 == 0:
          print(cnt)

     grid_path = grid_dir + str(img_id) + '.pt'
     grid_img = torch.load(grid_path).squeeze()
     gridh, gridw = grid_img.shape[1:]
     
     size_path = size_dir + str(img_id) + '.npy'
     size_img = np.load(size_path)
     h, w = size_img[0], size_img[1]

     min_size = min(w, h)
     max_size = max(w, h)

     # compute image scale factor
     img_scale = 600 / min_size
     if img_scale * max_size > 1000:
          img_scale = 1000 / max_size

     neww = int(w * img_scale + 0.5)
     newh = int(h * img_scale + 0.5)


     # pad to nearest multiples of 32, reflects how model downsamples
     wmod = neww % 32
     hmod = newh % 32
     
     if wmod != 0:
          neww += (32 - wmod)
     
     if hmod != 0:
          newh += (32 - hmod) 

     for qa in dataset[img_id]:
          for pt in qa['points']:
               x, y = pt['x'], pt['y']
               fname = out_dir + str(img_id) + ',' + str(x) + ',' + str(y) + '.pt'
               
               if os.path.exists(fname): continue

               new_x, new_y = x * (neww/w), y * (newh/h)
               
               # downsample point
               down_x, down_y = int(math.floor(new_x / 32)), int(math.floor(new_y / 32)) 
               grid_feat = grid_img[:, down_y, down_x]

               torch.save(grid_feat, fname)
