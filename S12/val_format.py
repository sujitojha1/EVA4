import io
import glob
import os
from shutil import move
from os.path import join
from os import listdir, rmdir
import random
random.seed(128)


target_folder = './tiny-imagenet-200/val/'
#test_folder   = './tiny-imagenet-200/test/'

#os.mkdir(test_folder)
val_dict = {}
with open('./tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        val_dict[split_line[0]] = split_line[1]
        
paths = glob.glob('./tiny-imagenet-200/val/images/*')
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    if not os.path.exists(target_folder + str(folder)):
        os.mkdir(target_folder + str(folder))
        os.mkdir(target_folder + str(folder) + '/images')
        
        
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    dest = target_folder + str(folder) + '/images/' + str(file)
    move(path, dest)
    
rmdir('./tiny-imagenet-200/val/images')

path = 'tiny-imagenet-200/'
list1 = range(0, 500)

id_list = []
for i, line in enumerate(open(path + 'wnids.txt', 'r')):
    id_list.append(line.replace('\n', ''))

for id in id_list:
  print(id)
  sample_list = random.sample(list1, k=135)
  source_files = ['./tiny-imagenet-200/train/'+str(id)+'/images/'+str(id)+'_'+str(file_num)+'.JPEG' for file_num in sample_list]
  dest_files = ['./tiny-imagenet-200/val/'+str(id)+'/images/'+str(id)+'_'+str(file_num)+'.JPEG' for file_num in sample_list]

  for (src_file, dest_file) in zip(source_files, dest_files): 
     move(src_file, dest_file)