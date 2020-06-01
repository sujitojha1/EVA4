#!/bin/bash

# Installing latest Albumentation library
pip install -U git+https://github.com/albu/albumentations -q --quiet
pip install kornia --quiet
echo "Pip installation of Albumentation & Kornia done !!"

mkdir 'dataset'

echo "Copying files from google drive to runtime !!"
cp -r '/content/drive/My Drive/EVA4/S15/fg_bg.zip' dataset/fg_bg.zip 
cp -r '/content/drive/My Drive/EVA4/S15/mask.zip' dataset/mask.zip 
cp -r '/content/drive/My Drive/EVA4/S15/depth_map.zip' dataset/depth_map.zip 
cp -r '/content/drive/My Drive/EVA4/S15/filelist.csv' dataset/filelist.csv

!cp -r '/content/EVA4/S14_15/images/bk_grnd' dataset/
!mv 'dataset/bk_grnd' 'dataset/bg'