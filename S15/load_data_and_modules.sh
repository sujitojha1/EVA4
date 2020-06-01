#!/bin/bash

# Installing latest Albumentation library
pip install -U git+https://github.com/albu/albumentations -q --quiet
echo "Pip installation of Albumentation done !!"

mkdir 'dataset'

echo "Copying files from google drive to runtime !!"
cp -r '/content/drive/My Drive/EVA4/S15/fg_bg.zip' dataset/fg_bg.zip 
cp -r '/content/drive/My Drive/EVA4/S15/mask.zip' dataset/mask.zip 
cp -r '/content/drive/My Drive/EVA4/S15/depth_map.zip' dataset/depth_map.zip 