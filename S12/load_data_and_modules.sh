#!/bin/bash

# Installing latest Albumentation library
pip install -U git+https://github.com/albu/albumentations -q --quiet
echo "Pip installation of Albumentation done !!"

echo "Downloading data"
set echo off
wget -nc -q http://cs231n.stanford.edu/tiny-imagenet-200.zip 
unzip tiny-imagenet-200.zip

set echo on
rm -r ./tiny-imagenet-200/test
python3 EVA4/S12/val_format.py
# find . -name "*.txt" -delete