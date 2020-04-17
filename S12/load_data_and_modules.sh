# Load modules/models from EVA course
rm -rf EVA4
git clone -b rev4 https://github.com/sujitojha1/EVA4 -q --quiet
echo "Loaded EVA4 Modules and Models"

# Installing latest Albumentation library
pip install -U git+https://github.com/albu/albumentations -q --quiet
#pip install apex -q

echo "Loaded EVA4 Modules and Models"
wget -nc http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
# rm -r ./tiny-imagenet-200/test
# python3 val_format.py
# find . -name "*.txt" -delete
# mkdir models
# cp -r tiny-imagenet-200 tiny-224
# python3 resize.py