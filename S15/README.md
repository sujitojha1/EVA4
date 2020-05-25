# Problem Statement EVA 4, Session 15 : DepthMap & Mask Creation

## Assignment: Problem Statement

- Given an image with foreground objects and background image, predict the depth map as well as a mask for the foreground object. 
  - Modular code
  - Data management (Chaning data format, changing image size for training model, loss function design, data augmentation, saving checkpoints, time analysis, present result)

## Data Source
- Link [readme](https://github.com/sujitojha1/EVA4/blob/rev7/S14_15/README.md)
- Size, Shape info fg_bg : 224x224x3 fg : 65x65x4 mask : 224x224x4 Depth images: 112x112x3
- Ground Truth: Generated based of custom made dataset, mask and depth ground truth


## Understanding 
what is the input 224x224 fg_bg + 224x224 bg -> 112x112x1 depth map & 224x224x1 fg/bg
What should be metric ?? SSIM? 

## Intermediate Steps
  1. Load, unzip and calculate statistics - Understand the dataset [link](https://github.com/sujitojha1/EVA4/blob/rev8/S15/EVA4_S15_Solution_DenseDepth_step1.ipynb)
  2. Different problem resnet 18 with load state dictionary and tensorboard, [link](https://github.com/sujitojha1/EVA4/blob/rev8/S15/EVA4_S15_Solution_DenseDepth_step2.ipynb)
  3. Use the densedepth architecture - Understand the training setup (encoder & decoder) [link]()
  4. Recreate Rohan's architecture & flow [link]()
  5. Explore & iterate on different error functions (Not done)
  6. Architecture evaluation & code for intermediate plotting (Not done)

## Approach
  1. Image merging to creat channels
  2. Ricap kind data augmentations
  3. Volume 
  4. Encoder & Decoder

## Results

## Observations

## Learning Notes

## References
Keywords: Background Substraction, Change detection
1. Background Subtraction on Depth Videos with Convolutional Neural Networks [link](https://arxiv.org/pdf/1901.05676.pdf)
2. http://changedetection.net/
3. https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
4. https://github.com/kuangliu/pytorch-cifar/blob/master/main.py