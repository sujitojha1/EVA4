# Problem Statement EVA 4, Session 15 : DepthMap & Mask Creation

## Assignment: Problem Statement

- Given an image with foreground objects and background image, predict the depth map as well as a mask for the foreground object. 
  - Modular code
  - Data management (Chaning data format, changing image size for training model, loss function design, data augmentation, saving checkpoints, time analysis, present result)

## Data Source
Link
Size, Shape
Ground Truth


## Understanding 
what is the input 224x224 fg_bg + 224x224 bg -> 112x112x1 depth map & 224x224x1 fg/bg
What should be metric ?? SSIM? 

## Intermediate Steps
  1. Load, unzip and calculate statistics - Understand the dataset
  2. Different problem resnet 18 with load state dictionary and tensorboard
  3. Use the densedepth architecture - Understand the training setup
  4. Recreate Rohan's architecture & flow
  5. Explore & iterate on different error functions
  6. Architecture evaluation & code for intermediate plotting

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