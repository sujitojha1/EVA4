# Problem Statement EVA 4, Session10 : CIFAR 10

## Target:

-  Make sure to Add CutOut to your code. It should come from your transformations (albumentations)
-  Use this repo: https://github.com/davidtvs/pytorch-lr-finder (Links to an external site.)
    -  Move LR Finder code to your modules
    -  Implement LR Finder (for SGD, not for ADAM)
    -  Implement ReduceLROnPlatea: https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau (Links to an external site.)
-  Find best LR to train your model
-  Use SDG with Momentum
-  Train for 50 Epochs.
-  Show Training and Test Accuracy curves
-  Target Accuracy is 88%
-  Run GradCAM on the any 25 misclassified images. Make sure you mention what is the prediction and what was the ground truth label.
-  Submit answers to S10-Assignment-Solution.

## Results

-  Total Parameters = 11,173,962
-  After 50 Epochs, Final Train Accuracy = 95.13%
-  After 50 Epochs, Final Test Accuracy = 92.65%


## Analysis

-  Target accuracy > 88% using Reset18 architecture. Model performance is good as train and test accuracy is small.
