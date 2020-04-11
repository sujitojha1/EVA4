# Problem Statement EVA 4, Session10 : CIFAR 10

## Target:

*  Write a code that draws this curve (without the arrows). In submission, you'll upload your drawn curve and code for that
    * 11s11.png 
*  Write a code which
  1. uses this new ResNet Architecture for Cifar10:
    * PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
    * Layer1 -
        * X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        * R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
        * Add(X, R1)
    * Layer 2 -
        * Conv 3x3 [256k]
        * MaxPooling2D
        * BN
        * ReLU
    * Layer 3 -
        * X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        * R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        * Add(X, R2)
    * MaxPooling with Kernel Size 4
    * FC Layer 
    * SoftMax
  2. Uses One Cycle Policy such that:
    * Total Epochs = 24
    * Max at Epoch = 5
    * LRMIN = FIND
    * LRMAX = FIND
    * NO Annihilation
  3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
  4. Batch size = 512
  5. Target Accuracy: 90%. 
  6. The lesser the modular your code is (i.e. more the code you have written in your Colab file), less marks you'd get. 
* Questions asked are:
  1. Upload the code you used to draw your ZIGZAG or CYCLIC TRIANGLE plot.
  2. Upload your triangle Plot which was drawn with your code.
  3. Upload the link to your GitHub copy of Colab Code. 
  4. Upload the github link for the model as described in A11. 
  5. What is your test accuracy?

## Results

*  Total Parameters = 6,573,120
*  After 24 Epochs, Final Train Accuracy = 95.75%
*  After 24 Epochs, Final Test Accuracy = 90.27%


## Analysis

*  Target accuracy > 90% using Reset18 new architecture. Model performance is slightly overfitting.