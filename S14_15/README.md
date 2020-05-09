# Problem Statement EVA 4, Session 14 & 15 : DepthMap

Assignment: 

- You must have 100 background, 100x2 (including flip), and you randomly place the foreground on the background 20 times, you have in total 100x200x20 images. 

In total you MUST have:
- 400k fg_bg images
- 400k depth images
- 400k mask images
generated from:
- 100 backgrounds
- 100 foregrounds, plus their flips
- 20 random placement on each background.

Now add a readme file on GitHub for Project 15A:
- Create this dataset and share a link to GDrive (publicly available to anyone) in this readme file. 
- Add your dataset statistics:
  - Kinds of images (fg, bg, fg_bg, masks, depth)
  - Total images of each kind
  - The total size of the dataset
  - Mean/STD values for your fg_bg, masks and depth images
- Show your dataset the way I have shown above in this readme
- Explain how you created your dataset
  - how were fg created with transparency
  - how were masks created for fgs
  - how did you overlay the fg over bg and created 20 variants
  - how did you create your depth images? 
  - Add the notebook file to your repo, one which you used to create this dataset
  - Add the notebook file to your repo, one which you used to calculate statistics for this dataset
