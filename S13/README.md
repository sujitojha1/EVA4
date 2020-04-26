# Problem Statement EVA 4, Session13 : YoloV3

Assignment: 

1. OpenCV Yolo: SOURCE (Links to an external site.)
  1. Run this above code on your laptop or Colab. 
  2. Take an image of yourself, holding another object which is there in COCO data set (search for COCO classes to learn). 
  3. Run this image through the code above. 
  4. Upload the link to GitHub implementation of this
  5. Upload the annotated image by YOLO. 
2. Training Custom Dataset on Colab for YoloV3
  1. Refer to this Colab File: LINK (Links to an external site.)
  2. Refer to this GitHub Repo (Links to an external site.)
  3. Collect a dataset of 500 images and annotate them. Please select a class for which you can find a YouTube video as well. Steps are explained in the readme.md file on GitHub.
  4. Once done:
    a. Download (Links to an external site.) a very small (~10-30sec) video from youtube which shows your class. 
    b. Use ffmpeg (Links to an external site.) to extract frames from the video. 
    c. Upload on your drive (alternatively you could be doing all of this on your drive to save upload time)
    d. Inter on these images using detect.py file. **Modify** detect.py file if your file names do not match the ones mentioned on GitHub. 
    e. `python detect.py --conf-thres 0.3 --output output_folder_name`
    f. Use ffmpeg (Links to an external site.) to convert the files in your output folder to video
    g. Upload the video to YouTube. 
  5. Share the link to your GitHub project with the steps as mentioned above
  6. Share the link of your YouTube video
  7. Share the link of your YouTube video on LinkedIn, Instagram, etc! You have no idea how much you'd love people complimenting you! 
