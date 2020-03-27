'''Plotting Utility.

Grad-CAM implementation in Pytorch

Reference:
[1] xyz
[2] xyz
'''

import matplotlib.pyplot as plt
import numpy as np
import torch

from .grad_cam import *

def plot_misclassified_images(model, device, dataloader, classes):

    counter=1
    fig = plt.figure(figsize=(10, 10))
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
                              
        output = model(data)
        # convert output probabilities to predicted class
        _, preds = torch.max(output, 1)
        images = denormalize(data,mean=(0.4914, 0.4822, 0.4465),std=(0.2023, 0.1994, 0.2010)).cpu().numpy()

        
        #print(np.arange(len(preds.cpu().numpy())))

        

        for idx in np.arange(len(preds.cpu().numpy())):
          if counter < 26:
            if preds[idx]!=target[idx]:
                ax = fig.add_subplot(5, 5, counter, xticks=[], yticks=[])

                img = images[idx]
                npimg = np.transpose(img,(1,2,0))
                ax.imshow(npimg, cmap='gray')
                ax.set_title("act={}\npred={}".format(str(classes[target[idx].item()]), str(classes[preds[idx].item()])),
                            color= "red")
            
                counter+=1
          else:
            break
            
    fig.tight_layout()  
    plt.show()


def plot_misclassified_images_w_gradcam(model, device, dataloader, classes):

    # initialize a model, model_dict and gradcam
    resnet = model
    resnet.eval()
    gradcam = GradCAM.from_config(model_type='resnet', arch=resnet, layer_name='layer4')


    counter=1
    fig = plt.figure(figsize=(15, 10))
    for data, target in dataloader:
        images, labels = data.to(device), target.to(device)
                              
        output = model(images)
        # convert output probabilities to predicted class
        _, preds = torch.max(output, 1)
        #images = denormalize(data,mean=(0.4914, 0.4822, 0.4465),std=(0.2023, 0.1994, 0.2010)).cpu().numpy()

        for idx in np.arange(len(preds.cpu().numpy())):
          if counter < 51:
            if preds[idx]!=target[idx]:
                

                img = images[idx]
                lbl = labels.cpu().numpy()[idx]

                # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                img = img.unsqueeze(0).to(device)
                org_img = denormalize(img,mean=(0.4914, 0.4822, 0.4465),std=(0.2023, 0.1994, 0.2010))

                # get a GradCAM saliency map on the class index 10.
                mask, logit = gradcam(img, class_idx=lbl)

                ax = fig.add_subplot(5, 10, counter, xticks=[], yticks=[])
                npimg = np.transpose(org_img[0].cpu().numpy(),(1,2,0))
                ax.imshow(npimg, cmap='gray')
                ax.set_title("Label={}".format(str(classes[lbl])))
                counter+=1

                ax = fig.add_subplot(5, 10, counter, xticks=[], yticks=[])
                # make heatmap from mask and synthesize saliency map using heatmap and img
                heatmap, cam_result = visualize_cam(mask, org_img, alpha=0.4)
                npimg = np.transpose(cam_result,(1,2,0))
                ax.imshow(npimg, cmap='gray')
                ax.set_title("pred={}".format(str(classes[preds[idx].item()])),
                            color= "red")
                

            
                counter+=1
          else:
            break
            
    fig.tight_layout()  
    plt.show()
    
def plot_train_test_acc_loss(train,test):
    
    
    fig, axs = plt.subplots(2,1,figsize=(10,10))
    x = np.linspace(0.0, 50.0, num=len(train.train_losses))

    axs[0].plot(x,train.train_losses, label='Training')
    axs[0].set_ylabel("Training Loss", fontsize=16)
    axs[0].tick_params(axis='y', labelsize=14)
    axs[0].tick_params(axis='x', labelsize=14)


    axs[1].plot(x,train.train_acc)
    axs[1].set_ylabel("Accuracy", fontsize=16)
    axs[1].set_xlabel("Epoch", fontsize=16)
    axs[1].tick_params(axis='y',  labelsize=14)
    axs[1].tick_params(axis='x',  labelsize=14)

    color = 'tab:red'
    axs0_sec = axs[0].twinx() 
    x = np.linspace(0.0, 50.0, num=len(test.test_losses))

    axs0_sec.plot(x,test.test_losses, color=color,label='Test')
    

    axs0_sec.set_ylabel("Test Loss", fontsize=16, color=color)
    axs0_sec.tick_params(axis='y', labelcolor=color, labelsize=14)


    #axs1_sec = axs[1].twinx() 
    axs[1].plot(x,test.test_acc, color=color)

    plt.legend()
    plt.tight_layout()
    plt.show()