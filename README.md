# Introduction
This repo started as a collection of projects with the intent of getting familiar with pytorch and basic neural networks.
It's developed into a space where I can gain some insight into all things deep learning through different projects.
# Conda environment setup
At the root of this repo is spec-file.txt, this file contains all the information needed to set up a conda virtual environment identical to that used during development.
After installing conda and python navigate to the spec-file.txt directory using the anaconda prompt (base). To set up the virtual environment run

`conda create --name myenv --file spec-file.txt`

Check that the virtual The virtual environment is created successfully by running
  
`conda info --envs` 
  
or 
  
`coda env list`
  
If myenv shows on the list it is available for activation. To activate it run

`conda activate myenv`

This conda terminal can now execute the appropriate python files by running

`python fileiwanttoexecute.py`

# MNIST-Project
The MNIST project uses the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) of hand written digits to train a neural network. It is the "hello world" of deep learning projects 
so it seemed suitable to start here. Specifically I wanted to implement 
[3blue1brown](https://www.youtube.com/c/3blue1brown)'s description of a neural network in his [Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
 series. This model can reach an accuracy of 90% in 100 epochs with a batch size of 100 and a learning rate of 0.1.
 
## Description of important files
### pipeline.py

pipeline.py is the training pipeline. It creates and trains a neural network from scratch then saves the model state dict to
model.pth. At the end of every epoch the pipeline will print the current epoch and the current loss. The total accuracy of the
model is then evaluated against some test data and printed to the terminal.
as a small fast learning first project there is no validation dataset and not a lot of visualisation.

### image-vis.py

image-vis.py is an attempt to get some visualisation of the neural network trying to label images in real time. The python file
loads the trained neural network from model.pth, then it plots an image from the test dataset and shows what the neural networks 
guess for that image is.

## Insights
From image-vis.py we can see that some of the images where the neural network goes wrong are very understandable like poorly drawn 8's. 
Expecting a NN to be 100% accurate is just not feasable. 

# CIFAR-10-Project
The CIFAR-10-Project uses [pytorch's dataloader](http://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html) for the [CIFAR 10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). 
The dataset is composed of images of vehicles and animals to train a convolutional neural network. Many more interesting features and explorations were made in this 
project since I was now familiar with pytorch basics from the MNIST Project. The best I could get this specific model to train was to a total accuracy of about 64%.

## Description of important files

### pipeline.py

pipeline.py is the training pipeline. Some of it's features include
* Training a convolutional neural network from scratch
* (optional) Loading the parameters of an already trainded CNN and training from there
* Outputing key information during the end of training epochs to validation-info.csv where the live data can be viewed by running validation.py
* Plots and saves a confusion matrix of the final trained CNN
* Saves checkpoints of the CNN during training to ./CIFAR-10-Project/Checkpoints

the pipeline also prints training information during each epoch and gives an accuracy report during the end of training

### validation.py

validation.py creates a live plot of the accuracies of all the classes (dogs, cats, etc.) during training as well as the current epoch, loss, and total accuracy.
## Insights

### The effect of batch size on the loss landscape

The most interesting insight to come from this project was the effect that the batch size has on the batch size has on the loss landscape during training. 
The smaller the batch size the "noisier/spikier" the loss landscape, where a larger batch size "smoothens out" the loss landscape. Ideally the batch size will be selected
such that the loss landscape will be noisy enough to prevent the NN from "memorising" the data.

![CNN trained with batch size 64](https://github.com/DimitrisGahtidis/Deep-Learning-Projects/blob/master/CIFAR-10-Project/bs64lr0%2C001ep256.png)
![CNN trained with batch size 28](https://github.com/DimitrisGahtidis/Deep-Learning-Projects/blob/master/CIFAR-10-Project/bs128lr0%2C001ep256.png)

### Confusion matrices are usefull

It suprised me how useful it is to convert the neural networks performance to visual information that I can interperet as a human. From a quick look to a confusion matrix we can see subsets of the confusion matrix that are lighter than others. For example the light 6x6 square in the middle of the confusion matrix indicates that when presented with the image of an animal the CNN is significantly more likely to confuse it with another animal than with another vehicle. This suggests that the CNN has learned to distinguish the difference between animals and vehicles in general. We can see similar 2x2 hotspots on the top right and bottom left of the confusion matrix, these similarly indicate that when presented with the image of a vehicle the CNN will likely confuse it with another vehicle than an animal. Thi is something I would never have picked up on by looking at the 
accuracy percentage printed in the terminal screen by the end of the test stage.

![confusion matrix of a CNN that is 64% accurate](https://github.com/DimitrisGahtidis/Deep-Learning-Projects/blob/master/CIFAR-10-Project/bs64.png)
