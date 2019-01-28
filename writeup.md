# **Traffic Light Detector ** 

Credits for data:
https://github.com/olegleyz/traffic-light-classification

### Dependencies:
https://github.com/udacity/CarND-Term1-Starter-Kit

### Launch instructions
1. Launch MINGW64 Docker Terminal
2. docker run -it --rm -p 8888:8888 -v ${PWD}:/src udacity/carnd-term1-starter-kit
3. Launch browser at http://[ip-of-container]:8888


python scripts/retrain.py --image_dir train_pics --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2 --flip_left_right 5 --random_crop 5 --random_scale 5 --random_brightness 5 --how_many_training_steps 100

#################
Useful links:
https://www.tensorflow.org/hub/
MobileNet v1: https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html
MobileNet v2: https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html
Lab Retraining: https://codelabs.developers.google.com/codelabs/tensorflow-for-poets
Retraining in tensorflow 1.3.0: https://github.com/tensorflow/tensorflow/tree/r1.3/tensorflow/examples/image_retraining
Tutorial on retraining: https://www.tensorflow.org/hub/tutorials/image_retraining

SSD+MobileNetv1 https://github.com/GoogleCloudPlatform/tensorflow-object-detection-example
Retrainc cmd:
python scripts/retrain.py --image_dir site_pics --architecture mobilenet_1.0_224  --flip_left_right 15 --random_crop 15 --random_scale 10 --how_many_training_steps 1000
Retrain log:
INFO:tensorflow:2019-01-25 19:32:29.653230: Step 900: Train accuracy = 87.0%
INFO:tensorflow:2019-01-25 19:32:29.653230: Step 900: Cross entropy = 0.455239
INFO:tensorflow:2019-01-25 19:32:29.725454: Step 900: Validation accuracy = 74.0% (N=100)
INFO:tensorflow:2019-01-25 19:34:17.145985: Step 910: Train accuracy = 84.0%
INFO:tensorflow:2019-01-25 19:34:17.146961: Step 910: Cross entropy = 0.455106
INFO:tensorflow:2019-01-25 19:34:17.268961: Step 910: Validation accuracy = 77.0% (N=100)
INFO:tensorflow:2019-01-25 19:36:06.044179: Step 920: Train accuracy = 77.0%
INFO:tensorflow:2019-01-25 19:36:06.045156: Step 920: Cross entropy = 1.272055
INFO:tensorflow:2019-01-25 19:36:06.266708: Step 920: Validation accuracy = 65.0% (N=100)
INFO:tensorflow:2019-01-25 19:37:54.152791: Step 930: Train accuracy = 95.0%
INFO:tensorflow:2019-01-25 19:37:54.152791: Step 930: Cross entropy = 0.148111
INFO:tensorflow:2019-01-25 19:37:54.297238: Step 930: Validation accuracy = 87.0% (N=100)
INFO:tensorflow:2019-01-25 19:39:42.900684: Step 940: Train accuracy = 76.0%
INFO:tensorflow:2019-01-25 19:39:42.900684: Step 940: Cross entropy = 1.116211
INFO:tensorflow:2019-01-25 19:39:43.043212: Step 940: Validation accuracy = 68.0% (N=100)
INFO:tensorflow:2019-01-25 19:41:32.383501: Step 950: Train accuracy = 88.0%
INFO:tensorflow:2019-01-25 19:41:32.384478: Step 950: Cross entropy = 0.252802
INFO:tensorflow:2019-01-25 19:41:32.521117: Step 950: Validation accuracy = 82.0% (N=100)
INFO:tensorflow:2019-01-25 19:43:20.028514: Step 960: Train accuracy = 92.0%
INFO:tensorflow:2019-01-25 19:43:20.029492: Step 960: Cross entropy = 0.220961
INFO:tensorflow:2019-01-25 19:43:20.161248: Step 960: Validation accuracy = 84.0% (N=100)
INFO:tensorflow:2019-01-25 19:45:09.066277: Step 970: Train accuracy = 83.0%
INFO:tensorflow:2019-01-25 19:45:09.066277: Step 970: Cross entropy = 0.529786
INFO:tensorflow:2019-01-25 19:45:09.202951: Step 970: Validation accuracy = 74.0% (N=100)
INFO:tensorflow:2019-01-25 19:46:57.780982: Step 980: Train accuracy = 74.0%
INFO:tensorflow:2019-01-25 19:46:57.781959: Step 980: Cross entropy = 1.528041
INFO:tensorflow:2019-01-25 19:46:57.943975: Step 980: Validation accuracy = 67.0% (N=100)
INFO:tensorflow:2019-01-25 19:48:45.941322: Step 990: Train accuracy = 91.0%
INFO:tensorflow:2019-01-25 19:48:45.941322: Step 990: Cross entropy = 0.321146
INFO:tensorflow:2019-01-25 19:48:46.094556: Step 990: Validation accuracy = 84.0% (N=100)
INFO:tensorflow:2019-01-25 19:50:23.666267: Step 999: Train accuracy = 94.0%
INFO:tensorflow:2019-01-25 19:50:23.666267: Step 999: Cross entropy = 0.156539
INFO:tensorflow:2019-01-25 19:50:23.796075: Step 999: Validation accuracy = 90.0% (N=100)





#######
Actual command for retrain our network

python scripts/retrain.py --image_dir train_pics --architecture mobilenet_1.0_224  --flip_left_right 5 --random_crop 5 --random_scale 5 --random_brightness 5 --how_many_training_steps 1000

INFO:tensorflow:2019-01-05 13:37:44.945261: Step 980: Train accuracy = 99.0%
INFO:tensorflow:2019-01-05 13:37:44.945261: Step 980: Cross entropy = 0.051631
INFO:tensorflow:2019-01-05 13:37:45.015074: Step 980: Validation accuracy = 98.0% (N=100)
INFO:tensorflow:2019-01-05 13:39:25.872570: Step 990: Train accuracy = 100.0%
INFO:tensorflow:2019-01-05 13:39:25.872570: Step 990: Cross entropy = 0.020044
INFO:tensorflow:2019-01-05 13:39:25.939391: Step 990: Validation accuracy = 99.0% (N=100)
INFO:tensorflow:2019-01-05 13:40:56.217310: Step 999: Train accuracy = 100.0%
INFO:tensorflow:2019-01-05 13:40:56.217310: Step 999: Cross entropy = 0.023648
INFO:tensorflow:2019-01-05 13:40:56.309064: Step 999: Validation accuracy = 99.0% (N=100)
INFO:tensorflow:Final test accuracy = 97.9% (N=194)


####################
Command for label_image

 python scripts/label_image.py --image=train_pics/green/001.jpg --graph=models/mobilenet_1.0_224.pb --labels=models/labels.txt --input_layer=input --output_layer=final_result --input_height=224 --input_width=224
 


tensorboard.exe --logdir /c/tmp/retrain_logs/


## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[imageYTrain]: ./writeup/ytrain_hist.png "Training set histogram"
[imageYTrainAug]: ./writeup/ytrain_hist.png "Augmented training set histogram"
[imageEffects]: ./writeup/effects.png "Preprocessing effects"
[imageRndEffects]: ./writeup/random_effects.png "Random effects"
[imageWeb]: ./writeup/webimages.png "Web images"

[//]: # (Other files References)

[tests]: ./writeup/tests.txt "Training iterations results"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data in the training set is divided into classes:

![alt text][imageYTrain]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I'm doing a preprocessing of two steps:
1. Convert image from RGB to Gray scale with the formula: 29.9% of Red, 58.7% of Green and 11.4% of Blue.
2. Normalize the grayscale image to be centered around the origin ( (image-128)/128 ), as seen in the previous lessons.

As a last step, I decided to generate additional data because, after some tests, I have seen the accuracy of my model increases a lot. 

To add more data to the the data set, I used the following techniques:
* Rotate the image with a random angle between -15 and 15 degrees.
* Translate the image in both edges with a random translation between [-2, 2] pixels.
* Perspective transformation, working as a zoom in into the image.
* Affine transformation.
* Bitwise not operation.

Other techniques applied in first stages were:
* Gaussian blur.
* Rotate the image in a range between [10, 350] degrees.
These techniques were affecting the accuracy of my model. Once removed, my model jumped from 92-93% to 96-97%. 

Finally, I decided to increase every class to the same level, so all classes have the same amount of training images. To do that, I found the class with maximum images, increase it 50%, and increase the other classes to the same number.

Here is an example of the different effects I will apply for augment the training set.

![alt text][imageEffects]

And here is an example of how the augmentation preprocessing will work for one random image (applied 20 times):

![alt text][imageRndEffects]

The new size of training set after preprocessing is 129645, and this is how the histogram of the training set looks like

![alt text][imageYTrainAug]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is based on the LeNet architecture seeing in the previous lessons, in which I added a few modifications due to trial-error during the Training phase, like dropout to avoid overfitting.
The model consists of the following layers:

| Layer         	|     Description	        		| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 32x32x1 Grayscale Normalized image		| 
| Layer1 Convolution 2d	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU			|						|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  		|
| Layer2 Convolution 2d	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU			|						|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   		|
| Flatten		| outputs 400					|
| Layer3 FullyConnected | outputs 120					|
| MatMul y RELU		|						|
| Dropout 		|						|
| Layer4 FullyConnected | outputs 84					|
| MatMul y RELU		|						|
| Dropout 		|						|
| Layer5 FullyConnected | outputs 43					|
| MatMul		|						|
|:---------------------:|:---------------------------------------------:| 
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following parameters: 
1. Epochs = 10
2. Learning rate = 0.005
3. Batch Size = 512
4. Mu = 0, Sigma = 0.1
5. Dropout = 0.85

I came up with these values after experimenting during the training phase. My initial values where the sames as the proposed LeNet architecture.

All training sesion´s results are in:
![alt text][tests]

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Validation set accuracy of 0.9669
* Test set accuracy of 0.9486

As said, I chose LeNet architecture. At first I did several training sessions tuning different parameters, adding drop out, with no huge improvement (accuracy was always around 92%). After that, I read that generating new training data will help my accuracy and so it did. I started using more and more effects to generate new data, with some effects also affecting on the other way.
As conclusion I can say that improving my training data had more impact than tuning the parameters. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are some German traffic signs that I found on the web:

![alt text][imageWeb]

The images are bigger than 32X32 but similar appearance to the ones we used before. We need to do some preprocessing before using them, like remove alpha channel, resize to 32x32 and then, as before, transform to grayscale and normalize.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

* Image | Correct Label | Prediction

* Speed limit (50km/h) | 2 | 5
* Speed limit (70km/h) | 4 | 5
* End of speed limit (80km/h) | 6 | 3
* No passing | 9 | 9
* No passing for vehicles over 3.5 metric tons | 10 | 19

The model was able to correctly guess 1 out of the 5 traffic signs, which gives a very poor accuracy of 20%. This is a clear sign of Overfitting. Our model is behaving very good with training data but not with new data.
 
#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

First image: "Speed limit (50km/h)":
* Correct label: 2.
* Probability: [ 0.47391632,  0.33841556,  0.04837811,  0.03809149,  0.01927107].
* Prediction: [ 5,  3, 19,  9, 12].

Second image: "Speed limit (70km/h)":
* Correct label: 4.
* Probability: [ 0.50140744,  0.24218594,  0.07069884,  0.05612415,  0.030919  ].
* Prediction: [ 5,  3,  9, 19,  4].

Third image: "End of speed limit (80km/h)":
* Correct label: 6.
* Probability: [ 0.41057149,  0.33094698,  0.09637477,  0.05295381,  0.0255525 ].
* Prediction: [ 3,  5, 19,  9, 31].

Fourth image: "No passing":
* Correct label: 9.
* Probability: [ 0.33280474,  0.17103197,  0.13837901,  0.12856421,  0.06448509].
* Prediction: [ 9,  5,  3, 19, 12].

Five image: "No passing for vehicles over 3.5 metric tons":
* Correct label: 10.
* Probability: [ 0.28323454,  0.21641321,  0.18375671,  0.15051356,  0.03256079].
* Prediction: [19,  3,  5,  9, 12].

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


