# **Traffic Sign Recognition** 



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

[image1]: ./examples/visualization.PNG "Visualization"
[image2]: ./examples/sings.PNG "Signs"
[image3]: ./examples/image1.png "Traffic Sign 1"
[image4]: ./examples/image2.png "Traffic Sign 2"
[image5]: ./examples/image3.png "Traffic Sign 3"
[image6]: ./examples/image4.png "Traffic Sign 4"
[image7]: ./examples/image5.png "Traffic Sign 5"
[image8]: ./examples/image6.png "Traffic Sign 6"
[image9]: ./examples/image7.png "Traffic Sign 7"
[image10]: ./examples/image8.png "Traffic Sign 8"
[image11]: ./examples/softmax.png "softmax"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---


You're reading it! and here is a link to my [project code](https://github.com/bariscansecim/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration


I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribute

![alt text][image2]

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Apply pre-processing refers to techniques such as converting to grayscale, normalization.

As a first step, I decided to convert the images to grayscale because of reduce the number of image channel


Then, I normalized the image data because that the data has mean zero and equal variance.


#### 2. My final model architecture looks like including model type, layers, layer sizes, connectivity, etc.)

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Convolution 5x5	    | 1x1 stride, same padding, outputs 1x1x400     |
| RELU					|												|
| Flatten				| input 5x5x16, output 400		    			|
| Flatten				| input 1x1x400, output 400		    			|
| Concat				| input 400+400, output 800		    			|
| Dropout				|                           	    			|
| Fully Connected		| input 800, output 43                			|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* Epoch: 			20
* Batch size: 		128
* Learning rate:	0.0008

I used softmax_cross_entropy_with_logits to get a tensor representing the mean loss value to which applied tf.reduce_mean to compute the mean of elements across dimensions of the result. Finally I applied minimize to the AdamOptimizer for optimizer.


My final model results were:
* validation set accuracy of 0.944
* test set accuracy of 0.933

 

### Test a Model on New Images

#### 1. Choose eight German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9] ![alt text][image10] 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| The next intersection | The next intersection   						| 
| 30 km/h     			| 30 km/h										|
| No passing over 3.5	| No passing over 3.5							|
| Keep right	      	| Keep right					 				|
| Turn left ahead		| Turn left ahead     							|
| General caution		| General caution     							|
| Road work			    | Road work     							    |
| 60 km/h			    | 60 km/h    							        |


The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the eight new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

![alt text][image11] 


