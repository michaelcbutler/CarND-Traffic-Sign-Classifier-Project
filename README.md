# **Traffic Sign Recognition Project** 

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Refer to the [project code](https://github.com/michaelcbutler/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) for implementation details.


[//]: # (Image References)

[image1]: ./plots/histogram.png "Label Frequency by Data Set"
[image2]: ./plots/signs.png "Input Image Examples by Label"
[image3]: ./plots/grayscale.png "Grayscale Conversion and Normalization"
[image4]: ./test-images/30kph.png "Speed limit (30km/h)"
[image5]: ./test-images/50kph.png "Speed limit (50km/h)"
[image6]: ./test-images/70kph.png "Speed limit (70km/h)"
[image7]: ./test-images/100kph.png "Speed limit (100km/h)"
[image8]: ./test-images/nopassing.png "No passing"
[image9]: ./plots/baseline.png "Baseline: RGB input"
[image10]: ./plots/gray1.png "Grayscaled input"
[image11]: ./plots/gray2.png "Grayscaled/normalized input"
[image12]: ./plots/gray2.drop.png "Added dropout layer"
[image13]: ./plots/final.png "Final result"


## Data Set Summary & Visual Exploration

I used the python to calculate summary statistics of the traffic signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is 32, 32, 3)
* The number of unique classes/labels in the data set is 43

Below is an exploratory visualization of the data set distribution. This histogram compares the frequency of each label in the training, validation, and test data sets. The distribution appears relatively consistent between data sets.

![alt text][image1]

I also plotted an example image of each class/label from the orignal RGB data set. The first few images are shown below:

![alt text][image2]


## Design and Test a Model Architecture

### Input Image Preprocessing

As a first step, I converted the images to grayscale to reduce the complexity of the input data set. The paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks" by Sermanet and LeCunn also suggested improved accuracy from grayscale conversion. Intutively, the example images showed diverse lighting - some images were quite dark - which would affect coloring. Normalization gray values to a range of [-1,1] also helps to factor out lighting variations. 

Here is an example of a traffic sign image before and after grayscale conversion/normalization:

![alt text][image3]

### Model Architecture
**Layer 1:**
- Convolution. The output shape is 28x28x6.
- Activation. ReLu activation.
- Pooling. The output shape is 14x14x6.

**Layer 2:**
- Convolution. The output shape is 10x10x16.
- Activation. ReLu activation.
- Pooling. The output shape is 5x5x16.
- Flatten. 

**Layer 3:**
- Fully Connected. This has 120 outputs.
- Activation. ReLu activation.
- Drop out.

**Layer 4:**
- Fully Connected. This has 84 outputs.
- Activation. ReLu activation.

**Layer 5:**
- Fully Connected (Logits). This has 10 outputs.
 


### Model Training

The initial model used the Lenet classifier unmodified with RGB image input. Here is an exploratory plot showing accuracy and loss versus EPOCH for this configuration:

![alt test][image9]

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.931
* test set accuracy of 0.914

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 


