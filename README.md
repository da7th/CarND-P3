# CarND-P3

youtube link to the videos:
https://youtu.be/rFSvEyR4iOU


#**Behavioral Cloning**
**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_2017_02_12_19_28_54_611.jpg "Model Visualization"
[image2]: ./examples/center_2017_02_12_19_28_13_724.jpg "Recovery Image"
[image3]: ./examples/center_2017_02_12_19_28_44_633.jpg "Recovery Image"
[image4]: ./examples/left_2017_02_12_19_28_14_506.jpg "Recovery Image"
[image5]: ./examples/center_2017_02_12_19_28_54_611.jpg "Normal Image"
[image6]: ./examples/center_2017_02_12_19_28_54_611_flip.jpg "Flipped Image"
[image7]: ./examples/recovery.gif "Recovery gif"
[image8]: ./examples/-0.1_steering.jpg "steering example Image"
[image9]: ./examples/-0.2_steering.jpg "steering example Image"
[image10]: ./examples/0_steering.jpg "steering example Image"
[image11]: ./examples/-0.3_steering.jpg "steering example Image"
[image12]: ./examples/-0.6_steering.jpg "steering example Image"
[image13]: ./examples/-0.7_steering.jpg "steering example Image"
[image14]: ./examples/-0.08_steering.jpg "steering example Image"
[image15]: ./examples/0.2_steering.jpg "steering example Image"
[image16]: ./examples/0.4_steering.jpg "steering example Image"
[image17]: ./examples/0.5_steering.jpg "steering example Image"
[image18]: ./examples/0.6_steering.jpg "steering example Image"
[image19]: ./examples/0.17_steering.jpg "steering example Image"
[image20]: ./examples/0.06_steering.jpg "steering example Image"
[image21]: ./examples/0.37_steering.jpg "steering example Image"
[image22]: ./examples/figure_1.png "intial plot Image"
[image23]: ./examples/figure_2.png "equalised plot Image"
[image24]: ./examples/image_center.jpg "center camera Image"
[image25]: ./examples/image_left.jpg "left camera Image"
[image26]: ./examples/image_right.jpg "right camera plot Image"



My project includes the following files:
* steering_model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* equalise_function.py for equalising the data ranges loaded for training and augment brightness

###Model Architecture and Training Strategy

The data is normalized in the model using a Keras lambda layer (lines: 155).

My model consists of a convolution neural network with 3x3 filter sizes and depths consistent to 32 (lines: 162, 168, 174)

The model includes ELU layers to introduce nonlinearity (lines: 164, 170, 176, 182, 186)



####Attempts to reduce over-fitting in the model

The model includes MaxPool layers to help with generalization (lines: 166, 172)

The model contains dropout layers in order to reduce over-fitting (lines: 180, 184).

The model was trained and validated on different data sets to ensure that the model was not over-fitting.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.



####Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (lines: 190).



####Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, going on the tracks in reverse, and flipping all the input data to remove any directional bias. There is a secondary script that returns an equal amount of images for every one of 20 angle ranges to ensure there is no bias towards turning strength as opposed to going straight as the 0-angle data for driving straight is much higher than the other ranges.



####Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image5]

Here is the view from the center, left and right cameras:

![alt text][image8]
![alt text][image8]
![alt text][image8]

And here are some examples with corresponding steering angles. All images are for the center camera:

-0.1
![alt text][image8]
-0.2
![alt text][image9]
0
![alt text][image10]
-0.3
![alt text][image11]
-0.6
![alt text][image12]
-0.7
![alt text][image13]
-0.08
![alt text][image14]
0.2
![alt text][image15]
0.4
![alt text][image16]
0.5
![alt text][image17]
0.6
![alt text][image18]
0.17
![alt text][image19]
0.06
![alt text][image20]
0.37
![alt text][image21]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover by itself in case it steers off course accidentally. These images show what a recovery looks like starting from the side of the road and turning sharply inwards as:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image7]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would remove and directional bias and provide easy data for training. The images were also ran through a random brightness change for both the normal and flipped cases to increase the amount of input data and decrease the effect of brightness on the learning. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

After the collection process, I had 9,840 out of an initial pool of 231,720 data points (mostly angle = 0). I then preprocessed this data by resizing the image to half the actual resolution as it was easier to load. Doing this meant that I had to modify the drive.py file to resize prediction images to the same resolution.

Angle ratios in the data initially
![alt text][image22]

Angle ratios after equalising
![alt text][image23]

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the fact the more epochs only improved the loss very slightly (rule of 30 from the lesson played a part in the decision.) I used an Adam optimizer so that manually training the learning rate wasn't necessary.
