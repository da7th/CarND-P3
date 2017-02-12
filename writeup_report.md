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

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* steering_model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The steering_model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The data is normalized in the model using a Keras lambda layer (code line 18).

My model consists of a convolution neural network with 3x3 filter sizes and depths consistent to 32 (lines: 128, 134, 140)

The model includes RELU layers to introduce nonlinearity (lines: 130, 136, 142, 148, 152)

####2. Attempts to reduce over-fitting in the model

The model includes MaxPool layers to help with generalization (lines: 132, 138)

The model contains dropout layers in order to reduce over-fitting (lines: 146, 150).

The model was trained and validated on different data sets to ensure that the model was not over-fitting (lines: 111, 159)

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (lines: 156).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, going on the tracks in reverse, and flipping all the input data to remove any directional bias

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to modify an existing model
I had for classifying traffic sign images. Since this is a regression problem however I
removed the final layers and ended the architecture with a single output neuron. Also I re-ordered some layers to get a similar flow to the Nvidia paper's architecture provided in the instructions Github repo.

My first step was to use a convolution neural network model similar to the Traffic Sign Classifier Project. I thought this model might be appropriate because it has already proven to do well at classifying images. taking inspiration from the Nvidia end-to-end model, some changes were made to the structure to match the flow of the Nvidia model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was over-fitting.

To combat the over-fitting, I modified the model so that it had more Relu layers, Dropout layers and Maxpool layers.

Then I tried a few different settings for the filter depths and sizes on each layer as well as the width of the flat layers and their amount till a sufficient accuracy was achieved.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especially near the lake areas. to improve the driving behavior in these cases, I added the suggested cropping layer to remove the effect of the surroundings on the steering prediction, then I added more data specific to the areas of trouble in the track as well as data of recovery incase it still veers off course.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (steering_model.py lines: 119-154) consisted of a convolution neural network with the following layers and layer sizes:

#normalisation to a mean of 0 by dividing with pixel value 255 and subtracting 0.5
#crop the input of this layer by crop_t from the top and crop_b from the bottom
#convolution layer with input shape 60,160,3
#Traffic Sign Classifier model adapted to generally flow like the Nvidia model
#normalisation>convolutions>flat>single output
#convolutional layer, filters=32@3x3 stride, border mode is set to valid
#relu activation
#maxpool layer with 2x2 filter
#convolutional layer, filters=32@3x3 stride
#relu activation
#maxpool layer with 2x2 filter
#convolutional layer, filters=32@3x3 stride
#relu activation
#flatten the input to 1D
#drop out layer with keep = 20%
#relu activation
#drop out layer with keep = 50%
#relu activation
#full connected layer to the single output node

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover by itself incase it steers off course accidentally. These images show what a recovery looks like starting from the side of the road and turning sharply inwards as:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would remove and directional bias and provide easy data for training. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 115,860 data points. I then preprocessed this data by resizing the image to half the actual resolution as it was easier to load. Doing this meant that I had to modify the drive.py file to resize prediction images to the same resolution.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the fact the more epochs only improved the loss very slightly (rule of 30 from the lesson played a part in the decision.) I used an adam optimizer so that manually training the learning rate wasn't necessary.
