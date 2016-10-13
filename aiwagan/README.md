version 0.1 by Asim Imdad Wagan

# Udacity Challenge-2 Basic Implementation
A TensorFlow implementation of the Nvidia Self Driving based on the code developed by the sully chen with some changes.

#How to Use
Download the dataset (Small or Large) and extract the dataset.bag file into the repository folder

1- Use 'python dataprep.py --dataset ~/data/dataset.bag' to extract the images and steering angle from ROS bag
files into dataset folder and prepare the data.txt file.

2- Use `python train.py` to train the model from the data stored in the dataset folder which contains training images
and a data.txt file which contains the image and corresponding steering angle.


3- To simuate the driving with the challenge dataset vidviewer can be used.
Use 'python vidviewer.py --dataset ~/data/dataset.bag' where it is auumed the there is a save folder in your current
directory containing the driving model files. V

A sample model file is included for testing.
