# Udacity Challenge 2: Using Deep Learning to Predict Steering Angles 
### by Marcio Fonseca

TensorFlow implementation based on [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316).

For more details see [Challenge 2](https://medium.com/udacity/challenge-2-using-deep-learning-to-predict-steering-angles-f42004a36ff3#.32gnncto4).

## Extracting data

Driving datasets are available in [rosbag](http://wiki.ros.org/rosbag) files. To extract data from rosbag, use the provided python scripts. Note: if you are using MacOS, I recommend you to install Ubuntu and ROS in a virtual machine.

### Extracting camera images

```
./bag_to_images.py dataset.bag right_camera/ /right_camera/image_color
./bag_to_images.py dataset.bag left_camera/ /left_camera/image_color
./bag_to_images.py dataset.bag center_camera/ /center_camera/image_color
```

### Extracting camera timestamps

```
./camera_timestamps.py dataset.bag timestamps-center.csv /center_camera/image_color
./camera_timestamps.py dataset.bag timestamps-left.csv /left_camera/image_color
./camera_timestamps.py dataset.bag timestamps-right.csv /right_camera/image_color
```

### Extracting steering angles

This script can extract any topic data to csv.

```
./bag_to_csv.py dataset.bag steering.csv /vehicle/steering_report
```

## Data preprocessing

Image resizing, pickling and steering interpolation is implemented in [thefonseca-input.ipynb](https://github.com/wfs/ai-world-car-team-c2/blob/master/thefonseca/thefonseca-input.ipynb)

## Data augmentation

Coming soon...

## Model definition and training

See [thefonseca-nvidia.ipynb](https://github.com/wfs/ai-world-car-team-c2/blob/master/thefonseca/thefonseca-nvidia.ipynb)
