#!/usr/bin/python
"""
view_rosbag_video.py: version 0.1.0
Note:
Part of this code was copied and modified from
github.com/comma.ai/research (code: BSD License)

Todo:
Update steering angle projection.  Current version
is a hack from comma.ai's version Update enable left,
center and right camera selection.  Currently all three
cameras are displayed.
Update to enable display of trained steering data (green)
as compared to actual (blue projection).

History:
2016/10/06: Update to add --skip option to skip the first X
seconds of data from rosbag.

2016/10/02: Initial version to display left, center, right
cameras and steering angle.
"""

import argparse
import numpy as np
import pygame
import tensorflow as tf
import scipy
import scipy.misc
import cv2
import skimage.transform as sktf


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

x = tf.placeholder(tf.float32, shape=[None, 120, 160, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

x_image = x

#first convolutional layer
W_conv1 = weight_variable([5, 5, 3, 24])
b_conv1 = bias_variable([24])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 2) + b_conv1)
#print h_conv1.get_shape()
#second convolutional layer
W_conv2 = weight_variable([5, 5, 24, 36])
b_conv2 = bias_variable([36])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)
#print h_conv2.get_shape()
#third convolutional layer
W_conv3 = weight_variable([5, 5, 36, 48])
b_conv3 = bias_variable([48])

h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 2) + b_conv3)
#print h_conv3.get_shape()
#fourth convolutional layer
W_conv4 = weight_variable([5, 5, 48, 64])
b_conv4 = bias_variable([64])

h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, 1) + b_conv4)
#print h_conv4.get_shape()
#fifth convolutional layer
W_conv5 = weight_variable([8, 8, 64, 64])
b_conv5 = bias_variable([64])

h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, 1) + b_conv5)
#print h_conv5.get_shape()

#FCL 1
W_fc1 = weight_variable([384, 1164])
b_fc1 = bias_variable([1164])

h_conv5_flat = tf.reshape(h_conv5, [-1, 384])
#print h_conv5_flat.get_shape()
h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)
#print h_fc1.get_shape()
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#print h_fc1_drop.get_shape()
#FCL 2
W_fc2 = weight_variable([1164, 100])
b_fc2 = bias_variable([100])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#print h_fc2.get_shape()
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
#print h_fc1_drop.get_shape()
#FCL 3
W_fc3 = weight_variable([100, 50])
b_fc3 = bias_variable([50])

h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

#FCL 3
W_fc4 = weight_variable([50, 10])
b_fc4 = bias_variable([10])

h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)

h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

#Output
W_fc5 = weight_variable([10, 1])
b_fc5 = bias_variable([1])

y = tf.mul(tf.atan(tf.matmul(h_fc4_drop, W_fc5) + b_fc5), 2) #scale the atan output


sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "./save/model.ckpt")

# from keras.models import model_from_json

pygame.init()
size = (320 * 3, 240)
pygame.display.set_caption("Udacity SDC challenge 2: camera video viewer")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)

imgleft = pygame.surface.Surface((320, 240), 0, 24).convert()
imgcenter = pygame.surface.Surface((320, 240), 0, 24).convert()
imgright = pygame.surface.Surface((320, 240), 0, 24).convert()
pimg = np.zeros(shape=(320, 240, 3))

# ***** get perspective transform for images *****

rsrc = \
 [[43.45456230828867, 118.00743250075844],
  [104.5055617352614, 69.46865203761757],
  [114.86050156739812, 60.83953551083698],
  [129.74572757609468, 50.48459567870026],
  [132.98164627363735, 46.38576532847949],
  [301.0336906326895, 98.16046448916306],
  [238.25686790036065, 62.56535881619311],
  [227.2547443287154, 56.30924933427718],
  [209.13359962247614, 46.817221154818526],
  [203.9561297064078, 43.5813024572758]]
rdst = \
[[10.822125594094452, 1.42189132706374],
  [21.177065426231174, 1.5297552836484982],
  [25.275895776451954, 1.42189132706374],
  [36.062291434927694, 1.6376192402332563],
  [40.376849698318004, 1.42189132706374],
  [11.900765159942026, -2.1376192402332563],
  [22.25570499207874, -2.1376192402332563],
  [26.785991168638553, -2.029755283648498],
  [37.033067044190524, -2.029755283648498],
  [41.67121717733509, -2.029755283648498]]

tform3_img = sktf.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))

def perspective_tform(x, y):
  p1, p2 = tform3_img((x,y))[0]
  return p2, p1

# ***** functions to draw lines *****
def draw_pt(img, x, y, color, shift_from_mid, sz=1):
  col, row = perspective_tform(x, y)
  row = int(row) + shift_from_mid
  col = int(col+img.get_height()*2)/3
  if row >= 0 and row < img.get_width()-sz and\
     col >= 0 and col < img.get_height()-sz:
    img.set_at((row-sz,col-sz), color)
    img.set_at((row-sz,col), color)
    img.set_at((row-sz,col+sz), color)
    img.set_at((row,col-sz), color)
    img.set_at((row,col), color)
    img.set_at((row,col+sz), color)
    img.set_at((row+sz,col-sz), color)
    img.set_at((row+sz,col), color)
    img.set_at((row+sz,col+sz), color)

def draw_path(img, path_x, path_y, color, shift_from_mid):
  for x, y in zip(path_x, path_y):
    draw_pt(img, x, y, color, shift_from_mid)

# ***** functions to draw predicted path *****

def calc_curvature(v_ego, angle_steers, angle_offset=0):
  deg_to_rad = np.pi/180.
  slip_fator = 0.0014 # slip factor obtained from real data
  steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
  wheel_base = 2.67   # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

  angle_steers_rad = (angle_steers - angle_offset) * deg_to_rad
  curvature = angle_steers_rad/(steer_ratio * wheel_base * (1. + slip_fator * v_ego**2))
  return curvature

def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
  #*** this function returns the lateral offset given the steering angle, speed and the lookahead distance
  curvature = calc_curvature(v_ego, angle_steers, angle_offset)

  # clip is to avoid arcsin NaNs due to too sharp turns
  y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999))/2.)
  return y_actual, curvature

def draw_path_on(img, speed_ms, angle_steers, angle_predicted, color1=(0,0,255), color2=(0,255,0), shift_from_mid=0):
  path_x = np.arange(0., 50.1, 0.5)
  path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)

  ppath_x = np.arange(0., 50.1, 0.5)
  ppath_y, _ = calc_lookahead_offset(speed_ms, angle_predicted, ppath_x)

  draw_path(img, path_x, path_y, color1, shift_from_mid)
  draw_path(img, ppath_x, ppath_y, color2, shift_from_mid)


#def loadsavedmodel(modelfile):
#  saver = tf.train.Saver()
#  saver.restore(sess,modelfile )
#  return saver



imfiles=[]
steers=[]




with open("challenge2.csv") as f:
  for line in f:
    imfiles.append("/home/aiwagan/output/" + line.split()[0])
    steers.append(float(line.split()[1]) * scipy.pi / 180)


#shuffle list of images
imgsteers = list(zip(imfiles, steers))

for f in imgsteers:
  imgcenter = pygame.transform.scale(pygame.image.load( f[0] ), (320, 240))

  img_for_pred = scipy.misc.imresize ( scipy.misc.imread(f[0]), [120, 160] ) / 255.0
  #print (img_for_pred.shape)
  pred_steer = y.eval(feed_dict={x: [img_for_pred], keep_prob: 1.0})[0][0]      #* 180.0 / scipy.pi

  draw_path_on(imgcenter, 0, f[1]*100,pred_steer*100, (0,0,255), (0,255,0),0)

  screen.blit(imgcenter, (0,0))

  pygame.display.flip()


sess.close()


