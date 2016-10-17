#!/usr/bin/python
"""
view_rosbag_video.py: version 0.1.0
Note:
Part of this code was copied and modified from github.com/comma.ai/research (code: BSD License)

Todo:
Update steering angle projection.  Current version is a hack from comma.ai's version
Update enable left, center and right camera selection.  Currently all three cameras are displayed.
Update to enable display of trained steering data (green) as compared to actual (blue projection).

History:
2016/10/06: Update to add --skip option to skip the first X seconds of data from rosbag.
2016/10/02: Initial version to display left, center, right cameras and steering angle.
"""

import argparse
import sys
import numpy as np
import rosbag
import datetime
import scipy
import scipy.misc
import cv2
import os
from cv_bridge import CvBridge, CvBridgeError



size = (320*3, 240)
pimg=np.zeros(shape=(320,240,3))




# ***** main loop *****
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Udacity SDC Challenge-2 Video viewer')
  parser.add_argument('--dataset', type=str, default="dataset.bag", help='Dataset/ROS Bag name')
  parser.add_argument('--skip', type=int, default="0", help='skip seconds')
  parser.add_argument('--outdir', type=str, default="/home/aiwagan/challenge2/dataset/", help='Provide a specific folder for storing images.')
  args = parser.parse_args()

  dataset = args.dataset
  skip = args.skip
  outdir=args.outdir
  print(outdir)
  fmt='jpg'
  startsec = 0
  bridge = CvBridge()
  # Load the model
  #  model= loadsavedmodel(modelfile)
  angle_steers=0
  d=1

  with open("data.txt", "w") as data_file:
    print "reading rosbag ", dataset
    bag = rosbag.Bag(dataset, 'r')
    for topic, msg, t in bag.read_messages(topics=['/center_camera/image_color','/right_camera/image_color','/left_camera/image_color','/vehicle/steering_report']):
      if startsec == 0:
          startsec = t.to_sec()
          if skip < 24*60*60:
              skipping = t.to_sec() + skip
              print "skipping ", skip, " seconds from ", startsec, " to ", skipping, " ..."
          else:
              skipping = skip
              print "skipping to ", skip, " from ", startsec, " ..."
      else:
          if t.to_sec() > skipping:
              #if topic in ['/center_camera/image_color','/right_camera/image_color','/left_camera/image_color']:
                  #print(topic, msg.header.seq, t-msg.header.stamp, msg.height, msg.width, msg.encoding, t)
              #else:
                  #print(topic, msg.header.seq, t-msg.header.stamp, msg.steering_wheel_angle, t)
              if topic in ['/vehicle/steering_report']:
                  angle_steers = msg.steering_wheel_angle

              try:
                  if topic in ['/center_camera/image_color','/right_camera/image_color','/left_camera/image_color']:
                      # RGB_str = msg.data
                      cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                      cv_image = cv2.resize(cv_image, (160,120), interpolation=cv2.INTER_CUBIC)
                      image_filename = os.path.join(outdir, str(d) + '.' + fmt)
                      cv2.imwrite(image_filename, cv_image)
                      data_file.write("{0}.jpg {1}\n".format(d,angle_steers))
                      d=d+1
                      if ((d%1000)==0):
                          print ("Wrote {0} Images.\n".format(d))

              except Exception, e:
                 print("Error saving image.", e)




