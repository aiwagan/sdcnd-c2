import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def get_horizon_y(img, draw=False):
    ''' Estimate horizon y coordinate using Canny edge detector and Hough transform. '''

    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    if draw:
        fig = plt.figure()
        plt.imshow(gray, cmap='gray')

    edges = cv2.Canny(gray,20,150,apertureSize = 3)

    if draw:
        fig = plt.figure()
        plt.imshow(edges, cmap='gray')

    lines = cv2.HoughLines(edges,1,np.pi/180, 50)
    horizontal_lines = []

    horizon = None
    # empirical min horizontal y
    min_rho = 300

    for i, line in enumerate(lines):
        for rho,theta in line:

            # just the horizontal lines
            if np.sin(theta) > 0.9999:

                if rho < min_rho and rho >= img.shape[0]/2 and rho < 300:
                    min_rho = rho
                    horizon = line

    if draw and horizon.any():

        for rho,theta in horizon:
            a = np.cos(theta)
            b = np.sin(theta)

            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(gray,(x1,y1),(x2,y2),(255,255,255),2)

        fig = plt.figure()
        plt.imshow(gray, cmap='gray')

    return int(min_rho)

def eulerToRotation(theta):
    ''' Calculates Rotation Matrix given euler angles. '''

    R_x = np.array([
                    [1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])


    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R


def translation(t):
    ''' Returns a 2-dimension translation matrix '''

    T = np.array([[1, 0, t[0]],
                  [0, 1, t[1]],
                  [0, 0, 1]])
    return T


def apply_distortion(img, rotation=None, shift=None, rotation_mean=0, rotation_std=0.03,
                     shift_mean=0, shift_std=50, crop_x=50, crop_y=300, draw=False, crop=True):
    '''
    Applies shift and rotation distortion to image, assuming all points below the
    horizon are on flat ground and all points above the horizon are infinitely far away.
    The distorted image is also cropped to match the proportions used in "End to End Learning for Self-Driving Cars".

    Parameters:
    img - source image
    rotation - 'yaw' rotation angle in radians. If None, value is sampled from normal distribution.
    shift - shift in pixels (TODO: conversion meters -> pixels). If None, value is sampled from normal distribution.
    rotation_mean - rotation distribution mean
    rotation_std - rotation distribution standard deviation
    shift_mean - shift distribution mean
    shift_std - shift distribution standard deviation
    crop_x - number of pixels to be cropped from each side of the distorted image.
    crop_y - number of pixels to be cropped from the upper portion of the distorted image.
    crop - enables/disables cropping
    draw - enables/disables drawing using matplotlib (useful for debugging)

    '''

    if rotation is None:
        rotation = np.random.normal(rotation_mean, rotation_std)

    if shift is None:
        shift = np.random.normal(shift_mean, shift_std)

    copy = img.copy()
    horizon_y = get_horizon_y(img)

    if draw:
        fig = plt.figure(figsize=(12, 12))
        fig.add_subplot(3, 2, 1)
        plt.imshow(copy)

    pts = np.array([[0, horizon_y], [img.shape[1]-1, horizon_y], [img.shape[1]-1, img.shape[0]-1],
                    [0, img.shape[0]-1]], dtype=np.float32)

    birds_eye_magic_number = 20

    dst = np.array([
            [0, 0],
            [img.shape[1] - 1, 0],
            [img.shape[1] - 1, (img.shape[0] * birds_eye_magic_number) - 1],
            [0, (img.shape[0] * birds_eye_magic_number) - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(pts, dst)
    below_horizon = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0] * birds_eye_magic_number))

    if draw:
        fig.add_subplot(3, 2, 2)
        plt.imshow(below_horizon)

    T1 = translation([-(below_horizon.shape[1]/2 + shift), -below_horizon.shape[0]])
    T2 = translation([below_horizon.shape[1]/2, below_horizon.shape[0]])
    T = np.dot(T2, np.dot(eulerToRotation([0., 0., rotation]), T1))
    warped = cv2.warpPerspective(below_horizon, T, (below_horizon.shape[1], below_horizon.shape[0]))

    if draw:
        fig.add_subplot(3, 2, 3)
        plt.imshow(warped)

    warped = cv2.warpPerspective(warped, np.linalg.inv(M), (img.shape[1], img.shape[0]))

    if draw:
        fig.add_subplot(3, 2, 4)
        plt.imshow(warped)

    copy[horizon_y:] = warped[horizon_y:] * (warped[horizon_y:] > 0) + copy[horizon_y:] * (1 - (warped[horizon_y:] > 0))
    copy[horizon_y - 3: horizon_y + 3] = cv2.blur(copy[horizon_y - 3: horizon_y + 3], (1,5))

    if crop:
        copy = copy[crop_y:, crop_x:img.shape[1]-crop_x]

    if draw:
        fig.add_subplot(3, 2, 5)
        plt.imshow(copy)

    return copy


def get_steer_back_angle(steering_wheel_angle, speed, shift, rotation, steer_back_time = 2., fps = 20,
                   wheel_base = 2.84988, steering_ratio = 14.8):

    '''
    Calculate the "steer back" angle, that is, the steering angle that to steer the vehicle back to the desired
    location and orientation in "steer_back_time" seconds. Useful to calculate the steering labels for distorted images.
    '''

    dt = (1./fps)
    shift0 = shift
    rotation0 = rotation
    theta = math.pi/2. + rotation
    # true vehicle velocity
    v = speed
    vx = math.cos(theta) * v

    # assume constant acceleration
    ax = (-shift - vx * steer_back_time) * 2. / (steer_back_time * steer_back_time)

    # calculate velocity x and shift after dt
    vx = vx + ax * dt
    shift = shift + vx * dt + ax * dt * dt / 2.

    # steer back angular velocity
    vtheta = (math.acos(vx / v) - theta) / dt

    # calculate theta after dt
    #theta = theta + vtheta * dt
    theta = math.acos(vx / v)

    # true angular velocity
    vtheta_truth = math.tan(steering_wheel_angle / steering_ratio) * v / wheel_base

    #print(vtheta, vtheta_truth, left_steering.iloc[i].steering_wheel_angle)

    # we have two add "steer back" and true angular velocities to calculate final steering angle
    vtheta = vtheta + vtheta_truth

    wheel_angle = math.atan(vtheta * wheel_base / v)
    steer_back_angle = wheel_angle * steering_ratio

    rotation = -(math.pi/2. - theta)
    return (shift, rotation, steer_back_angle)
