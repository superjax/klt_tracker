#!/usr/bin/python

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from pyquaternion import Quaternion
from nav_msgs.msg import Odometry
import random

class KLT_tracker:
    def __init__(self, num_features):
        self.image_sub = rospy.Subscriber("camera/image_raw", Image, self.image_callback)
        self.bridge = CvBridge()

        self.prev_image = []
        self.initialized = False

        self.plot_matches = True

        self.num_features = num_features

        empty_feature_array = np.zeros((self.num_features, 2, 1))
        self.features = [empty_feature_array, np.zeros(self.num_features)]

        self.feature_params = dict(qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)

        self.lk_params = dict(winSize  = (15,15),
                              maxLevel = 2,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.color = np.random.randint(0, 255, (self.num_features, 3))

        self.feature_nearby_radius = 25
        self.next_feature_id = 0

    def image_callback(self, msg):
        # Load Image
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "mono8")
        except CvBridgeError as e:
            print(e)
            exit(1)

        # If first time initialize the feature
        if not self.initialized:
            # Capture a bunch of new features
            self.features[0] = cv2.goodFeaturesToTrack(img, mask=None, maxCorners=self.num_features, **self.feature_params)

            # Set the first set of indexes
            self.features[1] = np.array([i+1 for i in range(self.num_features)])
            self.next_feature_id = self.num_features+1



            # TODO: handle the case that we don't get enough features

            # Save off the image
            self.prev_image = img

            self.initialized = True
            return

        # Otherwise, we've already initialized
        else:

            # calculate optical flow
            new_features, st, err = cv2.calcOpticalFlowPyrLK(self.prev_image, img, self.features[0], None, **self.lk_params)

            # Select good points
            matched_features = new_features[st == 1]


            # good_old = [self.features[0][st == 1], self.features[1][st==1]]

            # Now update the previous frame and previous points
            self.prev_image = img.copy()
            self.features[0] = matched_features.reshape(-1, 1, 2)


            # If we lost a features, drop the index
            if 0 in st:
                self.features[1] = self.features[1][(st == 1).ravel()]

            # If we are missing points, collect new ones
            if len(matched_features) < self.num_features:
                # First, create a mask around the current points
                current_point_mask = np.ones_like(img)
                for point in matched_features:
                    a, b = point.ravel()
                    cv2.circle(current_point_mask, (a, b), self.feature_nearby_radius, 0, thickness=-1, lineType=0)

                num_new_features = self.num_features - len(matched_features)
                new_features = cv2.goodFeaturesToTrack(img, mask=current_point_mask, maxCorners=num_new_features, **self.feature_params)
                self.features[0] = np.concatenate((self.features[0], new_features), axis=0)
                self.features[1] = np.concatenate((self.features[1], np.array([i + self.next_feature_id for i in range(num_new_features)])), axis=0)
                self.next_feature_id = num_new_features + self.next_feature_id

            # If we are debugging, plot the points
            if self.plot_matches:
                # Convert the image to color
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # draw the features and ids
                for id, point in zip(self.features[1], self.features[0]):
                    x, y = point.ravel()
                    img = cv2.circle(img, (x, y), 5, self.color[id % self.num_features].tolist(), -1)
                    img = cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

                cv2.imshow("Image window", img)
                cv2.waitKey(1)

if __name__ == "__main__":
    rospy.init_node("rovio")

    thing = KLT_tracker(20)

    rospy.spin()
