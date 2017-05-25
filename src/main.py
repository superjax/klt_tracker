import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from pyquaternion import Quaternion
from nav_msgs.msg import Odometry

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

NUM_FEATURES = 15


class rovio:
    def __init__(self):
        self.image_sub = rospy.Subscriber("/camera/image_mono", Image, self.image_callback)
        self.imu_sub = rospy.Subscriber("imu/data", Imu, self.imu_callback)

        self.odom_pub = rospy.Publisher("estimate", Odometry, queue_size=1)
        self.bridge = CvBridge()

        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.prev_image = []
        self.initialized = False
        self.prev_kp = []
        self.prev_des = []

        # p, q, v, alpha, beta, c, z.  mu/rho will be added
        self.x = np.zeros([22 + NUM_FEATURES*3, 1])
        self.x[6] = 1.0
        self.prev_imu_time = 0

        self.g = np.array([0, 0, 9.80665])

    def dynamics(self, x, w, a):
        xdot = np.zeros(np.shape(x))
        xdot[:3] = -skew(w).dot(x[:3]) + x[3:6]
        q = Quaternion(x[6:10])
        xdot[3:6] = -skew(w).dot(x[3:6]) + a + np.array([q.inverse.rotate(self.g)]).T
        xdot[6:10] = np.array([(-1/2.0 * q * Quaternion(scalar=0, vector=w)).elements]).T
        return xdot

    def imu_callback(self, msg):
        if self.prev_imu_time == 0:
            self.prev_imu_time = msg.header.stamp
            return

        dt = (msg.header.stamp - self.prev_imu_time).to_sec()
        self.prev_imu_time = msg.header.stamp

        w = np.array([[msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]]).T
        a = np.array([[msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]]).T

        self.x = self.x + self.dynamics(self.x, w, a)*dt
        self.x[6:10] /= np.linalg.norm(self.x[6:10])


        odom_msg = Odometry()
        odom_msg.header.stamp = msg.header.stamp
        odom_msg.pose.pose.position.x = self.x[0]
        odom_msg.pose.pose.position.y = self.x[1]
        odom_msg.pose.pose.position.z = self.x[2]

        odom_msg.twist.twist.linear.x = self.x[3]
        odom_msg.twist.twist.linear.y = self.x[4]
        odom_msg.twist.twist.linear.z = self.x[5]

        odom_msg.pose.pose.orientation.w = self.x[6]
        odom_msg.pose.pose.orientation.x = self.x[7]
        odom_msg.pose.pose.orientation.y = self.x[8]
        odom_msg.pose.pose.orientation.z = self.x[9]

        odom_msg.twist.twist.angular.x = msg.angular_velocity.x
        odom_msg.twist.twist.angular.y = msg.angular_velocity.y
        odom_msg.twist.twist.angular.z = msg.angular_velocity.z

        self.odom_pub.publish(odom_msg)



    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "mono8")
        except CvBridgeError as e:
            print(e)

        orb = cv2.ORB_create()

        kp = orb.detect(img, None)
        kp, des = orb.compute(img, kp)

        img2 = np.zeros((480, 640, 3), np.uint8)
        cv2.drawKeypoints(img, kp, img2, color=(0, 255, 0), flags=0)

        if not self.initialized:
            self.initialized = True
            self.prev_image = img
            self.prev_kp = kp
            self.prev_des = des
            return

        matches = self.bf_matcher.match(self.prev_des, des)

        matches = sorted(matches, key = lambda x: x.distance)

        # img3 = np.zeros((480, 640, 3), np.uint8)
        # img3 = cv2.drawMatches(self.prev_image, self.prev_kp, img, kp, matches[:10], None)

        # cv2.imshow("Image window", img2)
        # cv2.waitKey(1)

        self.prev_image = np.copy(img)
        self.prev_kp = kp
        self.prev_des = des

if __name__ == "__main__":
    rospy.init_node("rovio")

    thing = rovio()

    rospy.spin()