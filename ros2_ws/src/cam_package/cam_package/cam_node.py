import rclpy
import cv2
from rclpy.node import Node
# from std_msgs.msg import Image, String
from std_msgs.msg import String

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraTalker(Node):

    def __init__(self):
        super().__init__('talker')
        # self.publisher = self.create_publisher(
        #     Image,
        #     'camera_stream',
        #     10)

        # self.publisher = self.create_publisher(
        #             String,
        #             'camera_stream',
        #             10)
        self.publisher = self.create_publisher(
            Image,
            'camera_stream',
            10
        )

        self.bridge = CvBridge()
        time_period = 1.0
        self.timer = self.create_timer(time_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        # msg = String()
        # msg.data = "Camera image: " + str(self.i)
        # self.publisher.publish(msg)
        # self.get_logger().info('Publishing ' + str(msg.data))

        image = cv2.imread('src/cam_package/cam_package/img_test.png')
        print(type(image))
        msg = self.bridge.cv2_to_imgmsg(
            image,
            encoding='bgr8'
        )

        self.publisher.publish(msg)

        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    node = CameraTalker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
