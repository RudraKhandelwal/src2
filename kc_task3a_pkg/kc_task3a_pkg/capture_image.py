#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
from datetime import datetime

class ImageCaptureNode(Node):
    def __init__(self):
        super().__init__('image_capture_node')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)
        self.cv_image = None
        self.get_logger().info("Image Capture Node Started. Press 's' to save image, 'q' to quit.")

    def image_callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv2.imshow("Camera Feed", self.cv_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.save_image()
            elif key == ord('q'):
                rclpy.shutdown()
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")

    def save_image(self):
        if self.cv_image is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"realsense_capture_{timestamp}.png"
            cv2.imwrite(filename, self.cv_image)
            self.get_logger().info(f"Image saved to {os.path.abspath(filename)}")

def main(args=None):
    rclpy.init(args=args)
    node = ImageCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
    finally:
        cv2.destroyAllWindows()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
