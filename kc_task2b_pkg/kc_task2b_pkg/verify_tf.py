#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
import numpy as np

class TFVerifier(Node):
    def __init__(self):
        super().__init__('tf_verifier')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Test point in Optical Frame (Z-forward)
        # Let's pick a point 1m straight ahead
        self.point_opt = [0.0, 0.0, 1.0] 
        
        self.timer = self.create_timer(1.0, self.check_transform)
        self.get_logger().info("Waiting for TF...")

    def manual_transform(self, point):
        # point is [x, y, z] in optical frame
        point_x, point_y, point_z = point
        base_x = 0.74314 * point_z - 0.66862 * point_y - 1.08
        base_y = -point_x + 0.007
        base_z = -0.66862 * point_z - 0.74314 * point_y + 1.09 
        return [base_x, base_y, base_z]

    def check_transform(self):
        try:
            # 1. Manual Calculation
            manual_res = self.manual_transform(self.point_opt)
            
            # 2. TF Calculation
            # We assume the TF frame 'camera_link' (or 'camera_optical_frame') is GEOMETRIC (X-forward)
            # So we must rotate our Optical point to Geometric before asking TF.
            # Optical (x,y,z) -> Geometric (z, -x, -y)
            geom_x = self.point_opt[2]
            geom_y = -self.point_opt[0]
            geom_z = -self.point_opt[1]
            
            point_stamped = PointStamped()
            point_stamped.header.frame_id = "camera_link" # Using the geometric frame
            point_stamped.header.stamp = rclpy.time.Time().to_msg()
            point_stamped.point.x = float(geom_x)
            point_stamped.point.y = float(geom_y)
            point_stamped.point.z = float(geom_z)

            if self.tf_buffer.can_transform('base_link', 'camera_link', rclpy.time.Time()):
                tf_res = self.tf_buffer.transform(point_stamped, 'base_link')
                
                self.get_logger().info("\n=== COMPARISON ===")
                self.get_logger().info(f"Input Point (Optical): {self.point_opt}")
                self.get_logger().info(f"Manual Result: [{manual_res[0]:.4f}, {manual_res[1]:.4f}, {manual_res[2]:.4f}]")
                self.get_logger().info(f"TF Result:     [{tf_res.point.x:.4f}, {tf_res.point.y:.4f}, {tf_res.point.z:.4f}]")
                
                diff = np.linalg.norm(np.array(manual_res) - np.array([tf_res.point.x, tf_res.point.y, tf_res.point.z]))
                self.get_logger().info(f"Difference: {diff:.4f}")
                
                if diff < 0.05:
                    self.get_logger().info("MATCH! The assumption (Optical->Geometric rotation) is correct.")
                else:
                    self.get_logger().warn("MISMATCH! The assumption might be wrong or TF is different.")
            else:
                self.get_logger().warn("Waiting for transform camera_link -> base_link")

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main():
    rclpy.init()
    node = TFVerifier()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
