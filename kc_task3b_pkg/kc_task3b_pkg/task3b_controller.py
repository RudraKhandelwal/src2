#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Bool
from tf_transformations import euler_from_quaternion
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped
import math
import numpy as np
import time

GLOBAL_WAYPOINTS = [
    (0.26, -5.0, 1.57, False),
    (0.26, -1.95, 1.57, True),     # P1 (Dock Station)
    (0.26, 1.2, 1.57, False),
    (-1.53, 0.6, -1.57, False),
    (-1.53, -0.67, -1.57, True),    # P2 (Strict)
    (-1.53, -6.61, -1.57, True)     # Final Stop (Strict)
]

KP_ANGULAR = 3.0
KI_ANGULAR = 0.0
KD_ANGULAR = 0.05
KP_LINEAR = 10
WAYPOINT_SWITCH_TOLERANCE = 0.6
STRICT_GOAL_TOLERANCE = 0.2
YAW_TOLERANCE = math.radians(8)
CONSTANT_TURN_SPEED = 1.0
OBSTACLE_THRESHOLD = 0.25
MAX_LINEAR_VEL = 50.0
MAX_ANGULAR_VEL = 50.0

class PIDController:
    def __init__(self, kp, ki, kd, max_out, min_out):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.max_out, self.min_out = max_out, min_out
        self.integral, self.last_error, self.last_time = 0.0, 0.0, time.time()
    def update(self, error):
        dt = time.time() - self.last_time
        if dt == 0: return 0.0
        p_term, self.integral = self.kp * error, self.integral + error * dt
        i_term, derivative = self.ki * self.integral, (error - self.last_error) / dt
        d_term = self.kd * derivative
        output = np.clip(p_term + i_term + d_term, self.min_out, self.max_out)
        self.last_error, self.last_time = error, time.time()
        return output
    def reset(self):
        self.integral, self.last_error, self.last_time = 0.0, 0.0, time.time()

class EbotNavigation(Node):
    def __init__(self):
        super().__init__('ebot_nav_task2a')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.pause_sub = self.create_subscription(Bool, '/pause_navigation', self.pause_callback, 10)
        self.angular_pid = PIDController(KP_ANGULAR, KI_ANGULAR, KD_ANGULAR, MAX_ANGULAR_VEL, -MAX_ANGULAR_VEL)
        
        # TF Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.global_waypoints, self.current_waypoint_index = GLOBAL_WAYPOINTS, 0
        self.is_navigating, self.is_paused, self.is_correcting_yaw = True, False, False
        self.current_x, self.current_y, self.current_yaw = -1.5339, -6.6156, 1.57
        self.lidar_ranges, self.odom_received = None, False
        self.get_logger().info('Navigator Node (Driver Only) Initialized.')





    def odom_callback(self, msg):
        if not self.odom_received: self.get_logger().info('Odometry data received!'); self.odom_received = True
        self.current_x, self.current_y = msg.pose.pose.position.x, msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.current_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.navigate()

    def lidar_callback(self, msg): self.lidar_ranges = np.array(msg.ranges)
    
    def pause_callback(self, msg):
        if msg.data and not self.is_paused:
            self.is_paused = True
            self.stop_robot(log=False)
            self.get_logger().warn('Navigation PAUSED by detector.')
        elif not msg.data and self.is_paused:
            # This is now handled by the detector, but keep for shape detection
            self.is_paused = False
            self.angular_pid.reset()
            self.get_logger().info('Navigation RESUMED by detector.')

    def navigate(self):
        if self.is_paused or not self.is_navigating or not self.odom_received or self.current_waypoint_index >= len(self.global_waypoints):
            self.stop_robot(); return

        target_x, target_y, target_yaw, is_strict = self.global_waypoints[self.current_waypoint_index]
        if self.is_correcting_yaw: self._correct_final_orientation(target_yaw); return
        
        twist_msg = Twist()
        min_front_dist, left_min, right_min = self._check_front_obstacles()
        if min_front_dist < OBSTACLE_THRESHOLD:
            self.get_logger().warn('Obstacle detected!', throttle_duration_sec=2)
            self._simple_avoidance_control(twist_msg, left_min, right_min)
            self.cmd_vel_pub.publish(twist_msg); return
        
        self._goto_goal_control(target_x, target_y, is_strict, twist_msg)
        self.cmd_vel_pub.publish(twist_msg)
        
        # Check for TFs
        self.check_perception_tfs()

    def check_perception_tfs(self):
        # Example: Check for fertilizer
        try:
            if self.tf_buffer.can_transform('base_link', '1994_fertilizer_1', rclpy.time.Time()):
                trans = self.tf_buffer.lookup_transform('base_link', '1994_fertilizer_1', rclpy.time.Time())
                self.get_logger().info(f"Controller detected Fertilizer at: {trans.transform.translation.x}, {trans.transform.translation.y}", throttle_duration_sec=2.0)
        except Exception:
            pass

    def _simple_avoidance_control(self, twist_msg, left_min, right_min):
        twist_msg.linear.x = 0.0; self.angular_pid.reset()
        twist_msg.angular.z = -MAX_ANGULAR_VEL * 0.7 if left_min < right_min else MAX_ANGULAR_VEL * 0.7

    def _goto_goal_control(self, target_x, target_y, is_strict, twist_msg):
        distance = math.sqrt((target_x - self.current_x)**2 + (target_y - self.current_y)**2)
        tolerance = STRICT_GOAL_TOLERANCE if is_strict else WAYPOINT_SWITCH_TOLERANCE
        if distance < tolerance:
            if is_strict:
                self.get_logger().info(f"Strict Goal {self.current_waypoint_index + 1} position reached. Correcting orientation.")
                self.stop_robot(); self.is_correcting_yaw = True; return
            else:
                self.get_logger().info(f"Switching to waypoint {self.current_waypoint_index + 2}")
                self.current_waypoint_index += 1; self.angular_pid.reset(); return
        angle_to_target = math.atan2(target_y - self.current_y, target_x - self.current_x)
        error = self.normalize_angle(angle_to_target - self.current_yaw)
        linear_vel = np.clip(KP_LINEAR * distance, 0.0, MAX_LINEAR_VEL)
        angular_vel = self.angular_pid.update(error)
        if abs(error) > math.radians(45): linear_vel *= 0.5
        twist_msg.linear.x, twist_msg.angular.z = linear_vel, angular_vel

    def _correct_final_orientation(self, target_yaw):
        error = self.normalize_angle(target_yaw - self.current_yaw)
        if abs(error) < YAW_TOLERANCE:
            self.get_logger().info("Orientation corrected.")
            self.stop_robot()
            
            is_final_task_goal = (self.current_waypoint_index == len(self.global_waypoints) - 1)
            if is_final_task_goal:
                self.get_logger().info("Final waypoint reached. Task complete.")
                self.is_navigating = False; self.destroy_node(); return

            self.current_waypoint_index += 1; self.is_correcting_yaw = False
            self.angular_pid.reset()
            self.get_logger().info(f"Proceeding to next waypoint ({self.current_waypoint_index + 1}).")
        else:
            twist = Twist(); twist.angular.z = CONSTANT_TURN_SPEED if error > 0 else -CONSTANT_TURN_SPEED
            self.cmd_vel_pub.publish(twist)

    def _check_front_obstacles(self):
        if self.lidar_ranges is None or self.lidar_ranges.size == 0: return float('inf'), float('inf'), float('inf')
        def min_f(arr): f_arr = arr[np.isfinite(arr)]; return np.min(f_arr) if f_arr.size > 0 else float('inf')
        left = min_f(self.lidar_ranges[300:359]); right = min_f(self.lidar_ranges[0:60])
        return min(left, right), left, right

    def stop_robot(self, log=True):
        if log: self.get_logger().info('Robot stopping.', throttle_duration_sec=2.0)
        self.cmd_vel_pub.publish(Twist())

    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    nav = EbotNavigation()
    try: rclpy.spin(nav)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException): pass
    finally:
        if 'nav' in locals() and rclpy.ok() and nav.context.ok(): nav.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()