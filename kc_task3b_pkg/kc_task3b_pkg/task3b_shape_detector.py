#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Bool
from tf_transformations import euler_from_quaternion
import numpy as np
import math
import time
import cv2

class ShapeDetector(Node):
    def __init__(self):
        super().__init__('task3b_shape_detector_node')
        self.lidar_range_max = 0.70
        self.cooldown_period = 8
        self.img_size = 400
        self.img_resolution = 0.01
        self.min_vertex_distance = 24
        self.max_vertex_distance = 80

        # Dock detection parameters
        self.DOCK_STATION_COORDS = (0.26, -1.95)
        self.DOCK_DETECTION_TOLERANCE = 0.01
        self.dock_procedure_done = False

        self.DETECTION_ZONES = [
            (-1.0, -4.5, 0.0, 0.4),    # Zone 1
            (-3.0, -4.5, -2.0, 0.4),   # Zone 2
        ]
        self.is_processing = False
        self.last_detection_time = 0
        self.current_pose = None
        self.current_yaw = 0.0

        # State Machine Variables
        self.state = "IDLE" # IDLE, WAITING_DELAY, PUBLISHING
        self.pending_detection_shape = None
        self.delay_start_time = None
        self.square_delay = 5.5 # Hold for 2 seconds
        self.triangle_delay = 0.5 # Hold for 2 seconds

        self.status_pub = self.create_publisher(String, '/detection_status', 10)
        self.pause_pub = self.create_publisher(Bool, '/pause_navigation', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, rclpy.qos.qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        self.triangle_buffer = None
        self.triangle_buffer_timeout = 1
        self.plant_id_counter = 1 # Simple counter for plant IDs

        self.timer = self.create_timer(0.1, self.state_machine_tick)

        self.get_logger().info('Task 3B Shape Detector Started.')

    def odom_callback(self, msg):
        self.current_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        _, _, self.current_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # Dock Check
        if self.current_pose and not self.dock_procedure_done and self.state == "IDLE":
            dist_to_dock = math.sqrt((self.current_pose[0] - self.DOCK_STATION_COORDS[0])**2 + (self.current_pose[1] - self.DOCK_STATION_COORDS[1])**2)
            if dist_to_dock < self.DOCK_DETECTION_TOLERANCE:
                self.dock_procedure_done = True
                self.state = "PUBLISHING"
                self.pending_detection_shape = "DOCK_STATION"
                self._publish_status("DOCK_STATION", self.DOCK_STATION_COORDS[0], self.DOCK_STATION_COORDS[1], is_dock=True)

    def scan_callback(self, msg):
        current_time = self.get_clock().now().nanoseconds / 1e9

        if self.state != "IDLE" or self.is_processing or \
           (current_time - self.last_detection_time) < self.cooldown_period or self.current_pose is None:
            return

        self.is_processing = True
        try:
            img = self._create_image_from_scan(msg)
            kernel = np.ones((3,3), np.uint8)
            dilated_img = cv2.dilate(img, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_shapes = []

            for cnt in contours:
                if cv2.contourArea(cnt) < 100: continue # Reduced area threshold
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True) # Increased epsilon for smoother approximation
                filtered_vertices = self._filter_close_vertices(approx)
                num_vertices = len(filtered_vertices)

                shape = None
                if num_vertices == 4:
                    shape = "SQUARE"
                elif num_vertices == 3 and self._has_perpendicular_angle(filtered_vertices):
                    shape = "TRIANGLE"
                
                if shape:
                    M = cv2.moments(cnt)
                    if M["m00"] == 0: continue
                    cx_pixel = int(M["m10"] / M["m00"]); cy_pixel = int(M["m01"] / M["m00"])
                    center_x_local = (cx_pixel - self.img_size / 2) * self.img_resolution
                    center_y_local = -(cy_pixel - self.img_size / 2) * self.img_resolution
                    shape_global_x, shape_global_y = self._transform_to_global(center_x_local, center_y_local)
                    
                    if self._is_shape_in_detection_zone(shape_global_x, shape_global_y):
                        detected_shapes.append({"shape": shape, "local_pos": (center_x_local, center_y_local)})
            
            current_square = next((s for s in detected_shapes if s["shape"] == "SQUARE"), None)
            current_triangle = next((s for s in detected_shapes if s["shape"] == "TRIANGLE"), None)

            accepted_shape = None
            if current_square:
                self.triangle_buffer = None
                accepted_shape = "SQUARE"
            elif current_triangle:
                self.triangle_buffer = {'timestamp': time.time()}
            
            if not current_triangle and self.triangle_buffer:
                if time.time() - self.triangle_buffer['timestamp'] > self.triangle_buffer_timeout:
                    accepted_shape = "TRIANGLE"
                    self.triangle_buffer = None
            
            if accepted_shape:
                self.get_logger().info(f"Accepted {accepted_shape}. Starting delay (Navigation Active).")
                self.state = "WAITING_DELAY"
                self.pending_detection_shape = accepted_shape
                self.delay_start_time = self.get_clock().now()
                self.last_detection_time = current_time
                # Ensure navigation is NOT paused
                self.pause_pub.publish(Bool(data=False))

        except Exception as e:
            self.get_logger().error(f'Error in scan_callback: {e}')
        finally:
            self.is_processing = False

    def state_machine_tick(self):
        now = self.get_clock().now()

        if self.state == "WAITING_DELAY":
            delay_duration = self.square_delay if self.pending_detection_shape == "SQUARE" else self.triangle_delay
            if (now - self.delay_start_time).nanoseconds > (delay_duration * 1e9):
                self.get_logger().info(f"Delay complete for {self.pending_detection_shape}. Entering publishing state.")
                self.state = "PUBLISHING"
                if self.current_pose:
                    robot_x, robot_y = self.current_pose
                    status = "FERTILIZER_REQUIRED" if self.pending_detection_shape == "TRIANGLE" else "BAD_HEALTH"
                    self._publish_status(status, robot_x, robot_y)
                else:
                    self.state = "IDLE"
                    self.pause_pub.publish(Bool(data=False))

    def _filter_close_vertices(self, vertices):
        if len(vertices) < 2: return vertices
        filtered = [vertices[0]]
        for i in range(1, len(vertices)):
            dists = [np.linalg.norm(vertices[i] - fv) for fv in filtered]
            if all(dist > self.min_vertex_distance for dist in dists) and all(dist < self.max_vertex_distance for dist in dists):
                filtered.append(vertices[i])
        if len(filtered) > 2:
             dist_last_first = np.linalg.norm(filtered[-1] - filtered[0])
             if dist_last_first < self.min_vertex_distance: filtered.pop()
        return np.array(filtered)

    def _has_perpendicular_angle(self, vertices):
        if len(vertices) != 3: return False
        pts = [v[0] for v in vertices]
        v1, v2, v3 = np.array(pts[1]) - np.array(pts[0]), np.array(pts[2]) - np.array(pts[1]), np.array(pts[0]) - np.array(pts[2])
        def perp(u, v):
            norm_u, norm_v = np.linalg.norm(u), np.linalg.norm(v)
            if norm_u == 0 or norm_v == 0: return False
            cos_angle = np.dot(u, v) / (norm_u * norm_v)
            return abs(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))) - 90) < 45
        return perp(v1, -v3) or perp(v2, -v1) or perp(v3, -v2)

    def _create_image_from_scan(self, msg):
        img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        for i, range_val in enumerate(msg.ranges):
            if not np.isfinite(range_val) or range_val > self.lidar_range_max: continue
            angle = msg.angle_min + i * msg.angle_increment
            x, y = range_val * math.cos(angle), range_val * math.sin(angle)
            px, py = int(self.img_size / 2 + x / self.img_resolution), int(self.img_size / 2 - y / self.img_resolution)
            if 0 <= px < self.img_size and 0 <= py < self.img_size:
                # Draw larger circles to connect points better
                cv2.circle(img, (px, py), 2, 255, -1)
        return img

    def _transform_to_global(self, local_x, local_y):
        if not self.current_pose: return 0, 0
        robot_x, robot_y = self.current_pose
        global_x = robot_x + local_x * math.cos(self.current_yaw) - local_y * math.sin(self.current_yaw)
        global_y = robot_y + local_x * math.sin(self.current_yaw) + local_y * math.cos(self.current_yaw)
        return global_x, global_y

    def _is_shape_in_detection_zone(self, shape_x, shape_y):
        for zone in self.DETECTION_ZONES:
            if not isinstance(zone, tuple) or len(zone) != 4: continue
            x_min, y_min, x_max, y_max = zone
            if x_min < shape_x < x_max and y_min < shape_y < y_max: return True
        return False

    def _publish_status(self, status, x, y, is_dock=False):
        log_msg = "DOCK STATION" if is_dock else status
        self.get_logger().warn(f"Publishing {log_msg}. Pausing navigation for 2 seconds.")
        
        pause_msg = Bool(data=True)
        for _ in range(5):
            self.pause_pub.publish(pause_msg)
            time.sleep(0.02)

        plant_id = 0
        if not is_dock:
            plant_id = self.plant_id_counter
            self.plant_id_counter += 1

        # Format: status,x,y,plant_ID
        status_msg = String(data=f"{status},{x:.2f},{y:.2f},{plant_id}")
        self.get_logger().info(f'Publishing message: "{status_msg.data}"')
        
        publish_start_time = self.get_clock().now()
        while rclpy.ok() and (self.get_clock().now() - publish_start_time).nanoseconds < 2e9:
            self.status_pub.publish(status_msg)
            time.sleep(0.1)
            
        self.get_logger().info('Resuming navigation.')
        resume_msg = Bool(data=False)
        for _ in range(5):
             self.pause_pub.publish(resume_msg)
             time.sleep(0.02)
             
        self.state = "IDLE"
        self.pending_detection_shape = None
        self.delay_start_time = None
        if not is_dock:
            if self.current_pose:
                 dist_to_dock = math.sqrt((self.current_pose[0] - self.DOCK_STATION_COORDS[0])**2 + (self.current_pose[1] - self.DOCK_STATION_COORDS[1])**2)
                 if dist_to_dock > (self.DOCK_DETECTION_TOLERANCE + 0.5):
                      self.dock_procedure_done = False

def main(args=None):
    rclpy.init(args=args)
    node = ShapeDetector()
    try: rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException): pass
    finally:
        cv2.destroyAllWindows()
        if 'node' in locals() and rclpy.ok() and node.context.ok():
            node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
