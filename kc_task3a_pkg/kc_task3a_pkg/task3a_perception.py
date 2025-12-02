#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
* ===============================================
* Krishi coBot (KC) Theme (eYRC 2025-26)
* ===============================================
*
* Filename: task2b_perception.py
*
* Description: This script detects objects, filters for stable positions, 
* and publishes their ideal grasp coordinates as TF frames.
*
*****************************************************************************************
'''

# Team ID:          1994
# Author List:      [ e-Yantra ]
# Filename:         task2b_perception.py

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tf2_ros
from tf2_ros import TransformBroadcaster
import threading
from geometry_msgs.msg import TransformStamped, PointStamped
import tf2_geometry_msgs

SHOW_IMAGE = True

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('task3a_perception_node')
        
        self.team_id = "1994"
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None
        self.camera_k_matrix = None
        self.camera_frame_id = "camera_link_optical" # Default, will be updated from camera_info
        self.lock = threading.Lock()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.cb_group = ReentrantCallbackGroup()

        # TF2 Setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # State for stabilizing TF publications
        self.last_positions = {}
        self.stability_threshold = 0.02  # Only publish if position moves less than 2cm

        # Subscribers and Timers
        # Hardware Topics
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_image_callback, 10, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_image_callback, 10, callback_group=self.cb_group)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10, callback_group=self.cb_group)
        self.create_timer(0.1, self.process_image, callback_group=self.cb_group)
        
        # ArUco setup
        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        else:
            self.aruco_params = cv2.aruco.DetectorParameters()

        global SHOW_IMAGE
        if SHOW_IMAGE:
            try:
                cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
            except cv2.error as e:
                self.get_logger().warn(f"Could not open display window (headless?): {e}. Disabling image visualization.")
                SHOW_IMAGE = False
        self.get_logger().info(f"Perception node started for Team ID: {self.team_id}.")

    def depth_image_callback(self, data):
        try:
            with self.lock:
                self.depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')

    def color_image_callback(self, data):
        try:
            with self.lock:
                self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert color image: {e}')

    def camera_info_callback(self, msg):
        with self.lock:
            if self.camera_k_matrix is None:
                self.camera_k_matrix = np.array(msg.k).reshape((3, 3))
                # The points are calculated in the optical frame (Z forward).
                # If CameraInfo reports 'camera_link' (geometric, X forward), we must use the optical frame instead.
                # We'll default to 'camera_link_optical' if the reported frame doesn't look optical.
                reported_frame = msg.header.frame_id
                if "optical" not in reported_frame:
                    self.camera_frame_id = "camera_link_optical"
                    self.get_logger().warn(f'CameraInfo reported frame "{reported_frame}" which does not look optical. Using "{self.camera_frame_id}" instead to match point calculation.')
                else:
                    self.camera_frame_id = reported_frame
                    self.get_logger().info(f'Camera intrinsics received. Frame ID: {self.camera_frame_id}')
                self.destroy_subscription(self.camera_info_sub)

    def detect_bad_fruit(self, rgb_image):
        bad_fruits_contours = []
        roi_coords = (0, 220, 320, 400)
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 160]); upper_white = np.array([180, 110, 255])
        lower_green = np.array([35, 40, 40]); upper_green = np.array([85, 255, 255])
        lower_red1 = np.array([0, 120, 70]); upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70]); upper_red2 = np.array([180, 255, 255])
        white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        red_mask = cv2.bitwise_or(cv2.inRange(hsv_image, lower_red1, upper_red1), cv2.inRange(hsv_image, lower_red2, upper_red2))
        all_parts_mask = cv2.bitwise_or(cv2.bitwise_or(white_mask, green_mask), red_mask)
        roi_mask = np.zeros_like(all_parts_mask)
        cv2.rectangle(roi_mask, (roi_coords[0], roi_coords[1]), (roi_coords[2], roi_coords[3]), 255, -1)
        all_fruits_mask = cv2.bitwise_and(all_parts_mask, roi_mask)
        all_fruits_mask = cv2.dilate(cv2.erode(all_fruits_mask, None, iterations=1), None, iterations=4)
        contours, _ = cv2.findContours(all_fruits_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            mask = np.zeros_like(all_parts_mask)
            cv2.drawContours(mask, [c], -1, 255, -1)
            if cv2.countNonZero(cv2.bitwise_and(white_mask, mask)) > 150:
                bad_fruits_contours.append(c)
        return bad_fruits_contours

    def detect_aruco_markers(self, rgb_image):
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)
        return corners, ids

    def transform_point(self, point_xyz, target_frame='base_link'):
        try:
            # The input point_xyz is in the OPTICAL frame (Z-forward, X-right, Y-down).
            # The TF frame (self.camera_frame_id) appears to be GEOMETRIC (X-forward, Y-left, Z-up)
            # based on the user's report and manual math.
            # So we must rotate the point to match the Geometric frame convention before transforming.
            # Geometric X = Optical Z
            # Geometric Y = -Optical X
            # Geometric Z = -Optical Y
            
            geom_x = point_xyz[2]
            geom_y = -point_xyz[0]
            geom_z = -point_xyz[1]

            point_stamped = PointStamped()
            point_stamped.header.frame_id = self.camera_frame_id
            point_stamped.header.stamp = rclpy.time.Time().to_msg() # Use latest available transform
            point_stamped.point.x = float(geom_x)
            point_stamped.point.y = float(geom_y)
            point_stamped.point.z = float(geom_z)

            # Wait for transform to be available (with timeout)
            if not self.tf_buffer.can_transform(target_frame, self.camera_frame_id, rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0)):
                self.get_logger().warn(f'Transform from {self.camera_frame_id} to {target_frame} not available yet.')
                return None

            transformed_point = self.tf_buffer.transform(point_stamped, target_frame)
            return [transformed_point.point.x, transformed_point.point.y, transformed_point.point.z]
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'TF2 transform error: {e}')
            return None

    def publish_stable_tf(self, frame_id, position):
        last_pos = self.last_positions.get(frame_id)
        current_pos = np.array(position)

        if last_pos is None or np.linalg.norm(current_pos - last_pos) < self.stability_threshold:
            self.last_positions[frame_id] = current_pos
            
            tf_msg = TransformStamped()
            tf_msg.header.stamp = self.get_clock().now().to_msg()
            tf_msg.header.frame_id = 'base_link'
            tf_msg.child_frame_id = frame_id
            tf_msg.transform.translation.x = position[0]
            tf_msg.transform.translation.y = position[1]
            tf_msg.transform.translation.z = position[2]
            
            self.tf_broadcaster.sendTransform(tf_msg)
            return True
        return False

    def process_image(self):
        with self.lock:
            if self.cv_image is None or self.depth_image is None or self.camera_k_matrix is None: return
            color_image, depth_image, k_matrix = self.cv_image.copy(), self.depth_image.copy(), self.camera_k_matrix.copy()

        fx, fy = k_matrix[0, 0], k_matrix[1, 1]
        cx, cy = k_matrix[0, 2], k_matrix[1, 2]

        # Bad Fruits
        for i, cnt in enumerate(self.detect_bad_fruit(color_image)):
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            center_x, center_y = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            depth = depth_image[center_y, center_x]
            if np.isnan(depth) or depth <= 0.0: continue

            point_z = float(depth); point_x = (center_x - cx) * point_z / fx; point_y = (center_y - cy) * point_z / fy
            
            transformed_pt = self.transform_point([point_x, point_y, point_z])
            if transformed_pt is None: continue
            
            base_x, base_y, base_z = transformed_pt 
            
            frame_id = f'{self.team_id}_bad_fruit_{i+1}'
            self.publish_stable_tf(frame_id, [base_x, base_y, base_z])
            
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(color_image, f"bad_fruit_{i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Fertilizer
        FERTILIZER_ARUCO_ID = 3
        aruco_corners, aruco_ids = self.detect_aruco_markers(color_image)
        if aruco_ids is not None:
            for i, marker_id in enumerate(aruco_ids):
                if marker_id[0] == FERTILIZER_ARUCO_ID:
                    corners = aruco_corners[i][0]
                    center_x, center_y = int(np.mean(corners[:, 0])), int(np.mean(corners[:, 1]))
                    depth = depth_image[center_y, center_x]
                    if not (np.isnan(depth) or depth <= 0.0):
                        point_z = float(depth); point_x = (center_x - cx) * point_z / fx; point_y = (center_y - cy) * point_z / fy
                        
                        transformed_pt = self.transform_point([point_x, point_y, point_z])
                        if transformed_pt is None: continue

                        base_x, base_y, base_z = transformed_pt
                        
                        # =================================================================
                        # FINAL ADJUSTMENT: Pulling back slightly more to prevent collision
                        # =================================================================
                        y_offset = 0; z_offset = 0
                        grasp_pos = [base_x, base_y + y_offset, base_z + z_offset]
                        frame_id = f'{self.team_id}_fertilizer_1'

                        if self.publish_stable_tf(frame_id, grasp_pos):
                             self.get_logger().info(f"Published STABLE grasp point for Fertiliser: x={grasp_pos[0]:.3f}, y={grasp_pos[1]:.3f}, z={grasp_pos[2]:.3f}")
                        
                        cv2.aruco.drawDetectedMarkers(color_image, [aruco_corners[i]])
                        cv2.putText(color_image, "fertilizer_1", (center_x, center_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    break

        if SHOW_IMAGE:
            cv2.imshow("Object Detection", color_image)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()


#topic names for camera
#fruit colour
#no. of fruit
#fertilizer can id/ aruco tags total and stuff
#base_x = 0.74314 * point_z - 0.66862 * point_y - 1.08
#base_y = -point_x + 0.007
#base_z = -0.66862 * point_z - 0.74314 * point_y + 1.09 
