#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D, Twist, PointStamped
from std_msgs.msg import Bool, String
from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs # Import for PointStamped transforms
import numpy as np
import math
import cv2
import time
from rclpy.duration import Duration


class ShapeTracker:
    def __init__(self):
        self.first_seen = None
        self.last_seen = None
        self.frames_seen = 0
        self.frames_total = 0
        
    def update(self, is_seen):
        now = time.time()
        if self.first_seen is None:
            if is_seen:
                self.first_seen = now
                self.last_seen = now
                self.frames_seen = 1
                self.frames_total = 1
        else:
            self.frames_total += 1
            if is_seen:
                self.frames_seen += 1
                self.last_seen = now
                
    def get_stats(self):
        if self.first_seen is None: return 0.0, 0.0
        duration = time.time() - self.first_seen
        consistency = self.frames_seen / self.frames_total if self.frames_total > 0 else 0.0
        return duration, consistency
        
    def reset(self):
        self.first_seen = None
        self.last_seen = None
        self.frames_seen = 0
        self.frames_total = 0

class ShapeDetector(Node):
    def __init__(self):
        super().__init__('task3b_shape_detector_node')
        
        # --- TUNING PARAMETERS ---
        # 1. RDP Epsilon: The "Simplification" factor.
        #    0.02 means a point must deviate 2cm from the line to be considered a "corner".
        self.rdp_epsilon = 0.08
        
        # 2. Segment Lengths (Meters)
        self.min_wall_len = 0.50   # Segments longer than 50cm are WALLS
        self.max_shape_len = 0.45  # Shape sides must be shorter than 45cm
        self.min_shape_len = 0.05  # Shape sides must be longer than 5cm (noise filter)
        
        # 3. Classification
        self.max_sqr_side = 0.28   # If a side is > 28cm, it is a TRIANGLE (Big Triangle vs Small Square)
        
        # 4. Clustering
        self.cluster_gap_thresh = 0.20 # 20cm gap breaks a cluster
        
        # Visualization
        self.img_size = 600
        self.scale = 40.0 # Pixels per meter (Zoom level)
        
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, rclpy.qos.qos_profile_sensor_data)
        self.center_pub = self.create_publisher(Pose2D, '/shape_center_location', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pause_pub = self.create_publisher(Bool, '/pause_navigation', 10)
        self.status_pub = self.create_publisher(String, '/detection_status', 10)
        
        # --- STATE VARIABLES ---
        self.tracker = ShapeTracker()
        self.current_shape_type = None # "SQUARE" or "TRIANGLE"
        
        # --- PLANT ZONES (User Provided) ---
        # Format: ID: (x_min, y_min, x_max, y_max)
        self.plant_zones = {
            1: (-1.0, -4.95, 0.0, -3.55),
            2: (-1.0, -3.45, 0.0, -2.05),
            3: (-1.0, -1.95, 0.0, -0.55),
            4: (-1.0, -0.45, 0.0, 0.95),
            5: (-3.0, -4.95, -2.0, -3.55),
            6: (-3.0, -3.45, -2.0, -2.05),
            7: (-3.0, -1.95, -2.0, -0.55),
            8: (-3.0, -0.45, -2.0, 0.95),
        }
        
        self.docking_active = False
        self.dock_target = None # (x, y)
        self.robot_pose = None # (x, y, theta)
        self.scan_frame = None # Store frame_id from scan message
        
        # --- TF2 ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # --- PROXIMITY FILTERING ---
        self.confirmed_shapes = [] # List of (gx, gy) for completed shapes
        self.current_shape_global_loc = None # (gx, gy) of shape currently being docked to
        self.proximity_thresh = 0.25 # 20cm radius to ignore around confirmed shapes
        
        # --- DOCKING PARAMETERS ---
        self.target_stop_dist = 0.03 # Meters (Tolerance for waypoint)
        self.dock_kp_linear = 50
        self.dock_kp_angular = 10
        
        # --- DETECTION ANNULUS (Min/Max Dist) ---
        self.min_sqr_dist = 0.25
        self.max_sqr_dist = 1.1
        
        self.min_tri_dist = 0.1
        self.max_tri_dist = 1

        
        # --- GLOBAL DETECTION ZONES (User Provided) ---
        # Format: (x_min, y_min, x_max, y_max)
        self.detection_zones = [
            (-1.0, -4.8, 0.0, 0.6),    # Zone 1
            (-3.0, -4.8, -2.0, 0.6),   # Zone 2
        ]
        
        self.robot_pose = None # (x, y, theta)
        self.current_twist = None # (linear, angular)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # --- MAIN DOCKING (ID=0) ---
        self.main_dock_loc = (0.26, -1.95)
        self.main_dock_done = False
        self.main_docking_active = False
        
        # --- ALIGNMENT STATE ---
        self.aligning_active = False
        self.align_target_yaw = 0.0
        self.post_align_callback = None
        
        self.window_name = "Math-Based Shape Detector"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Create Sliders for Real-Time Tuning
        cv2.createTrackbar("RDP Epsilon (cm)", self.window_name, int(self.rdp_epsilon*1000), 100, self.nothing)
        cv2.createTrackbar("Max Sqr Side (cm)", self.window_name, int(self.max_sqr_side*100), 100, self.nothing)
        cv2.createTrackbar("Min Wall (cm)", self.window_name, int(self.min_wall_len*100), 200, self.nothing)

        self.get_logger().info('Initialized Math-Based Shape Detector (RDP).')

    def nothing(self, x):
        pass

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, theta = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.robot_pose = (x, y, theta)
        
        # Capture Twist for stability check
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        self.current_twist = (v, w)

    def is_robot_stopped(self):
        if not self.current_twist: return False
        v, w = self.current_twist
        # Relaxed thresholds: 2cm/s linear, ~2.8 deg/s angular
        return abs(v) < 0.02 and abs(w) < 0.05

    def get_global_pose(self, local_x, local_y):
        if not self.scan_frame: return None
        
        try:
            # Create PointStamped in Laser Frame
            p = PointStamped()
            p.header.frame_id = self.scan_frame
            p.header.stamp = rclpy.time.Time().to_msg()
            p.point.x = float(local_x)
            p.point.y = float(local_y)
            p.point.z = 0.0
            
            # Transform to Odom Frame
            # Use latest available transform if time=0 doesn't work well, 
            # but usually Time() is fine if we have a buffer. 
            # Better: use the time of the scan if we had it, but 'now' is close enough for static shapes.
            # Using Time() (0) gets the latest available transform.
            p_global = self.tf_buffer.transform(p, 'odom', timeout=Duration(seconds=0.1))
            
            return (p_global.point.x, p_global.point.y)
            
        except (TransformException, Exception) as e:
            self.get_logger().warn(f"TF Error in get_global_pose: {e}", throttle_duration_sec=1.0)
            return None

    def is_in_zone(self, gx, gy):
        for zone in self.detection_zones:
            x_min, y_min, x_max, y_max = zone
            if x_min < gx < x_max and y_min < gy < y_max:
                return True
        return False

    def scan_callback(self, msg):
        self.scan_frame = msg.header.frame_id
        # 1. Update Parameters from GUI
        try:
            self.rdp_epsilon = cv2.getTrackbarPos("RDP Epsilon (cm)", self.window_name) / 1000.0
            self.max_sqr_side = cv2.getTrackbarPos("Max Sqr Side (cm)", self.window_name) / 100.0
            self.min_wall_len = cv2.getTrackbarPos("Min Wall (cm)", self.window_name) / 100.0
        except: pass

        # 2. Pre-process: Convert Lidar to Clusters of (x,y) points
        clusters = self.get_clusters(msg)
        
        # 3. Visualization Background
        vis_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # 4. Process Each Cluster
        frame_best_shape = None
        
        for cluster in clusters:
            if len(cluster) < 5: continue # Skip noise
            
            # --- STEP A: SIMPLIFY (Ramer-Douglas-Peucker) ---
            # Reduces 100 noisy points to just the ~4-5 key vertices
            vertices = self.rdp_simplify(cluster, self.rdp_epsilon)
            
            # Draw Raw Points (Gray)
            self.draw_polyline(vis_img, cluster, (50, 50, 50), 1)
            # Draw RDP Simplified Line (White)
            self.draw_polyline(vis_img, vertices, (255, 255, 255), 2)
            # Draw Vertices (Blue Dots)
            for p in vertices:
                self.draw_point(vis_img, p, (255, 0, 0))

            # --- STEP B: ANALYZE SEGMENTS ---
            # We look for a pattern: [Short Segment] -> [Corner] -> [Short Segment]
            num_v = len(vertices)
            if num_v < 3: continue
            
            for i in range(1, num_v - 1):
                p_prev = vertices[i-1]
                p_curr = vertices[i]   # The Corner Candidate
                p_next = vertices[i+1]
                
                # Calculate Lengths
                len1 = np.linalg.norm(p_prev - p_curr)
                len2 = np.linalg.norm(p_curr - p_next)
                
                # Check if these are "Shape Sides" (not walls, not tiny noise)
                is_side1 = self.min_shape_len < len1 < self.max_shape_len
                is_side2 = self.min_shape_len < len2 < self.max_shape_len
                
                if is_side1 and is_side2:
                    # FOUND A CORNER connected by two potential shape sides!
                    
                    # Calculate Angle
                    angle_deg = self.calculate_angle(p_prev, p_curr, p_next)
                    
                    # Logic Tree
                    shape_type = None
                    avg_len = (len1 + len2) / 2.0
                    
                    # 1. Check Size (Big Triangle vs Small Square)
                    if avg_len > self.max_sqr_side:
                        shape_type = "TRIANGLE"
                    
                    # 2. Check Angle (Fallback)
                    elif 80 < angle_deg < 110: # Roughly 90
                         shape_type = "SQUARE"
                    elif 30 < angle_deg < 80:  # Roughly 60
                         shape_type = "TRIANGLE"
                    
                    # 3. Draw & Publish
                    if shape_type == "TRIANGLE":
                        self.draw_result(vis_img, p_prev, p_curr, p_next, "TRIANGLE", (255, 0, 255))
                        self.publish_shape(p_prev, p_curr, p_next, None)
                        
                        # Update Best Shape for Logic
                        pts = np.array([p_prev, p_curr, p_next])
                        c = np.mean(pts, axis=0)
                        dist = np.linalg.norm(c)
                        # Prefer closer shapes or just take the last one found
                        frame_best_shape = ("TRIANGLE", c[0], c[1], dist)
                        
                    elif shape_type == "SQUARE":
                        # Infer 4th point
                        p4 = p_prev + (p_next - p_curr)
                        self.draw_result(vis_img, p_prev, p_curr, p_next, "SQUARE", (0, 255, 255), p4)
                        self.publish_shape(p_prev, p_curr, p_next, p4)

                        # Update Best Shape for Logic
                        pts = np.array([p_prev, p_curr, p_next, p4])
                        c = np.mean(pts, axis=0)
                        dist = np.linalg.norm(c)
                        frame_best_shape = ("SQUARE", c[0], c[1], dist)


        self.handle_state_machine(frame_best_shape)

        cv2.imshow(self.window_name, vis_img)
        cv2.waitKey(1)

    # --- CORE ALGORITHMS ---

    def get_clusters(self, msg):
        """ Converts Scan to list of clusters based on distance gaps """
        clusters = []
        current_cluster = []
        
        for i, r in enumerate(msg.ranges):
            if not np.isfinite(r) or r > 10.0: continue
            
            angle = msg.angle_min + i * msg.angle_increment
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            p = np.array([x, y])
            
            if len(current_cluster) > 0:
                dist = np.linalg.norm(p - current_cluster[-1])
                if dist > self.cluster_gap_thresh:
                    clusters.append(current_cluster)
                    current_cluster = []
            
            current_cluster.append(p)
            
        if len(current_cluster) > 0:
            clusters.append(current_cluster)
        return clusters

    def rdp_simplify(self, points, epsilon):
        """ Ramer-Douglas-Peucker geometric simplification """
        dmax = 0.0
        index = 0
        end = len(points) - 1
        
        for i in range(1, end):
            d = self.point_line_distance(points[i], points[0], points[end])
            if d > dmax:
                index = i
                dmax = d
        
        if dmax > epsilon:
            res1 = self.rdp_simplify(points[:index+1], epsilon)
            res2 = self.rdp_simplify(points[index:], epsilon)
            return np.vstack((res1[:-1], res2))
        else:
            return np.array([points[0], points[end]])

    def point_line_distance(self, point, start, end):
        if np.all(start == end): return np.linalg.norm(point - start)
        return np.abs(np.cross(end-start, start-point)) / np.linalg.norm(end-start)

    def calculate_angle(self, p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom == 0: return 0
        cosine = np.dot(v1, v2) / denom
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        return np.degrees(angle)

    # --- VISUALIZATION HELPERS ---
    
    def to_px(self, point):
        # Center the robot at (img_size/2, img_size/2)
        # Flip Y for image coords
        cx, cy = self.img_size // 2, self.img_size // 2
        px = int(cx + point[0] * self.scale)
        py = int(cy - point[1] * self.scale)
        return (px, py)

    def draw_polyline(self, img, points, color, thickness=1):
        if len(points) < 2: return
        pts_px = [self.to_px(p) for p in points]
        cv2.polylines(img, [np.array(pts_px)], False, color, thickness)

    def draw_point(self, img, point, color):
        cv2.circle(img, self.to_px(point), 3, color, -1)

    def draw_result(self, img, p1, p2, p3, label, color, p4=None):
        # Draw the detected corner arms thicker
        self.draw_polyline(img, [p1, p2, p3], color, 3)
        if p4 is not None:
             self.draw_polyline(img, [p3, p4, p1], color, 1) # Close the square
        
        # Label
        cx, cy = self.to_px(p2)
        cv2.putText(img, label, (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Calculate Centroid for Visualization
        if p4 is None:
            pts = np.array([p1, p2, p3])
        else:
            pts = np.array([p1, p2, p3, p4])
        c = np.mean(pts, axis=0)
        
        # Draw Center (Yellow)
        self.draw_point(img, (c[0], c[1]), (0, 255, 255))
        
        # Draw Local Coords
        cv2.putText(img, f"L:({c[0]:.2f}, {c[1]:.2f})", (cx+10, cy+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw Global Coords
        global_pose = self.get_global_pose(c[0], c[1])
        if global_pose:
             gx, gy = global_pose
             cv2.putText(img, f"G:({gx:.2f}, {gy:.2f})", (cx+10, cy+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    def publish_shape(self, p1, p2, p3, p4=None):
        # Calculate Centroid
        if p4 is None: # Triangle
             pts = np.array([p1, p2, p3])
        else: # Square
             pts = np.array([p1, p2, p3, p4])
             
        centroid = np.mean(pts, axis=0)
        
        msg = Pose2D()
        msg.x = float(centroid[0])
        msg.y = float(centroid[1])
        msg.theta = 0.0
        self.center_pub.publish(msg)
        
    def handle_state_machine(self, shape_data):
        """
        shape_data: (shape_type, centroid_x, centroid_y, distance) or None
        """
        now = time.time()
        
        # --- DOCKING MODE ---
        if self.docking_active:
            self.navigate_to_target()
            return
            
        # --- ALIGNMENT MODE ---
        if self.aligning_active:
            self.process_alignment()
            return

        # --- MAIN DOCKING CHECK ---
        if self.main_docking_active:
            # Already performing main docking sequence (blocking/sleeping in thread? No, we should avoid blocking spin)
            # Actually, perform_main_docking uses sleep, which blocks the node. 
            # Since we are in a callback, this is okay for simple logic, but ideally we'd use a timer.
            # For now, we'll assume perform_main_docking handles it and resets the flag.
            return

        self.check_main_docking()
        if self.main_docking_active: return # If we just started main docking, skip detection

        # --- DETECTION MODE ---
        
        # Check Proximity to Confirmed Shapes
        if shape_data:
             global_pose = self.get_global_pose(shape_data[1], shape_data[2])
             if global_pose:
                 gx, gy = global_pose
                 for (cx, cy) in self.confirmed_shapes:
                     dist_to_confirmed = math.sqrt((gx - cx)**2 + (gy - cy)**2)
                     if dist_to_confirmed < self.proximity_thresh:
                         self.get_logger().info(f"Ignoring shape at ({gx:.2f}, {gy:.2f}) - too close to confirmed shape at ({cx:.2f}, {cy:.2f})")
                         shape_data = None # Treat as not seen
                         break
             else:
                 # No odom yet, can't check global proximity
                 pass

        # 1. Determine if we are seeing the SAME shape as before
        is_same_shape = False
        if shape_data:
            s_type = shape_data[0]
            if self.current_shape_type is None:
                self.current_shape_type = s_type
                self.tracker.reset()
            
            if s_type == self.current_shape_type:
                is_same_shape = True
            else:
                # Shape type changed, reset
                self.current_shape_type = s_type
                self.tracker.reset()
                is_same_shape = True # It's a new shape, but "seen" for the new tracker

        # 2. Update Tracker
        self.tracker.update(is_same_shape)
        
        # 3. Check for Reset (if lost for too long)
        if self.tracker.last_seen and (now - self.tracker.last_seen > 0.5):
            self.tracker.reset()
            self.current_shape_type = None
            return

        # 4. Check Logic
        duration, consistency = self.tracker.get_stats()
        
        # Get current distance if available, else use a large value to fail the check
        current_dist = shape_data[3] if shape_data else 100.0
        
        # Check Global Zone
        in_zone = False
        if shape_data and self.robot_pose:
             # shape_data: (type, cx, cy, dist) - cx, cy are LOCAL
             global_pose = self.get_global_pose(shape_data[1], shape_data[2])
             if global_pose:
                 gx, gy = global_pose
                 if self.is_in_zone(gx, gy):
                     in_zone = True
        elif shape_data and self.robot_pose is None:
             # If no odom yet, maybe allow? Or block? 
             # Let's block to be safe, or allow if you trust local only.
             # User code required odom.
             in_zone = False

        if self.current_shape_type and consistency >= 0.85 and in_zone:
            if self.current_shape_type == "SQUARE":
                # Check Annulus
                if self.min_sqr_dist <= current_dist <= self.max_sqr_dist:
                    if duration >= 2.0:
                        global_pose = self.get_global_pose(shape_data[1], shape_data[2])
                        if global_pose:
                            gx, gy = global_pose
                            self.get_logger().info(f"SQUARE Confirmed! Center: ({gx:.4f}, {gy:.4f})")
                            self.calculate_and_start_docking(shape_data)
                else:
                     if duration > 1.0: # Only warn if we've been tracking it a while
                        self.get_logger().info(f"SQUARE detected but out of range ({current_dist:.2f}m). Req: {self.min_sqr_dist}-{self.max_sqr_dist}m", throttle_duration_sec=1.0)
                    
            elif self.current_shape_type == "TRIANGLE":
                # Check Annulus
                if self.min_tri_dist <= current_dist <= self.max_tri_dist:
                    if duration >= 0.5:
                         global_pose = self.get_global_pose(shape_data[1], shape_data[2])
                         if global_pose:
                             gx, gy = global_pose
                             self.get_logger().info(f"TRIANGLE Confirmed! Center: ({gx:.4f}, {gy:.4f})")
                             self.calculate_and_start_docking(shape_data)
                else:
                    if duration > 0.2:
                        self.get_logger().info(f"TRIANGLE detected but out of range ({current_dist:.2f}m). Req: {self.min_tri_dist}-{self.max_tri_dist}m", throttle_duration_sec=1.0)

    def calculate_and_start_docking(self, shape_data):
        # shape_data: (type, cx, cy, dist) - cx, cy are LOCAL
        if not self.robot_pose:
            self.get_logger().error("Cannot dock: No Odometry!")
            return

        global_pose = self.get_global_pose(shape_data[1], shape_data[2])
        if not global_pose:
            self.get_logger().error("Cannot dock: Global pose transform failed!")
            return
        gx, gy = global_pose
        self.current_shape_global_loc = (gx, gy) # Store for later confirmation
        rx, ry, _ = self.robot_pose
        
        # Determine offset direction based on relative X
        # If Shape X < Robot X (Left/Behind), we want Target X > Shape X (Right of shape) -> Shape X + Offset
        # If Shape X > Robot X (Right/Ahead), we want Target X < Shape X (Left of shape) -> Shape X - Offset
        # Note: This assumes robot moves along X or Y primarily. 
        # Let's use the logic: Target is 0.5m away from shape in the direction of the robot.
        
        # Vector from Shape to Robot
        dx = rx - gx
        dy = ry - gy
        
        # We want to maintain the Y coordinate of the shape (align Y)
        # And be 0.5m away in X.
        
        target_y = gy
        
        # Decide X side:
        if rx > gx:
            target_x = gx + 0.5
        else:
            target_x = gx - 0.5
            
        self.get_logger().info(f"Docking Target Calculated: ({target_x:.2f}, {target_y:.2f}) [Shape: ({gx:.2f}, {gy:.2f})]")
        self.start_docking(target_x, target_y)

    def start_docking(self, tx, ty):
        self.docking_active = True
        self.dock_target = (tx, ty)
        
        # Pause the main controller (Robust)
        msg = Bool(data=True)
        for _ in range(5):
            self.pause_pub.publish(msg)
            time.sleep(0.02)
        self.get_logger().info("Taking control for BLIND DOCKING...")

    def navigate_to_target(self):
        if not self.robot_pose or not self.dock_target: return
        
        rx, ry, rtheta = self.robot_pose
        tx, ty = self.dock_target
        
        dist = math.sqrt((tx - rx)**2 + (ty - ry)**2)
        
        if dist < self.target_stop_dist:
            self.get_logger().info("Docking Complete (Waypoint Reached). Finishing Docking...")
            self.finish_shape_docking()
            return

        # PID Control
        angle_to_target = math.atan2(ty - ry, tx - rx)
        angle_error = angle_to_target - rtheta
        
        # Normalize angle
        while angle_error > math.pi: angle_error -= 2*math.pi
        while angle_error < -math.pi: angle_error += 2*math.pi
        
        linear_vel = self.dock_kp_linear * dist
        angular_vel = self.dock_kp_angular * angle_error
        
        # Safety limits
        linear_vel = np.clip(linear_vel, -0.2, 0.2)
        angular_vel = np.clip(angular_vel, -1.0, 1.0)
        
        # Turn in place if error is large
        if abs(angle_error) > 0.5:
            linear_vel = 0.0
            
        twist = Twist()
        twist.linear.x = float(linear_vel)
        twist.angular.z = float(angular_vel)
        self.cmd_vel_pub.publish(twist)

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())

    def publish_completion_status(self):
        # Determine status
        status = "UNKNOWN"
        if self.current_shape_type == "TRIANGLE":
            status = "FERTILIZER_REQUIRED"
        elif self.current_shape_type == "SQUARE":
            status = "BAD_HEALTH"
            
        # HYBRID LOGIC:
        # 1. Use Shape Coordinates for Plant ID (since zones are defined for the trough)
        if self.current_shape_global_loc:
            sx, sy = self.current_shape_global_loc
            plant_id = self.get_plant_id(sx, sy)
        else:
            # Fallback if shape loc missing (shouldn't happen)
            plant_id = "0"

        # 2. Use Robot Coordinates for Reporting (User Request)
        if self.robot_pose:
            rx, ry, _ = self.robot_pose
        else:
            rx, ry = 0.0, 0.0
        
        # Format: status,x,y,plant_ID
        status_msg = String(data=f"{status},{rx:.2f},{ry:.2f},{plant_id}")
        self.get_logger().info(f'Publishing message: "{status_msg.data}"')
        
        # Manual braking logic removed as per user request
        time.sleep(0.5) # Minimal buffer
        
        # Publish for 2 seconds
        publish_start_time = self.get_clock().now()
        while rclpy.ok() and (self.get_clock().now() - publish_start_time).nanoseconds < 2e9:
            self.status_pub.publish(status_msg)
            time.sleep(0.1)
            
        # Post-publish wait
        time.sleep(0.2)
            
        # Add to confirmed list
        if self.current_shape_global_loc:
            self.confirmed_shapes.append(self.current_shape_global_loc)
            self.get_logger().info(f"Shape confirmed at {self.current_shape_global_loc}. Total confirmed: {len(self.confirmed_shapes)}")

        self.get_logger().info('Resuming navigation.')
        resume_msg = Bool(data=False)
        for _ in range(5):
             self.pause_pub.publish(resume_msg)
             time.sleep(0.02)
             
        # Reset State
        self.docking_active = False
        self.dock_target = None
        self.current_shape_type = None
        self.tracker.reset()

    def check_main_docking(self):
        if self.main_dock_done or not self.robot_pose: return
        
        rx, ry, _ = self.robot_pose
        mx, my = self.main_dock_loc
        
        dist = math.sqrt((rx - mx)**2 + (ry - my)**2)
        
        if dist < 0.075: # 7.5cm tolerance (Increased to match nav tolerance)
            self.get_logger().info(f"Arrived at Main Dock Station ({dist:.2f}m). Initiating sequence...")
            self.main_docking_active = True
            self.align_to_lane(self.finish_main_docking)

    def align_to_lane(self, callback):
        """ Sets up non-blocking alignment """
        if not self.robot_pose: 
            if callback: callback()
            return
        
        _, _, current_yaw = self.robot_pose
        
        # Determine closest target (+1.54 or -1.54)
        self.align_target_yaw = 1.54 if abs(current_yaw - 1.54) < abs(current_yaw + 1.54) else -1.54
        
        self.get_logger().info(f"Aligning to {self.align_target_yaw:.2f} rad...")
        self.aligning_active = True
        self.post_align_callback = callback

    def process_alignment(self):
        if not self.robot_pose: return
        _, _, current_yaw = self.robot_pose
        
        error = self.align_target_yaw - current_yaw
        while error > math.pi: error -= 2*math.pi
        while error < -math.pi: error += 2*math.pi
        
        if abs(error) < 0.075: # Tolerance ~4 degrees
            # Strict Stability Check
            if self.current_twist and abs(self.current_twist[1]) < 0.05:
                self.get_logger().info("Alignment Complete & Stable.")
                self.stop_robot()
                self.aligning_active = False
                if self.post_align_callback:
                    self.post_align_callback()
                    self.post_align_callback = None
                return
            else:
                 # In position but still rotating, keep stopping
                 self.stop_robot()
                 return
            
        twist = Twist()
        twist.angular.z = float(np.clip(2.0 * error, -1.0, 1.0))
        self.cmd_vel_pub.publish(twist)

    def finish_shape_docking(self):
        self.stop_robot()
        self.publish_completion_status()

    def finish_main_docking(self):
        self.perform_main_docking()

    def perform_main_docking(self):
        # 1. Pause Navigation (Robust)
        msg = Bool(data=True)
        for _ in range(5):
            self.pause_pub.publish(msg)
            time.sleep(0.02)
        self.get_logger().info("Main Docking: Navigation PAUSED.")
        
        # 2. Publish Status
        # Format: FERTILIZER_REQUIRED,x,y,0
        if self.robot_pose:
            rx, ry, _ = self.robot_pose
        else:
            rx, ry = self.main_dock_loc
            
        status_msg = String(data=f"DOCK_STATION,{rx:.2f},{ry:.2f},0")
        self.get_logger().info(f'Main Docking: Publishing "{status_msg.data}"')
        
        # Manual braking logic removed as per user request
        time.sleep(0.5) # Minimal buffer
        
        # Publish for 2 seconds
        start_time = time.time()
        while time.time() - start_time < 2.0:
            self.status_pub.publish(status_msg)
            time.sleep(0.1)
            
        # Post-publish wait
        time.sleep(0.2)
            
        # 3. Resume Navigation
        self.get_logger().info('Main Docking: Resuming navigation.')
        resume_msg = Bool(data=False)
        for _ in range(5):
             self.pause_pub.publish(resume_msg)
             time.sleep(0.02)
             
        self.main_dock_done = True
        self.main_docking_active = False

    def get_plant_id(self, x, y):
        """ Determines Plant ID based on global coordinates """
        for pid, (x_min, y_min, x_max, y_max) in self.plant_zones.items():
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return str(pid)
        return "0" # Unknown/Outside zones

def main(args=None):
    rclpy.init(args=args)
    node = ShapeDetector()
    try: rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException): pass
    finally:
        cv2.destroyAllWindows()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()