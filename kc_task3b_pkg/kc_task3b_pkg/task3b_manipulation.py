#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import time
import threading

from geometry_msgs.msg import Twist 
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from linkattacher_msgs.srv import AttachLink, DetachLink

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

class ManipulationController(Node):
    def __init__(self):
        super().__init__('task3b_manipulation_node')
        self.twist_publisher = self.create_publisher(Twist, '/delta_twist_cmds', 50)
        self.joint_velocity_pub = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 50)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.tf_buffer = Buffer(); self.tf_listener = TransformListener(self.tf_buffer, self)
        self.attach_client = self.create_client(AttachLink, '/attach_link')
        self.detach_client = self.create_client(DetachLink, '/detach_link')
        while not self.attach_client.wait_for_service(timeout_sec=2.0) or not self.detach_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for gripper services...')
        self.current_joint_angles = None; self.team_id = "1994"
        self.ebot_docked = False
        self.detection_sub = self.create_subscription(String, '/detection_status', self.detection_callback, 10)
        
        self.cartesian_kp = 7
        self.cartesian_threshold = 0.05
        self.joint_kp = 20; self.joint_ki = 0.05; self.joint_kd = 2
        self.joint_integral_error = np.zeros(6); self.joint_previous_error = np.zeros(6); self.last_time = self.get_clock().now()
        self.joint_threshold = 0.05
        self.trashbin_location = [-0.8028, -0.001708, 0.10871]
        # Adjusted drop location to be within reach (original: [0.86812, ...])
        # Subtracting ~0.15m from X to account for gripper length and reach limit
        self.ebot_drop_location = [0.72, -0.059204, 0.30196]
        self.home_angles = [0, -90, 90, -90, -90, 0]
        
        self.TASK_SEQUENCE = [
            {"type": "joint", "angles": [-180, -28, -107, -45, 90, 180]},
            {"type": "joint", "angles": [-270, -60, -100, -20, 90, 180]},
            {"type": "joint", "angles": [-283, -95, -85, 0, 100, 180]},
            {"type": "action", "action": "pick", "object_name": "fertiliser_can", "frame_name": f"{self.team_id}_fertiliser_1"},
            {"type": "joint", "angles": [-231, -52, -87, -43, 102, 180]},
            {"type": "joint", "angles": [-180, -120, -60, -70, 90, 180]},
            {"type": "joint", "angles": [-170, -117, -58, -95, 91, 180]},
            {"type": "joint", "angles": [-176, -153, -11, -105, 90, 185]},
            {"type": "action", "action": "place", "object_name": "fertiliser_can"},
            {"type": "joint", "angles": [-176, -120, -80, -100, 90, 180]}, # Safe retreat
            {"type": "joint", "angles": [-120, -90, -80, -100, 90, 180]},
            {"type": "joint", "angles": [-73, -122, -36, -114, 90, 180]},
            {"type": "action", "action": "pick", "object_name": "bad_fruit", "frame_name": f"{self.team_id}_bad_fruit_1"},
            {"type": "joint", "angles": [-30, -120, 60, -100, 90, 180]},
            {"type": "joint", "angles": [-7, -134, -67, -68, 90, 180]},
            {"type": "action", "action": "place", "object_name": "bad_fruit", "location": self.trashbin_location},
            {"type": "joint", "angles": [-7, -134, -67, -68, 90, 180]},  
            {"type": "joint", "angles": [-30, -120, 60, -100, 90, 180]},
            {"type": "joint", "angles": [-73, -122, -36, -114, 90, 180]},
            {"type": "action", "action": "pick", "object_name": "bad_fruit", "frame_name": f"{self.team_id}_bad_fruit_2"},
            {"type": "joint", "angles": [-30, -120, 60, -100, 90, 180]},
            {"type": "joint", "angles": [-7, -134, -67, -68, 90, 180]},
            {"type": "action", "action": "place", "object_name": "bad_fruit", "location": self.trashbin_location},
            {"type": "joint", "angles": [-7, -134, -67, -68, 90, 180]},
            {"type": "joint", "angles": [-30, -120, 60, -100, 90, 180]},
            {"type": "joint", "angles": [-73, -122, -36, -114, 90, 180]},
            {"type": "action", "action": "pick", "object_name": "bad_fruit", "frame_name": f"{self.team_id}_bad_fruit_3"},
            {"type": "joint", "angles": [-30, -120, 60, -100, 90, 180]},
            {"type": "joint", "angles": [-7, -134, -67, -68, 90, 180]},
            {"type": "action", "action": "place", "object_name": "bad_fruit", "location": self.trashbin_location},
        ]
        self.get_logger().info("Manipulation Controller started.")

    def joint_state_callback(self, msg):
        if msg.name and all(name in msg.name for name in JOINT_NAMES):
            self.current_joint_angles = np.array([msg.position[msg.name.index(name)] for name in JOINT_NAMES])

    def detection_callback(self, msg):
        if msg.data.startswith("DOCK_STATION"):
            self.get_logger().info("eBot Docked confirmed!")
            self.ebot_docked = True

    def wait_for_object_position(self, object_frame):
        self.get_logger().info(f"Waiting for TF frame '{object_frame}'...")
        self.get_logger().info(f"Waiting for TF frame '{object_frame}'...")
        # rate = self.create_rate(5000) 
        while rclpy.ok():
            try:
                if self.tf_buffer.can_transform('base_link', object_frame, rclpy.time.Time()):
                    t = self.tf_buffer.lookup_transform('base_link', object_frame, rclpy.time.Time())
                    pos = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
                    self.get_logger().info(f"Found stable TF for '{object_frame}' at {pos}")
                    return pos
            except TransformException as e:
                self.get_logger().warn(f"Cannot find TF for '{object_frame}', retrying...", throttle_duration_sec=1.0)
            time.sleep(0.001)
        return None 

    def move_to_position(self, target_pos, timeout=15.0):
        # rate = self.create_rate(5000)
        start_time = time.time()
        last_log_time = start_time
        
        while rclpy.ok():
            if time.time() - start_time > timeout:
                self.get_logger().error(f"move_to_position TIMEOUT after {timeout}s. Target: {target_pos}")
                self.stop_motion()
                return False

            try:
                # --- FIX: Control wrist_3_link as required by the competition rule ---
                t = self.tf_buffer.lookup_transform('base_link', 'wrist_3_link', rclpy.time.Time())
                current_pos = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
                error = np.array(target_pos) - current_pos
                error_norm = np.linalg.norm(error)
                
                if error_norm < self.cartesian_threshold: 
                    self.stop_motion(); return True
                
                # Log progress every 2 seconds
                if time.time() - last_log_time > 2.0:
                    self.get_logger().info(f"Moving to pos... Dist: {error_norm:.4f}m")
                    last_log_time = time.time()

                vel_cmd = self.cartesian_kp * error
                twist_msg = Twist(); twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = vel_cmd
                self.twist_publisher.publish(twist_msg)
                time.sleep(0.001)
            except TransformException: 
                self.get_logger().warn('Waiting for wrist_3_link transform...', throttle_duration_sec=1.0)
        self.stop_motion(); return False 

    def move_to_joint_angles(self, target_angles_deg, timeout=15.0):
        if self.current_joint_angles is None: return False
        target_rad = np.deg2rad(np.array(target_angles_deg))
        # rate = self.create_rate(5000)
        
        start_time = time.time()
        last_log_time = start_time
        
        while rclpy.ok():
            error = target_rad - self.current_joint_angles; error = np.arctan2(np.sin(error), np.cos(error))
            
            if time.time() - start_time > timeout:
                max_error = np.max(np.abs(error))
                if max_error < 0.2: # If reasonably close (within ~11 degrees), accept it
                    self.get_logger().warn(f"move_to_joint_angles TIMEOUT but close enough (Max Error: {max_error:.4f}). Proceeding.")
                    self.stop_motion(); return True
                else:
                    self.get_logger().error(f"move_to_joint_angles TIMEOUT after {timeout}s. Max Error: {max_error:.4f} rad")
                    self.stop_motion()
                    return False

            if np.linalg.norm(error) < self.joint_threshold: 
                self.stop_motion(); return True
            
            # Log progress every 2 seconds
            if time.time() - last_log_time > 2.0:
                self.get_logger().info(f"Moving joints... Max Error: {np.max(np.abs(error)):.4f} rad")
                last_log_time = time.time()

            current_time = self.get_clock().now(); dt = (current_time - self.last_time).nanoseconds / 1e9
            if dt > 0:
                self.joint_integral_error += error * dt
                derivative = (error - self.joint_previous_error) / dt
                vel_cmd = self.joint_kp * error + self.joint_ki * self.joint_integral_error + self.joint_kd * derivative
                self.joint_velocity_pub.publish(Float64MultiArray(data=vel_cmd.tolist()))
            self.joint_previous_error = error; self.last_time = current_time
            time.sleep(0.001)
        self.stop_motion(); return False 

    def get_wrist_position(self):
        try:
            # This is correct, we want the position of wrist_3_link
            t = self.tf_buffer.lookup_transform('base_link', 'wrist_3_link', rclpy.time.Time())
            return np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
        except:
            self.get_logger().error("Could not get wrist_3_link transform!"); return None

    def attach_detach_object(self, object_name, attach=True, target_model='ur5', target_link='wrist_3_link'):
        client = self.attach_client if attach else self.detach_client
        RequestType = AttachLink.Request if attach else DetachLink.Request
        req = RequestType()
        
        req.model1_name = object_name
        req.link1_name = 'body' 

        req.model2_name = target_model
        req.link2_name = target_link
        
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        action = "attach" if attach else "detach"
        if future.result() and future.result().success:
            self.get_logger().info(f'Successfully {action}ed {object_name} to {target_model}::{target_link}')
            return True
        else:
            self.get_logger().error(f'Failed to {action} {object_name} to {target_model}::{target_link}')
            return False

    def stop_motion(self):
        self.twist_publisher.publish(Twist()); self.joint_velocity_pub.publish(Float64MultiArray(data=[0.0]*6))
        self.joint_integral_error = np.zeros(6); self.joint_previous_error = np.zeros(6)

    def handle_pick_action(self, task):
        grasp_pos = self.wait_for_object_position(task["frame_name"])
        if not grasp_pos:
            return False
        self.get_logger().info(f"Moving to pick {task['object_name']} at {grasp_pos}")

        # Retreat position (12cm above object)
        retreat_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2]]

        # Move wrist_3_link directly to grasp_pos
        if not self.move_to_position(grasp_pos): return False
        
        self.stop_motion()
        
        # Pause to stabilize simulation
        self.get_logger().info("Pausing for 0.2 seconds to stabilize simulation...")
        time.sleep(0.2)
        
        wrist_pos = self.get_wrist_position()
        if wrist_pos is not None:
            dist = np.linalg.norm(np.array(grasp_pos) - wrist_pos)
            self.get_logger().info(f"Wrist-to-target distance = {dist:.4f} m")
            
            if dist > 0.1:
                self.get_logger().error(f"Aborting pick: Gripper is too far from target ({dist:.4f}m > 0.1m).")
                return False
        else:
            self.get_logger().error("Aborting pick: Could not get wrist position for safety check.")
            return False

        self.get_logger().info("Attempting to attach object...")
        if not self.attach_detach_object(task["object_name"], attach=True):
            self.get_logger().error(f"Failed to attach {task['object_name']}.")
            return False

        # self.get_logger().info("Pausing for 1 second after attachment.")
        # time.sleep(1.0)

        # Move to retreat_pos after picking
        if not self.move_to_position(retreat_pos): return False
        # time.sleep(0.2)
        return True

    def handle_place_action(self, task):
        loc = task.get("location")
        if loc:
            pre_place_pos = [loc[0], loc[1], loc[2] + 0.1]
        
        # Wait for eBot if placing fertilizer
        if task["object_name"] == "fertiliser_can":
            self.get_logger().info("Waiting for eBot to dock before placing fertilizer...")
            while not self.ebot_docked and rclpy.ok():
                time.sleep(0.5)
            self.get_logger().info("eBot is docked. Proceeding to place.")
            
        if loc:
            if not self.move_to_position(pre_place_pos): return False
            # time.sleep(0.2)
            if not self.move_to_position(loc): return False
            # time.sleep(0.2)
        
        # Detach from gripper
        if not self.attach_detach_object(task["object_name"], attach=False): return False
        
        # If fertilizer, attach to eBot
        if task["object_name"] == "fertiliser_can":
            # Robust Attachment Logic
            time.sleep(2.0) # Wait for stabilization
            
            attached = False
            # Prioritize 'ebot' as model name (from launch file) and 'ebot_base' (from xacro)
            model_candidates = ['ebot', 'ebot_model']
            link_candidates = ['ebot_base', 'ebot_base_link', 'base_link', 'base_footprint']
            
            for attempt in range(3):
                self.get_logger().info(f"Attachment Attempt {attempt+1}/3...")
                for model in model_candidates:
                    for link in link_candidates:
                        self.get_logger().info(f"Trying to attach to {model}::{link}")
                        if self.attach_detach_object(task["object_name"], attach=True, target_model=model, target_link=link):
                            self.get_logger().info(f"SUCCESS: Fertilizer attached to {model}::{link}.")
                            attached = True
                            break
                    if attached: break
                if attached: break
                time.sleep(1.0)
            
            if not attached:
                self.get_logger().error("CRITICAL FAILURE: Could not attach fertilizer to eBot after all attempts. Proceeding anyway...")
                # return True to allow the robot to move away even if attachment failed
                # return False 

        # time.sleep(0.2)
        # if loc:
        #     if not self.move_to_position(pre_place_pos): return False
        return True

    def run_task_sequence(self):
        while self.current_joint_angles is None and rclpy.ok():
            self.get_logger().info('Waiting for joint state...', throttle_duration_sec=5.0); rclpy.spin_once(self, timeout_sec=0.1)
        if not rclpy.ok(): return
        self.get_logger().info("=== STARTING TASK SEQUENCE ===")
        for i, task in enumerate(self.TASK_SEQUENCE):
            if not rclpy.ok(): break
            self.get_logger().info(f"STEP {i+1}: {task}")
            task_type = task.get("type")
            success = False
            if task_type == "joint": success = self.move_to_joint_angles(task["angles"])
            elif task_type == "action":
                action = task.get("action")
                if action == "home": success = self.move_to_joint_angles(self.home_angles)
                elif action == "pick": success = self.handle_pick_action(task)
                elif action == "place": success = self.handle_place_action(task)
            if not success: self.get_logger().error(f"FAIL in STEP {i+1}: {task}"); break 
        self.get_logger().info('=== TASK SEQUENCE FINISHED ===')
        # self.move_to_joint_angles(self.home_angles)

def main(args=None):
    rclpy.init(args=args)
    controller = ManipulationController()
    task_thread = threading.Thread(target=controller.run_task_sequence, daemon=True)
    task_thread.start()
    try: rclpy.spin(controller)
    except KeyboardInterrupt: pass
    finally: controller.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()