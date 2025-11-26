#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import time
import threading

from geometry_msgs.msg import Twist 
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from linkattacher_msgs.srv import AttachLink, DetachLink

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

class ManipulationController(Node):
    def __init__(self):
        super().__init__('task2b_manipulation_controller')
        self.twist_publisher = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        self.joint_velocity_pub = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.tf_buffer = Buffer(); self.tf_listener = TransformListener(self.tf_buffer, self)
        self.attach_client = self.create_client(AttachLink, '/attach_link')
        self.detach_client = self.create_client(DetachLink, '/detach_link')
        while not self.attach_client.wait_for_service(timeout_sec=2.0) or not self.detach_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for gripper services...')
        self.current_joint_angles = None; self.team_id = "1994"
        
        self.cartesian_kp = 5
        self.cartesian_threshold = 0.02
        self.joint_kp = 10.0; self.joint_ki = 0.2; self.joint_kd = 1.2
        self.joint_integral_error = np.zeros(6); self.joint_previous_error = np.zeros(6); self.last_time = self.get_clock().now()
        self.joint_threshold = 0.05
        self.trashbin_location = [-0.8028, -0.001708, 0.10871]
        self.ebot_drop_location = [0.67, 0.012, 0.41]
        self.home_angles = [0, -90, 90, -90, -90, 0]
        
        self.TASK_SEQUENCE = [
            {"type": "joint", "angles": [-180, -28, -107, -45, 90, 180]},
            {"type": "joint", "angles": [-270, -60, -100, -20, 90, 180]},
            {"type": "joint", "angles": [-283, -95, -85, 0, 100, 180]},
            {"type": "action", "action": "pick", "object_name": "fertiliser_can", "frame_name": f"{self.team_id}_fertiliser_can"},
            {"type": "joint", "angles": [-231, -52, -87, -43, 102, 180]},
            {"type": "joint", "angles": [-180, -120, -60, -70, 90, 180]},
            {"type": "joint", "angles": [-170, -117, -58, -95, 91, 180]},
            {"type": "action", "action": "place", "object_name": "fertiliser_can", "location": self.ebot_drop_location},
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

    def wait_for_object_position(self, object_frame):
        self.get_logger().info(f"Waiting for TF frame '{object_frame}'...")
        rate = self.create_rate(5000) 
        while rclpy.ok():
            try:
                if self.tf_buffer.can_transform('base_link', object_frame, rclpy.time.Time()):
                    t = self.tf_buffer.lookup_transform('base_link', object_frame, rclpy.time.Time())
                    pos = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
                    self.get_logger().info(f"Found stable TF for '{object_frame}' at {pos}")
                    return pos
            except TransformException as e:
                self.get_logger().warn(f"Cannot find TF for '{object_frame}', retrying...", throttle_duration_sec=1.0)
            rate.sleep()
        return None 

    def move_to_position(self, target_pos):
        rate = self.create_rate(5000)
        while rclpy.ok():
            try:
                # --- FIX: Control wrist_3_link as required by the competition rule ---
                t = self.tf_buffer.lookup_transform('base_link', 'wrist_3_link', rclpy.time.Time())
                error = np.array(target_pos) - [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
                
                if np.linalg.norm(error) < self.cartesian_threshold: 
                    self.stop_motion(); return True
                
                vel_cmd = self.cartesian_kp * error
                twist_msg = Twist(); twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = vel_cmd
                self.twist_publisher.publish(twist_msg)
                rate.sleep()
            except TransformException: 
                self.get_logger().warn('Waiting for wrist_3_link transform...', throttle_duration_sec=1.0)
        self.stop_motion(); return False 

    def move_to_joint_angles(self, target_angles_deg):
        if self.current_joint_angles is None: return False
        target_rad = np.deg2rad(np.array(target_angles_deg))
        rate = self.create_rate(5000)
        while rclpy.ok():
            error = target_rad - self.current_joint_angles; error = np.arctan2(np.sin(error), np.cos(error))
            if np.linalg.norm(error) < self.joint_threshold: 
                self.stop_motion(); return True
            current_time = self.get_clock().now(); dt = (current_time - self.last_time).nanoseconds / 1e9
            if dt > 0:
                self.joint_integral_error += error * dt
                derivative = (error - self.joint_previous_error) / dt
                vel_cmd = self.joint_kp * error + self.joint_ki * self.joint_integral_error + self.joint_kd * derivative
                self.joint_velocity_pub.publish(Float64MultiArray(data=vel_cmd.tolist()))
            self.joint_previous_error = error; self.last_time = current_time
            rate.sleep()
        self.stop_motion(); return False 

    def get_wrist_position(self):
        try:
            # This is correct, we want the position of wrist_3_link
            t = self.tf_buffer.lookup_transform('base_link', 'wrist_3_link', rclpy.time.Time())
            return np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
        except:
            self.get_logger().error("Could not get wrist_3_link transform!"); return None

    def attach_detach_object(self, object_name, attach=True):
        client = self.attach_client if attach else self.detach_client
        RequestType = AttachLink.Request if attach else DetachLink.Request
        req = RequestType()
        
        # This is correct for both fertiliser_can and bad_fruit
        req.model1_name = object_name
        req.link1_name = 'body' 

        # --- FIX: This MUST be wrist_3_link as per the rule ---
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'
        
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        action = "attach" if attach else "detach"
        if future.result() and future.result().success:
            self.get_logger().info(f'Successfully {action}ed {object_name}')
            return True
        else:
            self.get_logger().error(f'Failed to {action} {object_name}')
            return False

    def stop_motion(self):
        self.twist_publisher.publish(Twist()); self.joint_velocity_pub.publish(Float64MultiArray(data=[0.0]*6))
        self.joint_integral_error = np.zeros(6); self.joint_previous_error = np.zeros(6)

    def handle_pick_action(self, task):
        grasp_pos = self.wait_for_object_position(task["frame_name"])
        if not grasp_pos:
            return False
        self.get_logger().info(f"Moving to pick {task['object_name']} at {grasp_pos}")

        # --- FIX: Define a retreat position, but don't move to it yet ---
        # (This is just an arbitrary 12cm above the object origin)
        retreat_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2]]

        # --- FIX: Move wrist_3_link *directly* to the grasp_pos ---
        # This is to satisfy the "< 0.1m distance" rule.
        if not self.move_to_position(grasp_pos): return False
        
        self.stop_motion()
        
        # WORKAROUND: Add a long pause to let the simulation physics settle
        self.get_logger().info("Pausing for 0.2 seconds to stabilize simulation...")
       
        
        wrist_pos = self.get_wrist_position()
        if wrist_pos is not None:
            # --- FIX: Check distance between wrist and target ---
            dist = np.linalg.norm(np.array(grasp_pos) - wrist_pos)
            self.get_logger().info(f"Wrist-to-target distance = {dist:.4f} m")
            
            # This is the rule from your image
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

        self.get_logger().info("Pausing for 1 second after attachment.")
        # time.sleep(1.0)

        # --- FIX: Move to the retreat_pos *after* picking ---
        if not self.move_to_position(retreat_pos): return False
        # time.sleep(0.2)
        return True

    def handle_place_action(self, task):
        loc = task["location"]
        # This logic is fine, as it moves the wrist_3_link
        pre_place_pos = [loc[0], loc[1], loc[2] + 0.1]
        if not self.move_to_position(pre_place_pos): return False
        # time.sleep(0.2)
        if not self.move_to_position(loc): return False
        # time.sleep(0.2)
        if not self.attach_detach_object(task["object_name"], attach=False): return False
        # time.sleep(0.2)
        if not self.move_to_position(pre_place_pos): return False
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
        self.move_to_joint_angles(self.home_angles)

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