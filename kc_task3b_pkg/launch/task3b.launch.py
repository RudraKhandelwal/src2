from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='kc_task3b_pkg',
            executable='task3b_perception',
            name='task3b_perception_node',
            output='screen'
        ),
        Node(
            package='kc_task3b_pkg',
            executable='task3b_controller',
            name='task3b_controller_node',
            output='screen'
        ),
        Node(
            package='kc_task3b_pkg',
            executable='task3b_manipulation',
            name='task3b_manipulation_node',
            output='screen'
        ),
        Node(
            package='kc_task3b_pkg',
            executable='task3b_shape_detector',
            name='task3b_shape_detector_node',
            output='screen'
        ),
    ])
