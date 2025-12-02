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
            executable='ebot_nav_task3b',
            name='ebot_nav_task3b_node',
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
            executable='shape_detector_task3b',
            name='shape_detector_task3b_node',
            output='screen'
        ),
    ])
