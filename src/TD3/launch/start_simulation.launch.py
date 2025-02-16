import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    td3_rl_path = get_package_share_directory('td3_rl')
    
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true', description='Launch Gazebo with GUI'),
        DeclareLaunchArgument('rviz', default_value='true', description='Launch RViz'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([td3_rl_path, 'launch', 'empty_world.launch.py'])),
            launch_arguments={'gui': LaunchConfiguration('gui')}.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([td3_rl_path, 'launch', 'robot.gazebo.launch.py'])),
            launch_arguments={
                'model': 'burger',
                'x': '0.0',
                'y': '0.0',
                'z': '0.01',
                'R': '0.0',
                'P': '0.0',
                'Y': '0.0',
            }.items(),
        ),

        GroupAction(
            condition=IfCondition(LaunchConfiguration('rviz')),
            actions=[
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz',
                    arguments=['-d', os.path.join(td3_rl_path, 'rviz', 'turtlebot3.rviz')],
                )
            ],
        ),
    ])
