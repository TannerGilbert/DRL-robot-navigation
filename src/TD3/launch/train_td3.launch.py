import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    td3_rl_path = FindPackageShare('td3_rl')
    
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true', description='Launch Gazebo with GUI'),
        DeclareLaunchArgument('rviz', default_value='true', description='Launch RViz'),
        
        # Start Gazebo and spawn robot
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([td3_rl_path, 'launch', 'start_simulation.launch.py'])),
            launch_arguments={
                'gui': LaunchConfiguration('gui'), 
                'rviz': LaunchConfiguration('rviz')
            }.items(),
        ),

        # Start the TD3 training script with parameters loaded from YAML
        Node(
            package='td3_rl',
            executable='train_td3.py',
            name='td3_trainer',
            output='screen',
            parameters=[PathJoinSubstitution([td3_rl_path, 'config', 'td3_config.yaml'])],
        ),
    ])