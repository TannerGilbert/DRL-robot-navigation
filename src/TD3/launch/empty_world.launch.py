import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    td3_rl_path = get_package_share_directory('td3_rl')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation time'),
        DeclareLaunchArgument('gui', default_value='false', description='Launch Gazebo GUI'),
        DeclareLaunchArgument('world_name', default_value=os.path.join(td3_rl_path, 'worlds/TD3.world'), description='World file'),
        
        # Launch Gazebo server
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={'world': LaunchConfiguration('world_name'), 'verbose': 'true'}.items()
        ),

        # Launch Gazebo client (if GUI is enabled)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
            ),
            launch_arguments={'verbose': 'true'}.items(),
            condition=IfCondition(LaunchConfiguration('gui'))
        ),
    ])