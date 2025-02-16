import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')

    return LaunchDescription([
        DeclareLaunchArgument('model', default_value='turtlebot3_burger.urdf', description='TurtleBot3 model type'),
        DeclareLaunchArgument('x', default_value='0.0', description='X position'),
        DeclareLaunchArgument('y', default_value='0.0', description='Y position'),
        DeclareLaunchArgument('z', default_value='0.0', description='Z position'),
        DeclareLaunchArgument('R', default_value='0.0', description='Roll'),
        DeclareLaunchArgument('P', default_value='0.0', description='Pitch'),
        DeclareLaunchArgument('Y', default_value='0.0', description='Yaw'),
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use sim time'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
            ),
            launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items()
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')
            ),
            launch_arguments={
                'x_pose': LaunchConfiguration('x'),
                'y_pose': LaunchConfiguration('y')
            }.items()
        )
    ])
