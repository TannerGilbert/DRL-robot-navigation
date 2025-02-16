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

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([td3_rl_path, 'launch', 'start_simulation.launch.py'])),
            launch_arguments={
                'gui': LaunchConfiguration('gui'),
                'rviz': LaunchConfiguration('rviz'),
            }.items(),
        ),

        Node(
            package='td3_rl',
            executable='test_td3',
            name='td3_tester',
            output='screen',
            parameters=[PathJoinSubstitution([td3_rl_path, 'config', 'td3_config.yaml'])],
        ),
    ])