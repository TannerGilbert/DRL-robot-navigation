import glob

from setuptools import find_packages, setup

package_name = 'td3_rl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.py') + glob.glob('launch/*.xml')),
        ('share/' + package_name + '/rviz', glob.glob('rviz/*.rviz')),
        ('share/' + package_name + '/worlds', glob.glob('worlds/*.world')),
        ('share/' + package_name + '/pytorch_models', glob.glob('pytorch_models/*')),
        ('share/' + package_name + '/config', glob.glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Gilbert Tanner',
    maintainer_email='gilberttanner.contact@gmail.com',
    description='TD3 Reinforcement Learning integrated with ROS and Gazebo',
    license='BSD',
    entry_points={
        'console_scripts': [
            'test_td3 = td3_rl.test_td3:main',
        ],
    },
)
