RedBull F1TENTH Controller

ROS 2 Jazzy Python controller implementing Pure Pursuit for the CIE-KFUPM F1TENTH Competition.

Algorithm
Pure Pursuit path tracking using vehicle odometry.
Levine centerline waypoint tracking.
LiDAR forward-obstacle monitoring.
Watchdog that publishes zero-speed commands when sensor data is missing or unsafe.
ROS 2 Interface
Topic	Message type	Direction
/scan	sensor_msgs/msg/LaserScan	Subscribe
/ego_racecar/odom	nav_msgs/msg/Odometry	Subscribe
/drive	ackermann_msgs/msg/AckermannDriveStamped	Publish
Dependencies
ROS 2 Jazzy
rclpy
sensor_msgs
nav_msgs
ackermann_msgs
ament_index_python
f1tenth_gym_ros
Default Parameters
Parameter	Default
max_speed	0.5 m/s
lookahead_distance	1.0 m
wheelbase	0.3302 m
max_steering	0.4189 rad
safety_distance	0.45 m

Maximum requested speed: 0.5 m/s

Build
bash
python /usr/bin/colcon build --symlink-install
source install/setup.bash


Build from the controller workspace after sourcing the ROS 2 and competition environments.

Run
bash
ros2 run redbull_controller controller

Competition Information

Uses the officially supplied Levine centerline for Stage 1. Map and waypoint usage must follow the restrictions of the active competition stage.
