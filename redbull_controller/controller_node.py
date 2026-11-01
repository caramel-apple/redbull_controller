
import csv
import math
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.clock import Clock
from rclpy.clock_type import ClockType

from ament_index_python.packages import get_package_share_directory

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.qos import qos_profile_sensor_data


class PurePursuit(Node):

    def __init__(self):
        super().__init__('redbull_controller')

        # Controller parameters
        self.declare_parameter('max_speed', 0.5)
        self.declare_parameter('lookahead_distance', 1.0)
        self.declare_parameter('wheelbase', 0.3302)
        self.declare_parameter('max_steering', 0.4189)
        self.declare_parameter('safety_distance', 0.45)

        self.max_speed = float(
            self.get_parameter('max_speed').value
        )
        self.lookahead = float(
            self.get_parameter('lookahead_distance').value
        )
        self.wheelbase = float(
            self.get_parameter('wheelbase').value
        )
        self.max_steering = float(
            self.get_parameter('max_steering').value
        )
        self.safety_distance = float(
            self.get_parameter('safety_distance').value
        )

        # Load Levine centerline waypoints
        maps_directory = Path(
            get_package_share_directory('f1tenth_gym_ros')
        ) / 'maps'

        waypoint_file = maps_directory / 'levine_centerline.csv'

        self.waypoints = []

        with open(waypoint_file, 'r') as file:
            reader = csv.reader(file)

            for row in reader:
                if not row or row[0].strip().startswith('#'):
                    continue

                self.waypoints.append((
                    float(row[0]),
                    float(row[1])
                ))

        if len(self.waypoints) < 3:
            raise RuntimeError('Not enough waypoints loaded')

        self.last_nearest_index = None

        # Sensor data
        self.last_scan_time = None
        self.last_odom_time = None
        self.front_obstacle_distance = None

        # Publisher
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )

        # LiDAR subscription
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data
        )

        # Odometry subscription
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.odom_callback,
            10
        )

        # Watchdog using steady time
        self.watchdog = self.create_timer(
            0.1,
            self.watchdog_callback,
            clock=Clock(clock_type=ClockType.STEADY_TIME)
        )

        self.get_logger().info(
            f'Pure Pursuit started. Loaded {len(self.waypoints)} waypoints.'
        )

    def publish_drive(self, speed, steering):

        msg = AckermannDriveStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)

        self.drive_publisher.publish(msg)

    def publish_stop(self):
        self.publish_drive(0.0, 0.0)

    def scan_callback(self, msg):

        self.last_scan_time = time.monotonic()

        front_distances = []

        for i, distance in enumerate(msg.ranges):

            angle = (
                msg.angle_min + i * msg.angle_increment
            )

            # Only inspect the forward-facing sector
            if abs(angle) <= math.radians(15):

                if (
                    math.isfinite(distance)
                    and msg.range_min < distance <= msg.range_max
                ):
                    front_distances.append(distance)

        if front_distances:
            self.front_obstacle_distance = min(front_distances)
        else:
            self.front_obstacle_distance = None

    def watchdog_callback(self):

        now = time.monotonic()

        if self.last_scan_time is None:
            self.publish_stop()
            return

        if self.last_odom_time is None:
            self.publish_stop()
            return

        if now - self.last_scan_time > 0.5:
            self.publish_stop()
            return

        if now - self.last_odom_time > 0.5:
            self.publish_stop()
            return

        if self.front_obstacle_distance is None:
            self.publish_stop()
            return

        if self.front_obstacle_distance < self.safety_distance:
            self.publish_stop()

    def quaternion_to_yaw(self, orientation):

        siny = 2.0 * (
            orientation.w * orientation.z
            + orientation.x * orientation.y
        )

        cosy = 1.0 - 2.0 * (
            orientation.y ** 2
            + orientation.z ** 2
        )

        return math.atan2(siny, cosy)

    def find_nearest_waypoint(self, x, y):

        count = len(self.waypoints)

        if self.last_nearest_index is None:
            candidates = range(count)
        else:
            candidates = [
                (self.last_nearest_index + offset) % count
                for offset in range(-8, 51)
            ]

        best_index = min(
            candidates,
            key=lambda i: (
                (self.waypoints[i][0] - x) ** 2
                + (self.waypoints[i][1] - y) ** 2
            )
        )

        self.last_nearest_index = best_index

        return best_index

    def find_lookahead_point(self, nearest_index, x, y, yaw):

        count = len(self.waypoints)

        accumulated_distance = 0.0

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        previous = self.waypoints[nearest_index]

        for offset in range(1, count + 1):

            index = (nearest_index + offset) % count

            point = self.waypoints[index]

            segment_distance = math.hypot(
                point[0] - previous[0],
                point[1] - previous[1]
            )

            accumulated_distance += segment_distance
            previous = point

            if accumulated_distance < self.lookahead:
                continue

            dx = point[0] - x
            dy = point[1] - y

            # Transform waypoint into vehicle coordinates
            local_x = cos_yaw * dx + sin_yaw * dy
            local_y = -sin_yaw * dx + cos_yaw * dy

            if local_x > 0.05:
                return local_x, local_y

        return None

    def odom_callback(self, msg):

        self.last_odom_time = time.monotonic()

        # Require fresh LiDAR before driving
        if self.last_scan_time is None:
            self.publish_stop()
            return

        if time.monotonic() - self.last_scan_time > 0.5:
            self.publish_stop()
            return

        if self.front_obstacle_distance is None:
            self.publish_stop()
            return

        if self.front_obstacle_distance < self.safety_distance:
            self.publish_stop()
            return

        # Get vehicle position
        position = msg.pose.pose.position

        x = position.x
        y = position.y

        yaw = self.quaternion_to_yaw(
            msg.pose.pose.orientation
        )

        # Find nearest centerline waypoint
        nearest_index = self.find_nearest_waypoint(x, y)

        # Find lookahead target
        target = self.find_lookahead_point(
            nearest_index,
            x,
            y,
            yaw
        )

        if target is None:
            self.publish_stop()
            return

        local_x, local_y = target

        distance_squared = (
            local_x ** 2 + local_y ** 2
        )

        if distance_squared < 0.01:
            self.publish_stop()
            return

        # Pure Pursuit curvature
        curvature = (
            2.0 * local_y / distance_squared
        )

        # Calculate steering angle
        steering_angle = math.atan(
            self.wheelbase * curvature
        )

        steering_angle = max(
            -self.max_steering,
            min(self.max_steering, steering_angle)
        )

        # Reduce speed when steering sharply
        speed = self.max_speed / (
            1.0 + 2.0 * abs(steering_angle)
        )

        self.publish_drive(speed, steering_angle)


def main(args=None):

    # Disable rclpy's automatic SIGINT shutdown.
    # This lets us publish zero speed before destroying the node.
    rclpy.init(
        args=args,
        signal_handler_options=SignalHandlerOptions.NO
    )

    node = None

    try:
        node = PurePursuit()

        rclpy.spin(node)

    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info(
                'Ctrl+C received. Stopping vehicle...'
            )

    finally:

        if node is not None:

            # Send several final zero-speed commands while ROS is active
            if rclpy.ok():

                for _ in range(10):
                    try:
                        node.publish_stop()
                        time.sleep(0.05)
                    except Exception:
                        break

            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
