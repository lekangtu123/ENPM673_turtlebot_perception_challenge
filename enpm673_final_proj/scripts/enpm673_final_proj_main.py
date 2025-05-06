#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO

from ament_index_python.packages import get_package_share_directory
import os


class ArucoNavigatorNode(Node):
    def __init__(self):
        super().__init__("aruco_navigator")

        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()

        # Subscription
        self.subscription_image = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, 10
        )

        # Publisher
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz

        # Internal state
        self.marker_center = None
        self.image_width = None
        self.stop_sign_detected = False
        self.frame = None  # To store the current frame for processing

        # PD controller parameters
        self.Kp_linear = 0.3
        self.Kd_linear = 0.0000000001
        self.Kp_angular = 0.005
        self.Kd_angular = 0.001

        # Previous errors for derivative control
        self.prev_angular_error = 0.0

        # Load ArUco dictionary and parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Load YOLOv8 model for stop sign detection
        pkg_share = get_package_share_directory('enpm673_final_proj')
        model_path = os.path.join(pkg_share, 'models', 'best.pt')
        self.stop_sign_model = YOLO(model_path)
        self.conf_threshold = 0.5  # Confidence threshold for stop sign detection


        # Optical‑Flow unknown‑obstacle variables
        self.of_prev_gray = None
        self.of_mask = None
        self.of_obstacle = False
        self.of_fast_thr = 25
        self.of_fast_cnt_thr = 10
        self.of_safe_frames = 0
        self.of_lk_params = dict(winSize=(15, 15), maxLevel=2,
                                criteria=(cv2.TERM_CRITERIA_EPS |
                                   cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.of_feature_params = dict(maxCorners=500, qualityLevel=0.01,
                                    minDistance=7, blockSize=7)

        self.fast_movement_count = 0
        self.wait_counter        = 0


    def image_callback(self, msg):
        """Process RGB image and detect ArUco marker and stop signs."""
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            # Create a named window with a specific size
            cv2.namedWindow('Camera Feed With Markers', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Camera Feed With Markers', 640, 480)

            # Detect ArUco markers
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params
            )

            if ids is not None:
                marker_corners = corners[0]
                self.marker_center = np.mean(marker_corners[0], axis=0)
                cx, cy = int(self.marker_center[0]), int(self.marker_center[1])
                self.image_width = self.frame.shape[1]  # Store image width for control

                self.get_logger().info(
                    f"Marker detected at ({cx}, {cy}) with ID: {ids[0][0]}"
                )
                self.frame = cv2.aruco.drawDetectedMarkers(self.frame, corners, ids)

                # Overlay the marker coordinates
                cv2.putText(
                    self.frame,
                    f"Coords: ({cx}, {cy})",
                    (cx, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            else:
                self.marker_center = None
                self.get_logger().info("No marker detected.")

            # Detect stop signs
            self.detect_stop_signs()


            # Optical‑Flow unknown‑obstacle detection
            self.detect_unknown_obstacle(gray)
            # 



            cv2.imshow('Camera Feed With Markers', self.frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {str(e)}")

    def detect_stop_signs(self):
        """Detect stop signs in the current frame using YOLOv8."""
        if self.frame is not None:
            # Reset detection flag
            self.stop_sign_detected = False

            # Perform detection
            results = self.stop_sign_model(self.frame)

            # Process results
            for result in results:
                boxes = result.boxes.cpu().numpy()  # Convert boxes to numpy array
                for box in boxes:
                    # Check if detected class is stop sign (assuming class 0 is stop sign)
                    if box.cls[0] == 0 and box.conf[0] > self.conf_threshold:
                        self.stop_sign_detected = True

                        # Convert coordinates to integers
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Draw bounding box (for visualization)
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(self.frame,
                                    f"Stop Sign {float(box.conf[0]):.2f}",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 255),
                                    2)

            # Display detection status
            status = "STOP SIGN DETECTED!" if self.stop_sign_detected else "No stop sign"
            cv2.putText(self.frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)



    # Optical‑Flow unknown‑obstacle helper
    def detect_unknown_obstacle(self, gray):
        """
        LK optical-flow obstacle detector, using full frame (no horizon ROI).
        If more than 10 feature points move >25px between two frames,
        self.of_obstacle = True and remains True for 50 frames (wait_counter).
        """

        # First frame: initialize previous frame and return
        if self.of_prev_gray is None:
            self.of_prev_gray = gray
            self.of_mask = np.zeros_like(self.frame)
            self.wait_counter = 0
            self.fast_movement_count = 0
            return

        # Parameters copied from original perform_optical_flow
        initial_points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=500, qualityLevel=0.01,
            minDistance=5, blockSize=7
        )
        # reset mask each frame so lines don't accumulate
        self.of_mask = np.zeros_like(self.frame)

        if initial_points is not None and initial_points.size > 0:
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.of_prev_gray, gray, initial_points, None, **self.of_lk_params
            )

            if new_points is not None and status.sum() > 0:
                good_new = new_points[status == 1]
                good_old = initial_points[status == 1]

                fast_cnt = 0
                for new, old in zip(good_new, good_old):
                    if np.linalg.norm(new - old) > 25:
                        fast_cnt += 1
                    x_new, y_new = new.ravel().astype(int)
                    x_old, y_old = old.ravel().astype(int)
                    cv2.line(self.of_mask, (x_old, y_old),
                             (x_new, y_new), (0, 255, 0), 2)

                # same logic as original: 10 fast points → decide obstacle
                if fast_cnt > 10:
                    self.fast_movement_count += 1
                    if self.fast_movement_count > 1:        # keep counter ≥1
                        self.of_obstacle = True
                        self.wait_counter = 0
                # wait_counter to release
                if self.of_obstacle:
                    if self.wait_counter < 50:
                        self.wait_counter += 1
                    else:
                        self.of_obstacle = False
                        self.fast_movement_count = 0
        else:
            # no points → treat as safe frame
            if self.wait_counter < 50:
                self.wait_counter += 1
            else:
                self.of_obstacle = False

        # Overlay visualisation
        self.frame[:] = cv2.add(self.frame, self.of_mask)

        # cache current gray
        self.of_prev_gray = gray
    #
    
    

    def timer_callback(self):
        """Control loop to navigate to the ArUco marker with stop sign handling."""
        twist = Twist()

        if self.stop_sign_detected:
            # Stop if stop sign is detected
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.get_logger().info("Stop sign detected - stopping!")
            
            
        # Obstacle priority just below stop sign
        elif self.of_obstacle:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.get_logger().info("Unknown obstacle detected - stopping!")
        # 
        
        
        elif self.marker_center is not None and self.image_width is not None:
            cx = int(self.marker_center[0])

            # Calculate angular error (horizontal offset from image center)
            angular_error = cx - self.image_width // 2

            # Calculate angular derivative
            angular_derivative = angular_error - self.prev_angular_error

            # PD control for angular velocity (to center the marker)
            twist.angular.z = -(
                self.Kp_angular * angular_error + self.Kd_angular * angular_derivative
            )

            # Always move forward when marker is detected
            twist.linear.x = self.Kp_linear

            # Update previous error
            self.prev_angular_error = angular_error

            self.get_logger().info("Moving towards marker.")
        else:
            # Stop if no marker is detected
            twist.linear.x = 0.0
            twist.angular.z = 0.1
            self.get_logger().info("No marker detected. Searching for marker...")

        # Publish the Twist message
        self.publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoNavigatorNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
