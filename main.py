
import os
import cv2
import numpy as np
from djitellopy import Tello
import time
from ultralytics import YOLO
from wifi_connect import WiFi

class DroneFaceTracker:
    def __init__(self, tello_network_name, model_path, fb_range, pid_x):
        # Initialize WiFi and connect to Tello network
        self.wifi = WiFi()
        self.tello_network_name = tello_network_name
        self.tello_password = None
        self.wifi.connect_to_wifi(self.tello_network_name, self.tello_password, print_output=True)

        # Initialize Tello drone
        self.drone = Tello()
        self.drone.connect()
        print(f"Drone Battery: {self.drone.get_battery()}%")

        self.drone.streamon()
        self.drone.takeoff()
        time.sleep(1)  # Allow drone to stabilize
        self.drone.send_rc_control(0, 0,70, 0) # Change the altitude here

        # Frame dimensions
        self.frame_width = 640
        self.frame_height = 480

        # Face tracking parameters
        self.fb_range = fb_range
        self.pid_x = pid_x
        self.p_error_x = 0

        # Load YOLO model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found at {model_path}")
        self.model = YOLO(model_path)

    def find_face(self, img):
        """
        Detect faces using YOLOv8 and return the largest detected face's center and area.
        """
        results = self.model.predict(source=img, conf=0.5, verbose=False)
        face_centers = []
        face_areas = []

        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

            face_centers.append([cx, cy])
            face_areas.append(area)

        if face_areas:
            i = face_areas.index(max(face_areas))
            return img, [face_centers[i], face_areas[i]]
        else:
            return img, [[0, 0], 0]
            
    def track_face(self, info):
        """
        Track the detected face and send control commands to the drone.
        """
        area = info[1]
        x, _ = info[0]
        fb = 0

        error = x - self.frame_width // 2
        speed = self.pid_x[0] * error + self.pid_x[1] * (error - self.p_error_x)
        speed = int(np.clip(speed, -100, 100))

        if self.fb_range[0] < area < self.fb_range[1]:
            fb = 0
        elif area > self.fb_range[1]:
            fb = -20
        elif area < self.fb_range[0] and area != 0:
            fb = 20

        if x == 0:
            speed = 0
            error = 0

        self.drone.send_rc_control(0, fb, 0, speed)
        self.p_error_x = error

    def display_drone_parameters(self, img):
        """
        Overlay drone parameters on the image.
        """
        parameters = [
            f"Height: {self.drone.get_height()} cm",
            f"TOF: {self.drone.get_distance_tof()} cm",
            f"Battery: {self.drone.get_battery()}%",
            f"WiFi: Connected",
            f"Temp: {self.drone.get_temperature()} C",
            f"Baro: {self.drone.get_barometer()} m",
            f"Accel X: {self.drone.get_acceleration_x()}",
            f"Accel Y: {self.drone.get_acceleration_y()}",
            f"Accel Z: {self.drone.get_acceleration_z()}"
        ]

        y_offset = 20
        for param in parameters:
            cv2.putText(img, param, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 20

    def run(self):
        """
        Main loop to handle face detection, tracking, and video streaming.
        """
        try:
            while True:
                img = self.drone.get_frame_read().frame
                img = cv2.resize(img, (self.frame_width, self.frame_height))

                img, info = self.find_face(img)
                self.track_face(info)
                self.display_drone_parameters(img)

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_enhanced = cv2.convertScaleAbs(img_rgb, alpha=1.2, beta=30)
                cv2.imshow("Drone Camera Output", img_rgb)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except ConnectionError as e:
            print(f"Error: {e}")
        finally:
            print("Landing the drone...")
            self.drone.land()
            self.drone.streamoff()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    tello_network_name = "TELLO-C3A55B"  #  Tello network name 
    model_path = "/Users/mayuripatil/code/minor_project/yolov11n-face.pt"
    fb_range = [4200, 4800]
    pid_x = [0.4, 0.4, 0]

    tracker = DroneFaceTracker(tello_network_name, model_path, fb_range, pid_x)
    tracker.run()
