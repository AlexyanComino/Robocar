##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## stream
##

import argparse
import cv2
import socket
import pickle
import struct

class CameraStreamViewer:
    def __init__(self, host='172.20.10.14', port=8000, scale=1.0):
        self.host = host
        self.port = port
        self.scale = scale
        self.client_socket = None
        self.data = b""
        self.payload_size = struct.calcsize(">L")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None

        cv2.destroyAllWindows()

    def start(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.settimeout(3)
        self.client_socket.connect((self.host, self.port))
        self.client_socket.settimeout(None)
        print(f"Connected to {self.host}:{self.port}")

    def stream(self):
        while True:
            while len(self.data) < self.payload_size:
                packet = self.client_socket.recv(4096)
                if not packet:
                    break
                self.data += packet

            packed_msg_size = self.data[:self.payload_size]
            self.data = self.data[self.payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]

            while len(self.data) < msg_size:
                self.data += self.client_socket.recv(4096)

            frame_data = self.data[:msg_size]
            self.data = self.data[msg_size:]

            frame = pickle.loads(frame_data)

            if self.scale != 1.0:
                frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)

            cv2.imshow("Camera Feed", frame)
            if cv2.waitKey(1) == ord('q'):
                break

    def run(self):
        with self:
            self.stream()


def parse_args():
    parser = argparse.ArgumentParser(description="Camera Stream Viewer")
    parser.add_argument('--scale', type=float, default=1.0, help="Scale factor for the video feed")
    return parser.parse_args()

def main():
    args = parse_args()

    ip_matys = "192.168.178.192"
    ip_alex = "172.20.10.14"
    ip_ambre = "172.20.10.2"

    ips = [ip_matys, ip_alex, ip_ambre]

    for ip in ips:
        try:
            print(f"Trying to connect to {ip}...")
            with CameraStreamViewer(host=ip, port=8000, scale=args.scale) as viewer:
                print(f"Successfully connected to {ip} and streaming.")
                viewer.stream()
            break
        except (socket.error, pickle.UnpicklingError) as e:
            print(f"Failed to connect to {ip}: {e}")
            continue


if __name__ == "__main__":
    main()
