##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## camera_stream_server
##

import socket
import cv2
import pickle
import struct

class CameraStreamServer:
    def __init__(self, host='0.0.0.0', port=8000, camera_index=0):
        self.host = host
        self.port = port
        self.camera_index = camera_index
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn = None
        self.addr = None
        self.cap = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
        cv2.destroyAllWindows()

    def __del__(self):
        self.__exit__(None, None, None)

    def start(self):
        self._setup_socket()
        self._accept_connection()

    def _setup_socket(self):
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Listening on {self.host}:{self.port}")

    def _accept_connection(self):
        print("Waiting for connection...")
        self.conn, self.addr = self.server_socket.accept()
        print("Connected by", self.addr)

    def stream_image(self, image):
        data = pickle.dumps(image)
        size = struct.pack(">L", len(data))

        self.conn.sendall(size + data)


