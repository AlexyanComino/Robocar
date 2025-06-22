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
from queue import Queue
from threading import Thread

class CameraStreamServer:
    def __init__(self, host='0.0.0.0', port=8000, camera_index=0, on=False):
        self.host = host
        self.port = port
        self.camera_index = camera_index
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) if on else None
        self.conn = None
        self.addr = None
        self.cap = None
        self.on = on
        self.frame_queue = Queue(maxsize=1)
        self._stream_thread = None

    def __enter__(self):
        if not self.on:
            print("Camera streaming is turned off.")
            return self

        self.start()
        self._start_streaming_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
        if self._stream_thread and self._stream_thread.is_alive():
            self.frame_queue.put(None)
            self._stream_thread.join(timeout=2)

        cv2.destroyAllWindows()

    def __del__(self):
        self.__exit__(None, None, None)

    def _start_streaming_thread(self):
        def stream_loop():
            while True:
                image = self.frame_queue.get()
                if image is None:
                    break
                _, encoded_image = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                data = encoded_image.tobytes()
                size = struct.pack(">L", len(data))
                try:
                    self.conn.sendall(size + data)
                except Exception as e:
                    print(f"Streaming error: {e}")
                    break
        self._stream_thread = Thread(target=stream_loop, daemon=True)
        self._stream_thread.start()

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
        if not self.on:
            return

        if not self.frame_queue.full():
            self.frame_queue.put(image)

        # data = pickle.dumps(image)
        # size = struct.pack(">L", len(data))

        # self.conn.sendall(size + data)
