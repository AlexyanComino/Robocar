##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## camera
##

import depthai as dai

class Camera:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cam_device = None
        self.video_queue = None
        self.pipeline = self.init_camera()

    def __del__(self):
        if self.video_queue:
            self.video_queue = None

    def __enter__(self):
        self.cam_device = dai.Device(self.pipeline).__enter__()
        self.video_queue = self.cam_device.getOutputQueue(name="video", maxSize=4, blocking=False)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cam_device.__exit__(exc_type, exc_value, traceback)
        self.__del__()


    def init_camera(self, width=783, height=256):
        print("Résolutions supportées par la caméra :")
        for res in dai.ColorCameraProperties.SensorResolution:
            print("-", res.name)

        pipeline = dai.Pipeline()
        cam_color = pipeline.createColorCamera()
        cam_color.setPreviewSize(width, height)
        cam_color.setInterleaved(False)
        cam_color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        xout = pipeline.createXLinkOut()
        xout.setStreamName("video")
        cam_color.preview.link(xout.input)

        return pipeline

    def get_frame(self):
        if not self.video_queue:
            raise RuntimeError("Camera not initialized. Use 'with Camera() as cam:' to initialize.")

        latest_frame = None
        while True:
            frame = self.video_queue.tryGet()
            if frame is None:
                break
            latest_frame = frame

        if latest_frame is None:
            latest_frame = self.video_queue.get()

        return latest_frame.getCvFrame()
