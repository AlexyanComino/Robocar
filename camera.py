##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## camera
##

import depthai as dai

RESOLUTION_SIZES = {
    "THE_400_P": (640, 400),
    "THE_480_P": (640, 480),
    "THE_576_P": (720, 576),
    "THE_720_P": (1280, 720),
    "THE_800_P": (1280, 800),
    "THE_864_P": (1280, 864),
    "THE_960_P": (1280, 960),
    "THE_1080_P": (1920, 1080),
    "THE_1200_P": (1920, 1200),
    "THE_1440_P": (1920, 1440),
    "THE_4_K": (3840, 2160),
    "THE_5_MP": (2592, 1944),
    "THE_12_MP": (4056, 3040),
    "THE_13_MP": (4208, 3120)
}

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
        for res in dai.ColorCameraProperties.SensorResolution.__members__.values():
            name = res.name
            size = RESOLUTION_SIZES.get(name, ("Unknown", "Unknown"))
            print(f"- {name}: {size[0]}x{size[1]}")

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
