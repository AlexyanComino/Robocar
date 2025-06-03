##
## EPITECH PROJECT, 2025
## camrobocar
## File description:
## main
##

import depthai as dai
import cv2

pipeline = dai.Pipeline()

cam_color = pipeline.createColorCamera()
cam_color.setPreviewSize(640, 480)
cam_color.setInterleaved(False)
cam_color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam_color.preview.link(xout.input)

with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    while True:
        in_video = video_queue.get()
        frame = in_video.getCvFrame()

        cv2.imshow("Color Camera", frame)

        if cv2.waitKey(1) == ord('q'):
            break
