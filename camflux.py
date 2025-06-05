##
## EPITECH PROJECT, 2025
## camrobocar
## File description:
## main
##

import depthai as dai
import cv2
from mask_generator.model_inference import load_model_from_run_dir, infer_mask

pipeline = dai.Pipeline()

cam_color = pipeline.createColorCamera()
cam_color.setPreviewSize(640, 480)
cam_color.setInterleaved(False)
cam_color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam_color.preview.link(xout.input)

model, pad_divisor = load_model_from_run_dir("mask_generator/best_run")

with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    while True:
        in_video = video_queue.get()
        frame = in_video.getCvFrame()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = infer_mask(model, pad_divisor=pad_divisor, image=image)

        distances, rays = generate_rays(mask, num_rays=50, fov_degrees=120)

        if cv2.waitKey(1) == ord('q'):
            break
