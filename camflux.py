##
## EPITECH PROJECT, 2025
## camrobocar
## File description:
## main
##

import time
import depthai as dai
import cv2
from collections import deque
from mask_generator.model_loader import load_model_from_run_dir
from mask_generator.model_inference import infer_mask
from mask_generator.transforms import EvalTransform, TensorDecoder
from mask_generator.ray_generator import generate_rays
import matplotlib.pyplot as plt
import numpy as np
import torch

pipeline = dai.Pipeline()
cam_color = pipeline.createColorCamera()
cam_color.setPreviewSize(455, 256)

cam_color.setInterleaved(False)
cam_color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam_color.preview.link(xout.input)

model, pad_divisor = load_model_from_run_dir("mask_generator/best_run")
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = EvalTransform(pad_divisor=pad_divisor, to_tensor=True)
decoder = TensorDecoder()

# plt.ion()
# fig, ax = plt.subplots(figsize=(12, 6))

with dai.Device(pipeline) as cam_device:
    video_queue = cam_device.getOutputQueue(name="video", maxSize=4, blocking=False)

    fps_history = deque(maxlen=30)

    prev_time = time.time()

    while True:
        now = time.time()
        delta = now - prev_time
        prev_time = now
        if delta > 0:
            fps_history.append(1.0 / delta)

        avg_fps = sum(fps_history) / len(fps_history)
        print(f"\rAverage FPS: {avg_fps:.2f}  ")
        in_video = video_queue.get()
        frame = in_video.getCvFrame()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = infer_mask(model, transform=transform, decoder=decoder, image=image_rgb, device=device)
        distances, rays = generate_rays(mask, num_rays=50, fov_degrees=120)

        # ax.clear()

        # ax.imshow(image_rgb, alpha=1.0)
        # ax.imshow(mask, cmap='gray', alpha=0.6)

        # height, width = mask.shape
        # origin_x = width // 2
        # origin_y = height - 1

        # dist_values = np.array([distances[f"ray_{i}"] for i in range(len(rays))])
        # dist_norm = (dist_values - dist_values.min()) / (np.ptp(dist_values) + 1e-8)
        # cmap = plt.get_cmap('viridis')

        # for i, (end_x, end_y) in enumerate(rays):
        #     color = cmap(dist_norm[i])
        #     ax.plot([origin_x, end_x], [origin_y, end_y], color=color, linewidth=1)

        #     ax.plot(origin_x, origin_y, 'ro')  # Origine
        # ax.axis('equal')
        # ax.axis('off')
        # ax.set_title("Mask + Rays")

        # # Affichage non-bloquant
        # plt.pause(0.001)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

# plt.ioff()
# plt.close()