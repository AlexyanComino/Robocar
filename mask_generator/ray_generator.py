##
## EPITECH PROJECT, 2025
## root [SSH: robocar-scaleway]
## File description:
## ray_distance
##

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import math

# -------------------------------------------
# Ray Tracing Core
# -------------------------------------------

# NumPy 2.x compatibility patch
def ensure_numpy_array(data):
    """Ensure data is a properly formatted NumPy array for ray calculations"""
    if isinstance(data, np.ndarray):
        # Make sure we're working with a 2D array
        if data.ndim == 2:
            return data
        elif data.ndim > 2:
            return data.squeeze()
    return np.array(data)

def generate_rays(mask, num_rays=50, fov_degrees=160):
    height, width = mask.shape[:2]
    center_x = width / 2
    start_angle_rad = math.radians(90 + fov_degrees / 2)
    angle_step = math.radians(-fov_degrees / (num_rays - 1))
    step_size = 1

    distances = {}
    hits = []

    for k in range(num_rays):
        angle = start_angle_rad + k * angle_step
        x = center_x
        y = height - 1
        hit_dist = 0

        max_distance = _get_max_raycast_distance(width, height, angle, step_size)

        while 0 <= int(x) < width and 0 <= int(y) < height:
            px = int(x)
            py = int(y)

            if mask[py, px] > 0.9:
                break

            x += step_size * math.cos(angle)
            y -= step_size * math.sin(angle)
            hit_dist += 1

        distances[f"ray_{k}"] = hit_dist / max_distance if max_distance != 0 else 1.0
        hits.append((int(x), int(y)))

    return distances, hits


def _get_max_raycast_distance(width, height, angle, step_size):
    x = width / 2
    y = height - 1
    max_dist = 0
    while 0 <= x < width and 0 <= y < height:
        x += step_size * math.cos(angle)
        y -= step_size * math.sin(angle)
        max_dist += 1
    return max_dist

def get_max_distance(mask, angle: float, step_size: float):
    """
    Calculate the maximum distance a ray can travel in a given direction before leaving the mask bounds.
    Starts from the bottom center of the mask.
    """
    height, width = mask.shape

    x = width / 2.0
    y = height - 1
    hit_dist = 0

    while 0 <= x < width and 0 <= y < height:
        x += step_size * np.cos(angle)
        y -= step_size * np.sin(angle)
        hit_dist += 1

    return hit_dist

def generate_rays_vectorized(mask, num_rays=50, fov_degrees=120):
    """
    Vectorized version of ray generation for improved performance.
    """
    mask = ensure_numpy_array(mask)
    height, width = mask.shape
    origin_x = width // 2
    origin_y = height - 1

    max_distance = int(np.sqrt(np.power(origin_x, 2) + np.power(height, 2)))

    # Prepare angle array
    angles = np.linspace(-fov_degrees / 2, fov_degrees / 2, num_rays)
    angles_rad = np.radians(angles)

    # Compute direction vectors for all rays
    dx = np.sin(angles_rad)  # shape: (num_rays,)
    dy = -np.cos(angles_rad) # shape: (num_rays,)

    # Prepare all distances for each ray (broadcasted)
    dists = np.arange(1, max_distance + 1).reshape(-1, 1)

    # Compute all x, y positions for each step of each ray
    x = (origin_x + dx * dists).astype(np.int32)  # shape: (max_distance, num_rays)
    y = (origin_y + dy * dists).astype(np.int32)  # shape: (max_distance, num_rays)

    # Mask bounds
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)

    # Sample mask values
    mask_vals = np.zeros_like(valid, dtype=bool)
    mask_vals[valid] = mask[y[valid], x[valid]] > 0  # obstacle = white pixel

    # Find first hit (axis=0 = distance steps)
    hit_indices = mask_vals.argmax(axis=0)
    hit_mask = mask_vals.any(axis=0)

    distances = {}
    ray_endpoints = []

    for i in range(num_rays):

        max_distance = get_max_distance(mask, angles_rad[i], 1.0)

        if hit_mask[i]:
            dist = hit_indices[i] + 1  # +1 since range starts from 1
            end_x = x[hit_indices[i], i]
            end_y = y[hit_indices[i], i]
        else:
            dist = max_distance
            end_x = x[-1, i]
            end_y = y[-1, i]

        dist /= max_distance
        distances[f"ray_{i}"] = dist
        ray_endpoints.append((end_x, end_y))

    return distances, ray_endpoints

def generate_rays_torch(mask_tensor: torch.Tensor, num_rays=50, fov_degrees=120, max_distance=400, device='cuda'):
    assert isinstance(mask_tensor, torch.Tensor), "mask_tensor must be a torch.Tensor"
    assert mask_tensor.ndim == 2, "mask_tensor must be a 2D tensor"
    assert mask_tensor.device.type == device, f"mask_tensor must be on {device} device"

    height, width = mask_tensor.shape
    origin_x = width // 2
    origin_y = height - 1

    angles = torch.linspace(-fov_degrees/2, fov_degrees/2, steps=num_rays, device=device)
    angles_rad = torch.deg2rad(angles)

    dx = torch.sin(angles_rad)
    dy = -torch.cos(angles_rad)

    dists = torch.arange(1, max_distance+1, device=device).unsqueeze(1)  # (D, 1)

    x = (origin_x + dx * dists).long()  # (D, R)
    y = (origin_y + dy * dists).long()

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)

    mask_vals = torch.zeros_like(valid, dtype=torch.bool)
    mask_vals[valid] = mask_tensor[y[valid], x[valid]] > 0.0

    hit_indices = torch.argmax(mask_vals.int(), dim=0)
    hit_mask = mask_vals.any(dim=0)

    ray_endpoints = []
    distances = {}

    for i in range(num_rays):
        if hit_mask[i]:
            dist = hit_indices[i].item() + 1
            ex = x[hit_indices[i], i].item()
            ey = y[hit_indices[i], i].item()
        else:
            dist = max_distance
            ex = x[-1, i].item()
            ey = y[-1, i].item()

        distances[f"ray_{i}"] = dist
        ray_endpoints.append((ex, ey))

    return distances, ray_endpoints

def show_rays(mask, ray_endpoints, distances, image=None, alpha=0.6, show_text=False,
              text_interval=5, colormap_name='viridis', generate_image=False):

    height, width = mask.shape
    origin_x = width // 2
    origin_y = height - 1

    # Prepare base image
    if image is not None:
        if image.shape[:2] != mask.shape:
            image = cv2.resize(image, (width, height))
        base = image.copy()
    else:
        base = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Overlay mask with alpha if image is provided
    if image is not None and alpha < 1.0:
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        base = cv2.addWeighted(base, 1 - alpha, mask_colored, alpha, 0)

    # Normalize distances for color mapping
    dist_values = np.array([distances[f"ray_{i}"] for i in range(len(ray_endpoints))])
    dist_norm = (dist_values - dist_values.min()) / (np.ptp(dist_values) + 1e-8)
    cmap = plt.get_cmap(colormap_name)
    colors = (np.array([cmap(val)[:3] for val in dist_norm]) * 255).astype(np.uint8)

    # Draw rays
    for i, (end_x, end_y) in enumerate(ray_endpoints):
        color = tuple(int(c) for c in colors[i])
        cv2.line(base, (origin_x, origin_y), (int(end_x), int(end_y)), color, 1, cv2.LINE_AA)

        if show_text and i % text_interval == 0:
            distance = distances[f"ray_{i}"]
            mid_x = int((origin_x + end_x) / 2)
            mid_y = int((origin_y + end_y) / 2)
            offset = int(20 * np.sin(i / 2.0))
            text_pos = (mid_x, mid_y + offset)
            cv2.putText(base, f"{distance}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

    # Draw origin
    cv2.circle(base, (origin_x, origin_y), 4, (0,0,255), -1)

    if generate_image:
        return base
    else:
        cv2.imshow("Rays visualization", base)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Create two lines
    cv2.line(mask, (10, 90), (10, 10), 255, 2)
    cv2.line(mask, (90, 10), (90, 90), 255, 2)

    distances, ray_endpoints = generate_rays(mask, num_rays=50, fov_degrees=120, max_distance=80)

    # Show the rays
    image = show_rays(mask, ray_endpoints, distances, generate_image=True)

    if image is not None:
        cv2.imshow("Rays Visualization", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
