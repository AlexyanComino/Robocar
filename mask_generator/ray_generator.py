##
## EPITECH PROJECT, 2025
## root [SSH: robocar-scaleway]
## File description:
## ray_distance
##

import numpy as np
import matplotlib.pyplot as plt
import cv2

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

def generate_rays(mask, num_rays=50, fov_degrees=120, max_distance=None):
    """
    Generate rays from the bottom center of the mask and calculate distances to obstacles.
    Compatible with NumPy 2.x
    """
    # Ensure mask is properly formatted
    mask = ensure_numpy_array(mask)

    # Rest of the function remains the same
    height, width = mask.shape
    origin_x = width // 2
    origin_y = height - 1

    # Calculate the angle range
    fov_radians = np.radians(fov_degrees)
    half_fov = fov_radians / 2

    # Calculate the angles for each ray
    angles = np.linspace(-half_fov, half_fov, num_rays)

    ray_endpoints = []
    distances = {}

    if max_distance is None:
        max_distance = int(np.sqrt(width**2 + height**2))

    # Cast rays and find intersections
    for i, angle in enumerate(angles):
        # Direction vector
        dx = np.sin(angle)
        dy = -np.cos(angle)  # Negative because y-axis is inverted in images

        # Ray tracing
        found_obstacle = False
        for dist in range(1, max_distance):
            x = int(origin_x + dx * dist)
            y = int(origin_y + dy * dist)

            # Check if we're out of bounds
            if x < 0 or x >= width or y < 0 or y >= height:
                ray_endpoints.append((x, y))
                distances[f"ray_{i}"] = dist
                found_obstacle = True
                break

            # Check if we hit an obstacle (white pixel)
            if mask[y, x] > 0:
                ray_endpoints.append((x, y))
                distances[f"ray_{i}"] = dist
                found_obstacle = True
                break

        # If no obstacle was found, add the maximum distance
        if not found_obstacle:
            x = int(origin_x + dx * max_distance)
            y = int(origin_y + dy * max_distance)
            ray_endpoints.append((x, y))
            distances[f"ray_{i}"] = max_distance

    return distances, ray_endpoints

def generate_rays_vectorized(mask, num_rays=50, fov_degrees=120, max_distance=None):
    """
    Vectorized version of ray generation for improved performance.
    """
    mask = ensure_numpy_array(mask)
    height, width = mask.shape
    origin_x = width // 2
    origin_y = height - 1

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
        if hit_mask[i]:
            dist = hit_indices[i] + 1  # +1 since range starts from 1
            end_x = x[hit_indices[i], i]
            end_y = y[hit_indices[i], i]
        else:
            dist = max_distance
            end_x = x[-1, i]
            end_y = y[-1, i]
        distances[f"ray_{i}"] = dist
        ray_endpoints.append((end_x, end_y))

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
