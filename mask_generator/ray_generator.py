##
## EPITECH PROJECT, 2025
## root [SSH: robocar-scaleway]
## File description:
## ray_distance
##

import numpy as np
import matplotlib.pyplot as plt

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


def show_rays(mask, ray_endpoints, distances, image=None, alpha=0.6, show_text=False,
              text_interval=5, colormap_name='viridis', generate_image=False):

    height, width = mask.shape
    origin_x = width // 2
    origin_y = height - 1

    plt.figure(figsize=(12, 6))

    if image is not None:
        if image.shape[:2] != mask.shape:
            raise ValueError("Image and mask must have the same height and width")
        plt.imshow(image, alpha=1.0)
        plt.imshow(mask, cmap='gray', alpha=alpha)
    else:
        plt.imshow(mask, cmap='gray')

    dist_values = np.array([distances[f"ray_{i}"] for i in range(len(ray_endpoints))])
    dist_norm = (dist_values - dist_values.min()) / (np.ptp(dist_values) + 1e-8)
    cmap = plt.get_cmap(colormap_name)

    for i, (end_x, end_y) in enumerate(ray_endpoints):

        color = cmap(dist_norm[i])

        plt.plot([origin_x, end_x], [origin_y, end_y], color=color)

        if show_text and i % text_interval == 0:
            distance = distances[f"ray_{i}"]

            mid_x = (origin_x + end_x) / 2
            mid_y = (origin_y + end_y) / 2
            offset = 20 * np.sin(i / 2.0)


            plt.text(mid_x, mid_y + offset, f"{distance}", color='white',
                    ha='center', va='center', fontsize=8)

    plt.plot(origin_x, origin_y, "ro")
    plt.title("Rays visualization with distances" if show_text else "Rays visualization")
    plt.axis("equal")

    if generate_image:
        plt.axis("off")
        plt.tight_layout()
        fig = plt.gcf()
        fig.canvas.draw()
        image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image_data
    else:
        plt.show()


if __name__ == "__main__":
    import cv2
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
