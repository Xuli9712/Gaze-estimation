from datetime import timedelta
import cv2
import random
import pyautogui
import numpy as np
from datetime import datetime
import math
import csv

def get_monitor_dimensions():
    screenWidthPixel, screenHeightPixel = pyautogui.size()
    print(f"Screen size: {screenWidthPixel}x{screenHeightPixel}")
    return (screenWidthPixel, screenHeightPixel)

def create_image(monitor_pixels, center, circle_scale, inner_radius):
    width, height = monitor_pixels
    img = np.zeros((height, width, 3), np.float32)
    radius = int(circle_scale)
    cv2.circle(img, center, radius, (1, 1, 1), -1)  # Fill the circle with white color
    cv2.circle(img, center, 5, (0, 1, 0), -1)  # Mark the center with a different color (green in this case)
    end_animation_loop = circle_scale <= inner_radius  # Stop when the circle is smaller than 20 pixels
    return img, end_animation_loop

def generate_points(monitor_pixels, num_points):
    width, height = monitor_pixels
    num_points_per_row = int(math.sqrt(num_points))
    num_points_per_column = num_points // num_points_per_row
    grid_x = width // num_points_per_row
    grid_y = height // num_points_per_column

    points = [(i*grid_x + grid_x//2, j*grid_y + grid_y//2) for i in range(num_points_per_row) for j in range(num_points_per_column)]
    points.sort(key=lambda x: (x[1], x[0]))  # Sort the points from top-left to bottom-right

    # Get the first and last points, and remove them from the list
    first_point = points.pop(0)
    last_point = max(points, key=lambda x: (x[0], -x[1]))
    points.remove(last_point)

    # Shuffle the remaining points
    random.seed(1)  # Set seed for reproducibility
    random.shuffle(points)

    # Add the first and last points back to their respective positions
    points.insert(0, first_point)
    points.append(last_point)

    return points

def visualize_points(monitor_pixels, points):
    width, height = monitor_pixels
    img = np.zeros((height, width, 3), np.float32)
    for point in points:
        cv2.circle(img, point, 5, (0, 1, 0), -1)
    cv2.imwrite('points_visualization.png', img * 255)  # Save the image

def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param[0] = True  # Set 'start_animation' to True when mouse is clicked

from datetime import timedelta

def show_point_on_screen(window_name: str, outer_radius: float, inner_radius: float, display_time: float):
    monitor_pixels = get_monitor_dimensions()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    start_animation = [False]
    points = generate_points(monitor_pixels, 60)

    visualize_points(monitor_pixels, points)

    cv2.setMouseCallback(window_name, on_mouse_click, param=start_animation)

    circle_reduction_per_frame = (outer_radius - inner_radius) / (display_time * 10)  # Keep the current frame rate

    # 创建CSV文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'circle_data_{timestamp}.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Circle Center", "Start Time", "End Time"]) # 写入标题行

        while True:
            if start_animation[0]:
                for center in points:
                    circle_scale = outer_radius
                    end_animation_loop = False
                    start_time = datetime.now()
                    print(f"Circle center: {center}, start at {start_time}", end=' ')
                    while not end_animation_loop:
                        image, end_animation_loop = create_image(monitor_pixels, center, circle_scale, inner_radius)
                        cv2.imshow(window_name, image)
                        circle_scale -= circle_reduction_per_frame
                        if cv2.waitKey(100) & 0xFF == 27: # 100ms per frame
                            cv2.destroyAllWindows()
                            return

                    end_time = datetime.now()
                    print(f"and end at {end_time}, last {end_time-start_time}")

                    # 将中心点、起始时间和结束时间写入CSV
                    csv_writer.writerow([f"{center[0]},{center[1]}", start_time, end_time])

                cv2.destroyAllWindows()
                return

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 输入参数为 外径，每个圆持续时间(s)
    show_point_on_screen("Display", 40, 10, 2.06)
