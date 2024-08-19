import cv2
import numpy as np
from Utils import l1_distance, l2_distance
import os

# Create results directory for Problem 3 if it doesn't exist
results_dir = 'result-problem-3'
os.makedirs(results_dir, exist_ok=True)

def window_based_matching_l1(left_img, right_img, disparity_range, kernel_size=5, save_result=True):
    left = cv2.imread(left_img, 0).astype(np.float32)
    right = cv2.imread(right_img, 0).astype(np.float32)
    height, width = left.shape[:2]
    depth = np.zeros((height, width), np.uint8)
    kernel_half = int((kernel_size - 1) / 2)
    scale = 3
    max_value = 255 * 9

    for y in range(kernel_half, height - kernel_half):
        for x in range(kernel_half, width - kernel_half):
            disparity = 0
            cost_min = 65534
            for j in range(disparity_range):
                total = 0
                for v in range(-kernel_half, kernel_half + 1):
                    for u in range(-kernel_half, kernel_half + 1):
                        value = max_value
                        if (x + u - j) >= 0:
                            value = l1_distance(int(left[y + v, x + u]), int(right[y + v, (x + u) - j]))
                        total += value
                if total < cost_min:
                    cost_min = total
                    disparity = j
            depth[y, x] = disparity * scale

    if save_result:
        cv2.imwrite(os.path.join(results_dir, 'window_based_l1.png'), depth)
        cv2.imwrite(os.path.join(results_dir, 'window_based_l1_color.png'), cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    return depth

def window_based_matching_l2(left_img, right_img, disparity_range, kernel_size=5, save_result=True):
    left = cv2.imread(left_img, 0).astype(np.float32)
    right = cv2.imread(right_img, 0).astype(np.float32)
    height, width = left.shape[:2]
    depth = np.zeros((height, width), np.uint8)
    kernel_half = int((kernel_size - 1) / 2)
    scale = 3
    max_value = 255 ** 2

    for y in range(kernel_half, height - kernel_half):
        for x in range(kernel_half, width - kernel_half):
            disparity = 0
            cost_min = 65534
            for j in range(disparity_range):
                total = 0
                for v in range(-kernel_half, kernel_half + 1):
                    for u in range(-kernel_half, kernel_half + 1):
                        value = max_value
                        if (x + u - j) >= 0:
                            value = l2_distance(int(left[y + v, x + u]), int(right[y + v, (x + u) - j]))
                        total += value
                if total < cost_min:
                    cost_min = total
                    disparity = j
            depth[y, x] = disparity * scale

    if save_result:
        cv2.imwrite(os.path.join(results_dir, 'window_based_l2.png'), depth)
        cv2.imwrite(os.path.join(results_dir, 'window_based_l2_color.png'), cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    return depth

# Define image paths and parameters
left_img_path = 'images/aloe/Aloe/Aloe_left_1.png'
right_img_path = 'images/aloe/Aloe/Aloe_right_2.png'
disparity_range = 64
kernel_size = 5

left = cv2.imread(left_img_path)
right = cv2.imread(right_img_path)

cv2.imshow('Left Image', left)
cv2.imshow('Right Image', right)
cv2.waitKey(0)
cv2.destroyAllWindows()

# L1 Result
depth = window_based_matching_l1(left_img_path, right_img_path, disparity_range, kernel_size=kernel_size, save_result=True)
cv2.imshow('Depth L1', depth)
cv2.imshow('Depth L1 Color', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
cv2.waitKey(0)
cv2.destroyAllWindows()

# L2 Result
depth = window_based_matching_l2(left_img_path, right_img_path, disparity_range, kernel_size=kernel_size, save_result=True)
cv2.imshow('Depth L2', depth)
cv2.imshow('Depth L2 Color', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
cv2.waitKey(0)
cv2.destroyAllWindows()
