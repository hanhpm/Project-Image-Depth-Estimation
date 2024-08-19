import cv2
import numpy as np
import os
from Utils import l1_distance, l2_distance

# Create results directory if it doesn't exist
results_dir = 'results-problem-1'
os.makedirs(results_dir, exist_ok=True)

def pixel_wise_matching_l1(left_img, right_img, disparity_range, save_result=True):
    left = cv2.imread(left_img, 0).astype(np.float32)
    right = cv2.imread(right_img, 0).astype(np.float32)
    height, width = left.shape[:2]
    depth = np.zeros((height, width), np.uint8)
    scale = 16
    max_value = 255

    for y in range(height):
        for x in range(width):
            disparity = 0
            cost_min = max_value
            for j in range(disparity_range):
                cost = max_value if (x - j) < 0 else l1_distance(int(left[y, x]), int(right[y, x - j]))
                if cost < cost_min:
                    cost_min = cost
                    disparity = j
            depth[y, x] = disparity * scale

    if save_result:
        print('Saving L1 result...')
        cv2.imwrite(os.path.join(results_dir, 'pixel_wise_l1.png'), depth)
        cv2.imwrite(os.path.join(results_dir, 'pixel_wise_l1_color.png'), cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    return depth, cv2.applyColorMap(depth, cv2.COLORMAP_JET)

def pixel_wise_matching_l2(left_img, right_img, disparity_range, save_result=True):
    left = cv2.imread(left_img, 0).astype(np.float32)
    right = cv2.imread(right_img, 0).astype(np.float32)
    height, width = left.shape[:2]
    depth = np.zeros((height, width), np.uint8)
    scale = 16
    max_value = 255 ** 2

    for y in range(height):
        for x in range(width):
            disparity = 0
            cost_min = max_value
            for j in range(disparity_range):
                cost = max_value if (x - j) < 0 else l2_distance(int(left[y, x]), int(right[y, x - j]))
                if cost < cost_min:
                    cost_min = cost
                    disparity = j
            depth[y, x] = disparity * scale

    if save_result:
        print('Saving L2 result...')
        cv2.imwrite(os.path.join(results_dir, 'pixel_wise_l2.png'), depth)
        cv2.imwrite(os.path.join(results_dir, 'pixel_wise_l2_color.png'), cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    return depth

left_img_path = 'images/tsukuba/left.png'
right_img_path = 'images/tsukuba/right.png'
disparity_range = 16

# L1 Result
depth, color = pixel_wise_matching_l1(
    left_img_path,
    right_img_path,
    disparity_range,
    save_result=True
)

# Display images
cv2.imshow('Depth L1', depth)
cv2.imshow('Depth L1 Color', color)
cv2.waitKey(0)
cv2.destroyAllWindows()

# L2 Result
depth = pixel_wise_matching_l2(
    left_img_path,
    right_img_path,
    disparity_range,
    save_result=True
)

# Display images
cv2.imshow('Depth L2', depth)
cv2.imshow('Depth L2 Color', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
cv2.waitKey(0)
cv2.destroyAllWindows()
