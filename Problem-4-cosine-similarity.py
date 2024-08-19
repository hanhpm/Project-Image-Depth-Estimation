import cv2
import numpy as np
from Utils import cosine_similarity
import os

# Create results directory for Problem 4 if it doesn't exist
results_dir = 'results-problem-4'
os.makedirs(results_dir, exist_ok=True)

def window_based_matching(left_img, right_img, disparity_range, kernel_size=5, save_result=True):
    left = cv2.imread(left_img, 0).astype(np.float32)
    right = cv2.imread(right_img, 0).astype(np.float32)
    height, width = left.shape[:2]
    depth = np.zeros((height, width), np.uint8)
    kernel_half = int((kernel_size - 1) / 2)
    scale = 3

    for y in range(kernel_half, height - kernel_half):
        for x in range(kernel_half, width - kernel_half):
            disparity = 0
            cost_optimal = -1
            for j in range(disparity_range):
                d = x - j
                cost = -1
                if (d - kernel_half) > 0:
                    wp = left[(y - kernel_half):(y + kernel_half + 1), (x - kernel_half):(x + kernel_half + 1)]
                    wqd = right[(y - kernel_half):(y + kernel_half + 1), (d - kernel_half):(d + kernel_half + 1)]

                    wp_flattened = wp.flatten()
                    wqd_flattened = wqd.flatten()

                    cost = cosine_similarity(wp_flattened, wqd_flattened)

                if cost > cost_optimal:
                    cost_optimal = cost
                    disparity = j

            depth[y, x] = disparity * scale

    if save_result:
        print('Saving result...')
        cv2.imwrite(os.path.join(results_dir, 'window_based_cosine_similarity.png'), depth)
        cv2.imwrite(os.path.join(results_dir, 'window_based_cosine_similarity_color.png'), cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    return depth

# Define image paths and parameters
left_img_path = 'images/aloe/Aloe/Aloe_left_1.png'
right_img_path = 'images/aloe/Aloe/Aloe_right_3.png'
disparity_range = 64
kernel_size = 5

# Display and process results
left = cv2.imread(left_img_path)
right = cv2.imread(right_img_path)

cv2.imshow('Left Image', left)
cv2.imshow('Right Image', right)
cv2.waitKey(0)
cv2.destroyAllWindows()

depth = window_based_matching(
    left_img_path,
    right_img_path,
    disparity_range,
    kernel_size=kernel_size,
    save_result=True
)

cv2.imshow('Depth Cosine Similarity', depth)
cv2.imshow('Depth Cosine Similarity Color', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
cv2.waitKey(0)
cv2.destroyAllWindows()
