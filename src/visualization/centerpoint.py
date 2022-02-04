from src import const
import numpy as np
import cv2

def to_file(center, heatmap, filename): 
    """ writes files to storage; testing
    Args:
      center: generated groundtruth centerpoint image
      heatmap: generated groundtruth heatmap image
    """

    # scaling encoding so the difference is visually apparent
    cv2.imwrite(filename + '-center.png', np.where(center == 1, 65000, center))
    cv2.imwrite(filename + '-heatmap.png', heatmap * 1000)

def avg_pos(position, new_position):
    """ updates the weighted average position to a center of mass
    Args:
      position: contains 3 elements, x-average, y-average and the pixels iterated upon
      new_position: contains x and y coordinates of an instance, to be added to the center of mass
        
    Returns:
      A list containing updated average coordinates, incrementing the iterated pixel count
    """
    return [(position[2] * position[0] + new_position[0]) / (position[2] + 1), (position[1] * position[2] + new_position[1]) / (position[2] + 1), position[2] + 1]

def get_center_targets(img, 
                       sigma=8,
                       path=None):
    """ generates targets for center heatmap
    Args:
      img: image input for which centerpoint and heatmap are derived; input numpy numpy array using: cv2.imread('filename.type', cv2.IMREAD_UNCHANGED)
      sigma: standard deviation of the gaussian heatmap from the centerpoint
      path: write location to test output. None by default
        
    Returns:
      A tuple containing groundtruth instance centerpoints and gaussian heatmaps
    """
    center_points = {}
    for length in range(img.shape[0]):
        for width, key in enumerate(img[length][:]): # 'key' here represents encodings on the label map
            if key > 9999: # 5-digit keys are 'things'
                if key not in center_points:
                    center_points[key] = [length, width, 1]
                else:
                    center_points[key] = avg_pos(center_points[key], [length, width])

    ## center points computed
    # import pprint
    # pprint.pprint(center_points) # debug

    ### gaussian heatmap - from detectron2/projects/Panoptic-DeepLab/panoptic_deeplab/target_generator.py 
    x = np.arange(0, 6 * sigma + 3, 1, float)
    y = x[:, np.newaxis]
    x0, y0 = 3 * sigma + 1, 3 * sigma + 1
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    height, width = img.shape[0], img.shape[1]
    heatmap = np.zeros((height, width), dtype=np.float32)
    center = np.zeros((height, width), dtype=np.float32)

    for key in center_points:
        center[round(center_points[key][0]), round(center_points[key][1])] = 1
        
        # generate center heatmap
        y, x = int(round(center_points[key][0])), int(round(center_points[key][1]))
        # upper left
        ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
        # bottom right
        br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

        # start and end indices in default Gaussian image
        gaussian_x0, gaussian_x1 = max(0, -ul[0]), min(br[0], width) - ul[0]
        gaussian_y0, gaussian_y1 = max(0, -ul[1]), min(br[1], height) - ul[1]

        # start and end indices in center heatmap image
        center_x0, center_x1 = max(0, ul[0]), min(br[0], width)
        center_y0, center_y1 = max(0, ul[1]), min(br[1], height)
        heatmap[center_y0:center_y1, center_x0:center_x1] = np.maximum(
            heatmap[center_y0:center_y1, center_x0:center_x1],
            g[gaussian_y0:gaussian_y1, gaussian_x0:gaussian_x1],
        )
        
    if path:
        to_file(center, heatmap, path)
    return {const.GT_KEY_INSTANCE_CENTER: center, 
            const.GT_KEY_CENTER_REGRESSION: heatmap}
