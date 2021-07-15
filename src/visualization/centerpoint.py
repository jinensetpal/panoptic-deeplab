import cv2
import pprint
import numpy as np

def avg_pos(p, np):
    return [(p[2] * p[0] + np[0]) / (p[2] + 1), (p[1] * p[2] + np[1]) / (p[2] + 1), p[2] + 1]

def get_center_targets(img, sigma=8):
    l = {}
    for i in range(img.shape[0]):
        for enum, j in enumerate(img[i][:]):
            if j > 9999: # 5-digit keys are 'things'
                if str(j) not in l:
                    l[str(j)] = [i, enum, 1]
                else:
                    l[str(j)] = avg_pos(l[str(j)], [i, enum])


    x = np.arange(0, 6 * sigma + 3, 1, float)
    y = x[:, np.newaxis]
    x0, y0 = 3 * sigma + 1, 3 * sigma + 1
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    height, width = img.shape[0], img.shape[1]
    heatmap = np.zeros((height, width), dtype=np.float32)
    center = np.zeros((height, width), dtype=np.float32)

    for i in l:
        center[round(l[i][0]), round(l[i][1])] = 1
        
        y, x = int(round(l[i][0])), int(round(l[i][1]))
        ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
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
    return (center, heatmap)
