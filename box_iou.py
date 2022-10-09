import math
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def return_box_iou_batch(com_1, com_2, threshold=0.45):
    t3 = torch.tensor(com_2).reshape(-1, 4)
    return_string = []
    redundant_idx = []
    for i, ele in enumerate(com_1):
        t1 = torch.tensor(ele).reshape(-1, 4)
        iou = bbox_iou(t1, t3, xywh=False)
        # print('iou : {}'.format(iou))
        iou_value = iou.reshape(1, -1)
        if iou_value.nelement():
            max_ele, max_id = torch.max(iou_value, 1)
            if max_ele.item() > threshold:
                redundant_idx.append(max_id.item())

            return_string.append(np.array(iou_value)[0])
            # print('box {} : {}'.format(i + 1, np.array(iou_value)[0]))

    string = []
    for ele in np.array(return_string):
        string.append(',  '.join([str(x) for x in ele]))

    redundant_idx = (list(set(redundant_idx)))
    return string, redundant_idx


def main():
    com_norm_1 = [(0.5863095238095238, 0.34227848101265823, 0.6537698412698413, 0.4830379746835443),
                  (0.7771164021164021, 0.21772151898734177, 0.8988095238095238, 0.4020253164556962),
                  (0.7056878306878307, 0.31291139240506327, 0.7956349206349206, 0.4506329113924051)]
    com2_remapped_norm = [(0.7003968253968254, 0.2440506329113924, 0.8306878306878307, 0.44759493670886075),
                          (0.8528439153439153, 0.035443037974683546, 1.0307539682539681, 0.35645569620253165),
                          (0.8141534391534392, 0.21164556962025316, 0.876984126984127, 0.2643037974683544)]

    t3 = torch.tensor(com2_remapped_norm).reshape(-1, 4)

    # time1 = time.perf_counter()
    for i, ele in enumerate(com_norm_1):
        t1 = torch.tensor(ele).reshape(-1, 4)
        iou = bbox_iou(t1, t3, xywh=False)
        iou_value = iou.reshape(1, -1)
        print('box {} : {}'.format(i + 1, iou_value))
        # print(torch.gt(iou_value, 0.4))
        # print(np.array(iou_value))
        # if iou_value > 0.4:
        #     print('iou for box {} and box {} : {}'.format(i + 1, j + 1, iou))

        # time1 = time.perf_counter()
        # for i, ele in enumerate(com_norm_1):
        #     for j, ele2 in enumerate(com2_remapped_norm):
        #         t1 = torch.tensor(ele).reshape(-1, 4)
        #         t2 = torch.tensor(ele2).reshape(-1, 4)
        #         iou = bbox_iou(t1, t2, xywh=False)
        #         iou_value = iou.item()
        #         # if iou_value > 0.4:
        #         print('iou for box {} and box {} : {}'.format(i + 1, j + 1, iou))
        # print((time.perf_counter() - time1))

    print(return_box_iou_batch(com_norm_1, com2_remapped_norm)[0])


if __name__ == '__main__':
    main()
