import cv2
import numpy as np
import torch
from torchvision.ops import nms

def process_image(img, original_size, size=(640, 640)):
    """
    Preprocess image before feeding into model

    Args:
        img: input image
        original_size: original size of input image
        size: input size of model
    
    Returns:
        img: preprocessed image
        size_pad: size of padding
        flag: flag = True if image is portrait
    """

    flag = False  # flag = True if image is portrait
    # Resize image to input size
    scale = 640 / max(original_size[0], original_size[1])
    new_h, new_w = round(original_size[0] * scale), round(original_size[1] * scale)
    img = cv2.resize(img, (new_w, new_h))
    # Pad image to avoid unmatched size between input size and original image size
    if new_w == 640:
        size_pad = (640 - new_h) // 2
        img = cv2.copyMakeBorder(img, size_pad, 640 - size_pad - new_h, 0, 0, cv2.BORDER_CONSTANT, None, value = 0)
        flag = True
    else:
        size_pad = (640 - new_w) // 2
        img = cv2.copyMakeBorder(img, 0, 0, size_pad, 640 - size_pad - new_w, cv2.BORDER_CONSTANT, None, value = 0)
    # Transpose image to [B, C, H, W] format
    img = torch.from_numpy(img).permute(2, 0, 1).to('cpu')
    img = img.float() / 255.0
    return img, size_pad, flag


def process_output(raw_result, original_size, pad_size, flag, conf_thres = 0.7, input_size = (640, 640), iou_thresh=0.2):

    """
    Process output of model

    Args:
        raw_result: output of model
        original_size: original size of input image
        pad_size: size of padding
        flag: flag = True if image is portrait
        conf_thres: confidence threshold
        input_size: input size of model
        iou_thresh: IoU threshold

    Returns:
        box: list of bounding box
        score: list of score
        class_id: list of class id
    """

    # Reshape output
    result = np.squeeze(np.array(raw_result))
    # Get score of box
    score1 = np.max(result[:, 4:], axis=1)
    # Sort score in descending order
    score = np.argsort(score1)[::-1]
    # Get box with score > confidence threshold
    result = result[score > conf_thres, :]
    score = score[score > conf_thres]

    # Check if model can't detect any box
    if len(score) == 0:
        return [], [], []
    
    # Get class id of box
    class_id = np.argmax(result[:, 4:], axis=1)

    # Apply NMS
    result_box = result[:, :4]
    indices = non_max_suppression(result_box, score, iou_thresh)
    boxes, scores, class_ids = result_box[indices], score[indices], class_id[indices]

    # Rescale coordinate in box (From input size to original image size)
    if flag:
        scale_input_arr = np.array([input_size[1], input_size[0] - 2 * pad_size, input_size[1], input_size[0] - 2 * pad_size])
        scale_image_arr = np.array([original_size[1], original_size[0], original_size[1], original_size[0]])
        pad_image_arr = np.array([0, pad_size, 0, 0])
    else:
        scale_input_arr = np.array([input_size[1] - 2 * pad_size, input_size[0], input_size[1] - 2 * pad_size, input_size[0]])
        scale_image_arr = np.array([original_size[1], original_size[0], original_size[1], original_size[0]])
        pad_image_arr = np.array([pad_size, 0, 0, 0])        

    #old_box = boxes - pad_image_arr
    old_box = np.divide(old_box, scale_input_arr, dtype=np.float32)
    old_box *= scale_image_arr

    # Transform box's coordinate from [x_center, y_center, width, height] => [x1, y1, x2, y2]
    #box = old_box.copy()
    #box[..., 0] = old_box[...,0] - old_box[..., 2] / 2
    #box[..., 1] = old_box[...,1] - old_box[..., 3] / 2
    #box[..., 2] = old_box[...,0] + old_box[..., 2] / 2
    #box[..., 3] = old_box[...,1] + old_box[..., 3] / 2

    # Convert box, score to Torch tensor for nms
    #box = torch.from_numpy(box).to('cpu')
    #score = torch.from_numpy(score).to('cpu')
    
    return old_box, scores, class_ids

def non_max_suppression(boxes, scores, iou_threshold):
    """
    Apply NMS to bounding boxes

    Args:
        boxes: list of bounding box
        scores: list of score
        iou_threshold: IoU threshold
    
    Returns:
        keep_boxes: list of bounding box after NMS
    """
    # Sort boxes by score
    sorted_indices = np.argsort(scores)[::-1]
    # Initialize list of keep boxes
    keep_boxes = []
    # Apply NMS
    while sorted_indices.size > 0:
        # Pick box with highest score and add it to keep_boxes
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Keep boxes whose IoU is lower than threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # Remove boxes whose IoU is higher than threshold
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    """
    Compute IoU of a box with the rest

    Args:
        box: a bounding box
        boxes: list of bounding boxes

    Returns:
        iou: IoU of box with boxes
    """
    # Compute intersection coordinates
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou
