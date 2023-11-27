import cv2
from support_function import *
import time
import torch
import os 
from utils.general import non_max_suppression

# Write video from /static/images/videoname
image_name = '/home/hasybach/Downloads/Head_detection_dataset/head_detection_web/static/images/0.mp4/0.jpg'
IMAGE_SIZE = 640

def image_detect(img, model, videoname, id_frame):
    """
    Detect head in image
    Draw bounding box, fps (in northwest corner of img) and return number of head in image

    Args:
        img: input image
        model: model

    Returns:
        num_head: number of head in image
    """

    orginal_img = img.copy()
    ori_h, ori_w = orginal_img.shape[:2]
    input_size = 640

    # Preprocess image
    t1 = time.time()
    img, pad_size, flag = process_image(img, img.shape[:2], size=(IMAGE_SIZE, IMAGE_SIZE))
    t2 = time.time()
    print(pad_size, flag)
    # Predict
    with torch.no_grad():
        pred = model(img[None])[0]
    t3 = time.time()


    #boxes, _, _ = process_output(pred, orginal_size, pad_size, flag, conf_thres=0.001, input_size=(IMAGE_SIZE, IMAGE_SIZE), iou_thresh=0.65)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.25)[0].cpu().numpy()
    t4 = time.time()
    print(pred[0, :4])
    # Rescale coordinate in box (From input size to original image size)
    if flag:
        scale_input_arr = np.array([input_size, input_size - 2 * pad_size, input_size, input_size - 2 * pad_size])
        scale_image_arr = np.array([ori_w, ori_h, ori_w, ori_h])
        pad_image_arr = np.array([0, pad_size, 0, pad_size])

        pred[:, :4] = pred[:, :4] - pad_image_arr

        pred[:, :4] = np.divide(pred[:, :4], scale_input_arr, dtype=np.float32)
        pred[:, :4] *= scale_image_arr
    else:
        scale_input_arr = np.array([input_size - 2 * pad_size, input_size, input_size - 2 * pad_size, input_size])
        scale_image_arr = np.array([ori_w, ori_h, ori_w, ori_h])
        pad_image_arr = np.array([pad_size, 0, 0, pad_size])

        pred[:, :4] = pred[:, :4] - pad_image_arr
        pred[:, :4] = np.divide(pred[:, :4], scale_input_arr, dtype=np.float32)
        pred[:, :4] *= scale_image_arr

    print(pred[0, :4])
    # Draw fps
    fps = 1 / (t4 - t1)
    cv2.putText(orginal_img, "FPS: {:.2f}".format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw bounding box
    num_head = 0
    for x1, y1, x2, y2, conf, class_id in pred:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(orginal_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        num_head += 1
    cv2.imwrite("Test_imageq" + ".jpg", orginal_img)

    
    return num_head

img = cv2.imread(image_name)
model = torch.load('./static/model/yolov7_so.pt', map_location='cpu')['ema'].float().eval()
num_head = image_detect(img, model, '0.mp4', 734)
print(num_head)