import cv2
from support_function import *
import time
import torch
import os 
from utils.general import non_max_suppression

WEIGHTS = "pretrain/yolov7_so.pt"
DEVICE = "cpu"
input_size = 640

CLASSES = ["Head"]

model = torch.load('./static/model/yolov7_so.pt', map_location='cpu')['ema'].float().eval()


def video_detect(video_path, videoname):
    """
    Detect head in video
    Draw bounding box, fps and return list number of head in each frame
    Save video in static/predict/videoname

    Args:
        video_path: path of video
        videoname: name of video

    Returns:
        num_head: number of head in video
    """
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    num_head_list = []

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        num_head = image_detect(frame, model, videoname, i)
        num_head_list.append(num_head)
    cap.release()

    # Write video from /static/images/videoname
    image_folder = './static/images/' + videoname

    # Get size of video
    img = cv2.imread(os.path.join(image_folder, '0.jpg'))
    size = img.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./static/predict/MPEG_' + videoname, fourcc, 25, (size[1], size[0]))  
    
    for i in range(len(num_head_list)):
        image_name = str(i) + '.jpg'
        img = cv2.imread(os.path.join(image_folder, image_name))
        out.write(img)
    out.release()
    # Convert 'MPEG' to 'h264'
    #os.system("ffmpeg -i ./static/predict/" + videoname + " -c:v libx264 -preset medium -crf 23 -c:a aac -b:a 128k ./static/predict/" + videoname)
    os.system("ffmpeg -i ./static/predict/MPEG_" +  videoname + ' -c:v libx264 -preset slow -crf 20 -c:a aac -b:a 160k -vf format=yuv420p -movflags +faststart ./static/predict/' + videoname)

    file = "video.MP4"
    video = cv2.VideoCapture(file)



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

    # Preprocess image
    t1 = time.time()
    img, pad_size, flag = process_image(img, img.shape[:2], size=(input_size, input_size))
    t2 = time.time()

    # Predict
    with torch.no_grad():
        pred = model(img[None])[0]
    t3 = time.time()


    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.25)[0].cpu().numpy()
    t4 = time.time()

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

    # Draw fps
    fps = 1 / (t4 - t1)
    cv2.putText(orginal_img, "FPS: {:.2f}".format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Create folder to save image
    if not os.path.exists("./static/images/" + videoname):
        os.makedirs("./static/images/" + videoname)

    # Draw bounding box
    num_head = 0
    for x1, y1, x2, y2, conf, class_id in pred:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(orginal_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        num_head += 1

    cv2.imwrite("./static/images/" + videoname + "/" + str(id_frame) + ".jpg", orginal_img)
    
    return num_head