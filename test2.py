import cv2
import os 

image_folder = './static/images/HT21-15-raw.mp4'
img = cv2.imread(os.path.join(image_folder, '0.jpg'))
size = img.shape
videoname = 'HT21-15-raw.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./static/predict/MPEG_' + videoname, fourcc, 25, (size[1], size[0]))  

for i in range(589):
    image_name = str(i) + '.jpg'
    img = cv2.imread(os.path.join(image_folder, image_name))
    out.write(img)
out.release()