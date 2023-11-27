from flask import Flask, render_template, request
import os 
from head_detection import video_detect
from datetime import datetime

app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')

@app.route('/', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        upload_file = request.files['video_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH, filename)
        upload_file.save(path_save)

        list_num_head = video_detect(path_save, filename)
        return render_template('index.html', video_name=filename, upload=True)
    return render_template('index.html', upload=False, filename="a")


if __name__ == "__main__":
    app.run(host= '0.0.0.0', port=8000)