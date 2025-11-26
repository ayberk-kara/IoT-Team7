from flask import Flask, request, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import torch
from PIL import Image
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.to(device)

is_processing = False
last_people_count = None
last_frame_name = None

# optional endpoint for checking if server is busy
# not necessary to use
@app.route('/ready', methods=['GET']) 
def server_ready():
    print("Is processing:", is_processing)
    return jsonify({"ready": not is_processing})

@app.route('/upload', methods=['POST'])
def upload_file():
    global is_processing, last_people_count, last_frame_name
    if is_processing:
        return 'Server is busy', 503

    is_processing = True
    start_time = time.time()
    if 'file' not in request.files:
        is_processing = False
        return 'No file part', 400
    file = request.files['file']
    
    if file.filename == '':
        is_processing = False
        return 'No selected file', 400
    
    filename = secure_filename("temp.jpg")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    image = Image.open(file_path)

    results = model(image)
    
    last_people_count = sum(1 for det in results.pred[0] if int(det[5]) == 0)
    last_frame_name = file.filename.replace(".jpg","")

    os.system('clear')

    total_time = time.time() - start_time
    print(f"Number of people: {last_people_count}")
    print(f"Total time: {total_time:.4f} seconds")
    print("Time passed in ms:", round((time.time() - float(last_frame_name)) * 1000))
    
    is_processing = False
    return jsonify({})

@app.route('/people_count', methods=['GET'])
def get_people_count():
    if last_people_count is None:
        return jsonify({"message": "No image uploaded yet"}), 400
    return jsonify({"people_detected": last_people_count, "last_frame": last_frame_name})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)