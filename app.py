from flask import Flask
UPLOAD_FOLDER = 'C:/biju/Experiments/yoloexp/Real-Time-Object-Detection/darkflow'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 * 1024