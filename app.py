from flask import Flask, request, send_from_directory, jsonify
from flask_ngrok import run_with_ngrok
import os
from werkzeug.utils import secure_filename
from predict_data import predict
from flask_cors import CORS

UPLOAD_FOLDER = '/content/be-ta/file'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg'}

app = Flask(__name__)
cors = CORS(app)
run_with_ngrok(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello():
    return 'Hello, World!'


@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    file = request.files['file']
    if file:
        # Save the file to disk
        # file.save('/file')
        filename = secure_filename(file.filename)
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], filename))
        # Perform additional processing on the file here
        print("s")
        return 'File uploaded successfully'
    else:
        print("g")
        return 'No file was uploaded'
    

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER']), filename)

@app.route('/get_prediction/<filename>')
def predict_data(filename):
    prediction = predict(filename)
    print(prediction)
    return jsonify({'predict': prediction}), 200, {
        'Access-Control-Allow-Origin': '*'
    } 

if __name__ == '__main__':
    app.run()