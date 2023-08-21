

import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from main import classify_image # imports function from main.py

app = Flask(__name__, static_url_path='/static')

app.config['UPLOAD_FOLDER'] = 'static/uploads'  


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        model_path = "models/ensemble_model_v2.h5"  # ensemble model
        result = classify_image(file_path, model_path)

        return render_template('result.html', result=result, image_path=url_for('static', filename='uploads/' + filename))

if __name__ == '__main__':
    app.run()  # debug=True to debug



