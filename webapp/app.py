from flask import Flask, render_template, request, jsonify
import numpy as np
import PIL

from predictor import Predictor

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img = request.files.get('image', '')
        img.seek(0)
        imgarray = np.asarray(PIL.Image.open(img))
        
        classifier = Predictor()
        pred = classifier.predict(imgarray)

        return render_template('result.html', pred = pred)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug = True)