import numpy as np
from flask import Flask, render_template, request, jsonify
from keras.models import load_model

from mocr.predict import predict as pdt

model = None
app = Flask(__name__)
app.config['SECRET_KEY'] = '22334455'


def load_data():
    global model
    model = load_model('mocr/mlocr.h5')

@app.route('/predict', methods=['POST'])
def predict():
    img = request.get_json(force=True)['img']
    data = pdt(img)
    print(type(data))
    return jsonify(data)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    print("* Loading Keras model and Flask starting server..."
          "please wait until server has fully started")
    load_data()
    app.run(threaded=False, debug = True)
