import base64
import io
import re

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("model/ai.h5")

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])

## Word labels
#word_dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'6',6:'6',7:'7',8:'8',9:'9', 10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',30:'U',31:'V',32:'W',33:'X',34:'Y',35:'Z'}
# word_dict = {0:'0',1:'1'}

def predict():
    data = request.get_json(force=True)
    encoded = data["image"]
    imgstr = re.search(r"base64,(.*)", encoded).group(1)
    decoded = base64.b64decode(imgstr)
    image = Image.open(io.BytesIO(decoded))
    # convert image to gray scale mode
    image = image.convert("L")
    image = image.resize((28, 28))
    image = np.array(image)
    # invert image
    image = 255 - image
    # normalize image
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)
    # make prediction with the model
    prediction = model.predict(image).reshape(-1)
    prob = prediction.tolist()
    prob = [format(num, ".4f") for num in prob]
    label = np.argmax(prob).tolist()
    word_dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'6',6:'6',7:'7',8:'8',9:'9', 10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',30:'U',31:'V',32:'W',33:'X',34:'Y',35:'Z'}
    confidence = prob[label]
    # cnfidence = prob[label]
    return jsonify({"prob": prob, "label": word_dict[label], "confidence": confidence})


if __name__ == "__main__":
    app.run()
