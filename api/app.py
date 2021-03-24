import io
import json

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request


app = Flask(__name__)
class_index = json.load(open('dest.json'))
model = models.resnext50_32x4d(pretrained=True)
classifier = nn.Sequential(nn.Linear(2048, 1024),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(1024, 256),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(256, 34),
                           )

model.fc = classifier
model.load_state_dict(torch.load('model_weights.pt'))
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return predicted_idx, class_index[predicted_idx]

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()