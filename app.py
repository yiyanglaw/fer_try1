from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
from model import EmotionRecognitionModel

app = Flask(__name__)

# Load the pre-trained model
model = EmotionRecognitionModel()
model.load_state_dict(torch.load('emotion_model.pth'))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400

    # Decode the base64 image
    image_data = base64.b64decode(request.json['image'])
    image = Image.open(io.BytesIO(image_data))

    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    emotions = ['Angry', 'Happy', 'Sad']
    result = emotions[predicted.item()]

    return jsonify({'emotion': result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)