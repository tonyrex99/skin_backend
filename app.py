from flask import Flask, request, jsonify, redirect
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from flasgger import Swagger
from flasgger.utils import swag_from
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)  # Enable CORS
swagger = Swagger(app)

# Load your trained TensorFlow model
model = tf.keras.models.load_model('skin_disease_detection_model_mobilenetv3_20240611_180748.keras')

def preprocess_image(image):
    """
    Preprocess the image to the required format for the TensorFlow model.
    """
    image = image.resize((224, 224))  # Example size, change as per your model's requirement
    image = np.array(image) / 255.0   # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def getDiseaseName(label):
    class_labels = {
    'AKIEC' : 'Actinic Keratosis',
    'BCC' : 'Basal Cell Carcinoma ',
    'BKL' : ' Benign Keratosis',
    'DF' : 'Dermatofibroma',
    'MEL' : 'Melanoma',
    'NV': 'Nevus',
    'VASC': 'Vascular lesion ',
     }
    print(f"label is {label} disease is {class_labels[label]}")
    return class_labels[label]

@app.route('/')
def index():
    """
    Redirect to the Swagger UI documentation.
    """
    return redirect('/apidocs')

@swag_from({
    'responses': {
        200: {
            'description': 'Predictions for the uploaded images',
            'examples': {
                'application/json': {
                    'predictions': [
                        {
                            'image_name': 'image1.jpg',
                            'predictions': [
                                {'label': 'cat', 'accuracy': 0.95},
                                {'label': 'dog', 'accuracy': 0.05}
                            ]
                        },
                        {
                            'image_name': 'image2.jpg',
                            'predictions': [
                                {'label': 'car', 'accuracy': 0.80},
                                {'label': 'bike', 'accuracy': 0.20}
                            ]
                        }
                    ]
                }
            }
        }
    },
    'parameters': [
        {
            'name': 'images',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'Upload one or more images',
            'collectionFormat': 'multi'
        }
    ],
    'produces': ['application/json'],
    'consumes': ['multipart/form-data'],
    'tags': ['Predictions'],
    'description': 'API endpoint to handle image predictions.'
})

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to handle image predictions.
    """
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    images = request.files.getlist('images')
    predictions = []

    for img in images:
        image_name = img.filename
        image = Image.open(io.BytesIO(img.read()))
        processed_image = preprocess_image(image)
        preds = model.predict(processed_image)
        label_encoder = LabelEncoder()
        # Assuming the model outputs probabilities and you have the class labels
        class_labels = [
    'AKIEC',
    'BCC',
    'BKL',
    'DF',
    'MEL',
    'NV',
    'VASC',
                       ]
        top_n = 3
        label_encoder.fit(class_labels)
        predicted_class_index = np.argmax(preds[0])
        predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
        top_predictions = np.argsort(preds[0])[-top_n:]  # Get indices of top N predictions (descending order)
        top_probabilities = preds[0][top_predictions]
        top_labels = [class_labels[i] for i in top_predictions]
        top_probs = [f"{prob:.2f}%" for prob in top_probabilities] 
      
        xtop_predictions = sorted(zip(class_labels, preds[0]), key=lambda x: x[1], reverse=True)
        top_predictions = [{'label': getDiseaseName(label), 'accuracy': float(accuracy)} for label, accuracy in zip(top_labels, top_probs)]

        predictions.append({
            'image_name': image_name,
            'predictions': top_predictions
        })

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)
