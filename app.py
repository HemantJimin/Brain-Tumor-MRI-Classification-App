from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load your trained model (you'll need to upload model.h5)
# Uncomment the line below after uploading model.h5
# model = tf.keras.models.load_model('model.h5')

# Define class labels
CLASS_LABELS = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def preprocess_image(image):
    """Preprocess image for model prediction"""
    img = image.resize((128, 128))  # Match your model's input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and preprocess image
        image = Image.open(file.stream).convert('RGB')
        processed_image = preprocess_image(image)
        
        # Make prediction (uncomment after uploading model)
        # predictions = model.predict(processed_image)
        # predicted_class_idx = np.argmax(predictions[0])
        # predicted_class = CLASS_LABELS[predicted_class_idx]
        # confidence = float(predictions[0][predicted_class_idx] * 100)
        
        # Demo response (remove after model is uploaded)
        predicted_class = 'glioma_tumor'
        confidence = 85.5
        all_predictions = {
            'glioma_tumor': 85.5,
            'meningioma_tumor': 8.2,
            'no_tumor': 3.1,
            'pituitary_tumor': 3.2
        }
        
        # Prepare all class probabilities (uncomment after model upload)
        # all_predictions = {
        #     CLASS_LABELS[i]: float(predictions[0][i] * 100) 
        #     for i in range(len(CLASS_LABELS))
        # }
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
