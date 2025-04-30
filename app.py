from flask import Flask, request, render_template
import pickle
import joblib
import numpy as np

from PIL import Image
from keras.models import load_model
from keras.applications.vgg19 import preprocess_input
from werkzeug.utils import secure_filename
import os, base64, io

# Initialize Flask app
app = Flask(__name__)

# Define the directory to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the UPLOAD_FOLDER directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained XGBoost model
xgb_model = joblib.load('gradient_boosting_classifier.joblib')

# Load the VGG19 model
vgg19_model = load_model('vgg19_finetuned_best.h5')

# Load the encoding from pickle
with open('label_encoded_dict.pkl', 'rb') as f:
    le_dict = pickle.load(f)

# Load the scaler from pickle
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Function to preprocess user tabular input
def preprocess_user_input(user_input):
    # Manually encode categorical features based on the observed encodings from your dataset
    encoded_input = {
        'Gender': le_dict['Gender'].transform([user_input['gender']])[0],
        'Hormonal Changes': le_dict['Hormonal Changes'].transform([user_input['hormone']])[0],
        'Family History': le_dict['Family History'].transform([user_input['history']])[0],
        'Race/Ethnicity': le_dict['Race/Ethnicity'].transform([user_input['race']])[0],
        'Body Weight': le_dict['Body Weight'].transform([user_input['weight']])[0],
        'Calcium Intake': le_dict['Calcium Intake'].transform([user_input['calcium']])[0],
        'Vitamin D Intake': le_dict['Vitamin D Intake'].transform([user_input['vitamind']])[0],
        'Physical Activity': le_dict['Physical Activity'].transform([user_input['activity']])[0],
        'Smoking': le_dict['Smoking'].transform([user_input['smoking']])[0],
        'Alcohol Consumption': le_dict['Alcohol Consumption'].transform([user_input['alcohol']])[0],
        'Medical Conditions': le_dict['Medical Conditions'].transform([user_input['medical']])[0],
        'Medications': le_dict['Medications'].transform([user_input['medication']])[0],
        'Prior Fractures': le_dict['Prior Fractures'].transform([user_input['fracture']])[0],
        'Age': float(user_input['age'])  # Assuming 'Age' is submitted as part of the form
    }

    # Convert to array and reshape to match the input shape expected by the model
    model_input = np.array(list(encoded_input.values())).reshape(1, -1)

    # Scale the features
    model_input_scaled = scaler.transform(model_input)

    return model_input_scaled

# Function to extract confidence scores
def get_confidence_scores(model, data):
    # Get probability predictions for each class
    probs = model.predict_proba(data)
    # Assuming the positive class (Osteoporosis) is at index 1
    return probs[:, 1]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/inference', methods=['GET', 'POST'])
def inference():
    if request.method == 'POST':
        # Get the entire form data
        form_data = request.form
        # get image file
        image_file = request.files['image']
        if image_file:
            # Save the file to the upload folder
            filename = secure_filename(image_file.filename)
            image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # Preprocess user tabular input
        user_input = {
            'age': form_data['age'],
            'gender': form_data['gender'].capitalize(),
            'hormone': form_data['hormone'].capitalize(),
            'history': form_data['history'].capitalize(),
            'race': form_data['race'].title(),
            'weight': form_data['weight'].capitalize(),
            'calcium': form_data['calcium'].capitalize(),
            'vitamind': form_data['vitamind'].capitalize(),
            'activity': form_data['activity'].capitalize(),
            'smoking': form_data['smoking'].capitalize(),
            'alcohol': form_data['alcohol'].capitalize(),
            'medical': form_data['medical'].title(),
            'medication': form_data['medication'].capitalize(),
            'fracture': form_data['fracture'].capitalize()
        }
        #print(user_input)
        user_input_processed = preprocess_user_input(user_input)
        #print(user_input_processed)

        # Make a prediction with the XGBoost model
        xgb_prediction = xgb_model.predict_proba(user_input_processed)
        xgb_pred_arg = np.argmax(xgb_prediction, axis=1)
        xgb_predicted_label = 'Osteoporosis' if xgb_pred_arg == 1 else 'No Osteoporosis'
        print(xgb_predicted_label, xgb_prediction)

        # Preprocess user image input
        img = Image.open(image_file)
        img = img.resize((224, 224))  # Resize the image to the required dimensions
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # prediction using VGG19 model
        vgg19_pred = vgg19_model.predict(img_array, verbose=1)
        predicted_arg = np.argmax(vgg19_pred, axis=1)
        vgg_predicted_label = 'Osteoporosis' if predicted_arg == 1 else 'No Osteoporosis'
        print(vgg19_pred, predicted_arg, vgg_predicted_label)
        
        # Convert the output image to base64 format for rendering in the HTML template
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        image_output = base64.b64encode(img_bytes).decode('utf-8')
        print(xgb_prediction, vgg19_pred )
        avg_accuracy=(xgb_prediction[0][xgb_pred_arg] + vgg19_pred[0][predicted_arg] )/2
        return render_template('middle.html', 
                               vgg19_pred = vgg_predicted_label,
                               vgg19_pred_prob= vgg19_pred[0][predicted_arg], 
                               xgb_prediction = xgb_predicted_label,
                               xgb_pred_prob= xgb_prediction[0][xgb_pred_arg],
                               image_output = image_output,
                               avg_accuracy=avg_accuracy)
    return render_template('middle.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/treatment')
def treatment():
    return render_template('treatment.html')

if __name__ == "__main__":
    app.run(debug=True)
