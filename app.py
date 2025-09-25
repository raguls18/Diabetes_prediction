# app.py

from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import model_from_json
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from your .env file
load_dotenv()

app = Flask(__name__)

# --- Food Recommendation Dictionaries ---
HIGH_RISK_FOODS = {
    "Eat More Of ✅": [
        "Non-Starchy Vegetables (spinach, broccoli, bell peppers)",
        "Lean Proteins (salmon, chicken breast, lentils, beans)",
        "Whole Grains (quinoa, brown rice, oats)"
    ],
    "Limit or Avoid ❌": [
        "Sugary Drinks (soda, sweetened juice)",
        "Refined Carbs (white bread, pastries, white rice)",
        "Processed & Fast Foods"
    ]
}

LOW_RISK_FOODS = {
    "Maintain With ✅": [
        "A balanced mix of colorful vegetables",
        "Consistent lean protein intake",
        "Fiber-rich whole grains for energy"
    ],
    "Continue to Limit ❌": [
        "Excessive sugar and sugary drinks",
        "Highly processed snacks"
    ]
}


# --- Configure the Gemini AI API ---
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Please create a .env file and add your key.")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-pro')
    print("Gemini AI API configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    gemini_model = None


# --- Load your local Keras Model for prediction ---
def load_keras_model():
    try:
        with open('model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model.weights.h5")
        print("Keras prediction model loaded successfully from disk.")
        return loaded_model
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        return None

keras_model = load_keras_model()


@app.route('/')
def home():
    """Renders the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Receives data, gets a prediction, and generates all recommendations."""
    if not keras_model:
        return jsonify({'error': 'Keras model is not loaded'}), 500

    try:
        form_values = request.form.to_dict()
        feature_order = [
            'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
        ]
        
        data = [float(form_values[feature]) for feature in feature_order]
        final_features = np.array(data).reshape(1, -1)
        
        # Get the prediction from your local Keras model
        probability = keras_model.predict(final_features)[0][0]
        prediction = 1 if probability > 0.5 else 0
        
        ai_recommendations = []
        food_recs = {}
        
        if prediction == 1:
            result_text = "This person has a high risk of diabetes."
            food_recs = HIGH_RISK_FOODS
            
            # Call Gemini AI for personalized recommendations
            if gemini_model:
                input_details = ", ".join(f"{key}: {'Yes' if value == '1' else 'No'}" for key, value in form_values.items() if key not in ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income'])
                prompt = (f"A person has a high risk for diabetes with these factors: {input_details}, BMI: {form_values['BMI']}, "
                          f"General Health scale (1=Excellent, 5=Poor): {form_values['GenHlth']}. "
                          f"Based ONLY on these specific factors, provide 3-4 concise, actionable, and safe recommendations to help lower their risk. "
                          f"Start each recommendation with a bullet point (*). Do not write an introduction or a conclusion. "
                          f"You MUST NOT mention specific medications or give complex medical advice.")
                
                try:
                    print("Requesting recommendations from Gemini AI...")
                    response = gemini_model.generate_content(prompt)
                    ai_recommendations = [rec.strip() for rec in response.text.replace('*', '').split('\n') if rec.strip()]
                except Exception as e:
                    print(f"Gemini API call failed: {e}")
                    ai_recommendations = ["AI service is unavailable. Please consult a healthcare professional for personalized advice."]
            else:
                 ai_recommendations = ["AI model not configured. Please consult a doctor."]

        else:
            result_text = "This person has a low risk of diabetes."
            food_recs = LOW_RISK_FOODS
            ai_recommendations = [
                "Congratulations on your low-risk profile! Continue your healthy habits.",
                "Maintain a balanced diet and stay physically active to keep your risk low."
            ]
        
        return render_template('index.html', 
                               prediction_text=result_text, 
                               prediction_prob=f'Risk Score (Probability): {probability:.2f}',
                               recommendations=ai_recommendations,
                               food_recs=food_recs)
                               
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error processing input: {e}')


if __name__ == "__main__":
    app.run(debug=True, port=5000)