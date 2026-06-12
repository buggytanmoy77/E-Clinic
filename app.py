from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import joblib
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv(".env.local")
app = Flask(__name__, template_folder='templates')

# ================== Groq Client ==================
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ================== Chatbot Data ==================
ChatbotCSV_Path = 'medquad.csv'
data = pd.read_csv(ChatbotCSV_Path)
if len(data.columns) >= 2:
    data = data.iloc[:, :2]
    data.columns = ['question', 'answer']
else:
    raise ValueError("The CSV file must have at least two columns for 'Question' and 'Answer'.")

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(data['question'].values.astype('U'))

# ================== ChatBot Functionality ==================

def get_llm_response(query):
    Api_Url = "https://api-inference.huggingface.co/models/bigscience/bloomz-560m"
    Headers = {"Authorization": "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"}
    Payload = {"inputs": query}
    try:
        response = requests.post(Api_Url, headers=Headers, json=Payload)
        if response.status_code == 200:
            return response.json()[0]['generated_text']
        else:
            return "I'm sorry, I couldn't process your request at the moment."
    except Exception as e:
        return str(e)

@app.route('/chat', methods=['POST'])
def chat():
    UserInput = request.json.get('query', '')
    if not UserInput:
        return jsonify({"error": "Query is required."}), 400

    user_vector = vectorizer.transform([UserInput])
    simi = cosine_similarity(user_vector, question_vectors)
    BestIndex = simi.argmax()
    BestScore = simi[0, BestIndex]

    threshold = 0.6
    if BestScore >= threshold:
        response = data.iloc[BestIndex]['answer']
    else:
        response = get_llm_response(UserInput)

    return jsonify({"response": response})

# ================== Health Score and Recommendation Functionality ==================

# Load model + scaler saved together
_checkpoint = joblib.load('health_score_model.pkl')
model_score  = _checkpoint['model']
score_scaler = _checkpoint['scaler']

def generate_health_recommendations(patient_vitals: str) -> str:
    """Use Groq LLaMA-3.3-70b to generate structured health recommendations."""
    try:
        chat_completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical health advisor. Given patient vital signs, "
                        "provide concise, actionable health recommendations. "
                        "Format your response as a short bulleted list (5-7 points). "
                        "Each bullet should start with '- ' and be one clear sentence. "
                        "Cover: diet, exercise, sleep, hydration, and any vital-specific advice. "
                        "Be direct and medically accurate. Do not include disclaimers or preambles."
                    )
                },
                {
                    "role": "user",
                    "content": patient_vitals
                }
            ],
            max_tokens=400,
            temperature=0.5,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"- Unable to generate recommendations at this time.\n- Error: {str(e)}"

def check_vitals_for_alerts(hr, rr, temp, sys_bp, dia_bp, spo2, age, bmi):
    """Rule-based critical vital detection. Returns list of alert strings or empty list."""
    alerts = []
    hr, rr, temp, sys_bp, dia_bp, spo2 = (
        float(hr), float(rr), float(temp),
        float(sys_bp), float(dia_bp), float(spo2)
    )
    if hr < 40 or hr > 150:
        alerts.append(f"⚠ Heart Rate {hr:.0f} bpm is critically {'low' if hr < 40 else 'high'} (normal: 60–100 bpm).")
    if rr < 8 or rr > 30:
        alerts.append(f"⚠ Respiratory Rate {rr:.0f} br/min is critically {'low' if rr < 8 else 'high'} (normal: 12–20 br/min).")
    if temp < 35.0 or temp > 39.5:
        alerts.append(f"⚠ Body Temperature {temp:.1f}°C indicates {'hypothermia' if temp < 35 else 'high fever'} (normal: 36.1–37.2°C).")
    if sys_bp > 180 or sys_bp < 70:
        alerts.append(f"⚠ Systolic BP {sys_bp:.0f} mmHg is critically {'low' if sys_bp < 70 else 'high'} (normal: 90–120 mmHg).")
    if dia_bp > 120 or dia_bp < 40:
        alerts.append(f"⚠ Diastolic BP {dia_bp:.0f} mmHg is critically {'low' if dia_bp < 40 else 'high'} (normal: 60–80 mmHg).")
    if spo2 < 90:
        alerts.append(f"⚠ Oxygen Saturation {spo2:.0f}% is dangerously low — seek immediate medical attention (normal: 95–100%).")
    if bmi < 14 or bmi > 45:
        alerts.append(f"⚠ BMI {bmi:.1f} is critically {'low' if bmi < 14 else 'high'} — immediate dietary/medical review needed.")
    return alerts


@app.route('/submit', methods=['POST'])
def submit():
    heart_rate = request.form.get('heart-rate-text')
    respiratory_rate = request.form.get('respiratory-rate-text')
    body_temperature = request.form.get('body-temperature-text')
    systolic_bp = request.form.get('systolic-text')
    diastolic_bp = request.form.get('diastolic-text')
    oxygen_saturation = request.form.get('oxygen-saturation-text')
    age = request.form.get('age-text')
    weight = float(request.form.get('weight-text'))
    height = float(request.form.get('height-text'))

    # Order must match training features exactly:
    # Heart Rate, Respiratory Rate, Body Temperature, Oxygen Saturation,
    # Systolic BP, Diastolic BP, Age, Weight, Height
    raw_input = np.array([[heart_rate, respiratory_rate, body_temperature,
                           oxygen_saturation, systolic_bp, diastolic_bp,
                           age, weight, height]], dtype=float)
    scaled_input = score_scaler.transform(raw_input)
    health_score = int(round(model_score.predict(scaled_input)[0]))
    bmi = weight / (float(height) ** 2)

    # Check for critical vital alerts first
    alerts = check_vitals_for_alerts(
        heart_rate, respiratory_rate, body_temperature,
        systolic_bp, diastolic_bp, oxygen_saturation, age, bmi
    )

    if alerts:
        alert_text = "\n".join(alerts)
        alert_text += "\n\nPlease seek immediate medical attention or re-check your readings before proceeding."
        return render_template('index.html', score=health_score,
                               recommendation=None, is_alert=True,
                               alert_text=alert_text)

    patient_vitals = (
        f"Patient vitals: BP {systolic_bp}/{diastolic_bp} mmHg, "
        f"Heart Rate {heart_rate} bpm, BMI {bmi:.2f}, "
        f"Respiratory Rate {respiratory_rate} breaths/min, "
        f"Body Temperature {body_temperature}°C, "
        f"Oxygen Saturation {oxygen_saturation}%, Age {age}. "
        f"Health Score: {health_score}/100. "
        f"Provide personalised health recommendations."
    )

    recommendations = generate_health_recommendations(patient_vitals)
    if not recommendations:
        recommendations = "- No recommendations available at this time."

    return render_template('index.html', score=health_score,
                           recommendation=recommendations, is_alert=False,
                           alert_text=None)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)