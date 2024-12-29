from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import joblib
from transformers import GPT2LMHeadModel, GPT2Tokenizer 

app = Flask(__name__, template_folder='templates')

ChatbotCSV_Path = 'medquad.csv' # CSV having Medical QNA
data = pd.read_csv(ChatbotCSV_Path)
if len(data.columns) >= 2:
    data = data.iloc[:, :2]
    data.columns = ['question', 'answer']
else:
    raise ValueError("The CSV file must have at least two columns for 'Question' and 'Answer'.")

# ================== ChatBot Functionality ==================

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(data['question'].values.astype('U'))

# Function for getting llm response
def get_llm_response(query):
    Api_Url = "https://api-inference.huggingface.co/models/bigscience/bloomz-560m"
    Headers = {"Authorization":"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"}
    Payload = {"inputs":query}
    try:
        response = requests.post(Api_Url, headers=Headers, json=Payload)
        if response.status_code == 200:
            return response.json()[0]['generated_text']
        else:
            return "I'm sorry, I couldn't process your request at the moment."
    except Exception as e:
        return str(e)

@app.route('/chat', methods=['POST'])
# Chatbot Function using threshold factor
def chat():
    UserInput = request.json.get('query','')
    if not UserInput:
        return jsonify({"error": "Query is required."}), 400
    
    # Defining Cosine similarity
    user_vector = vectorizer.transform([UserInput])
    simi = cosine_similarity(user_vector,question_vectors)
    BestIndex = simi.argmax()
    BestScore = simi[0, BestIndex]

    # Threshold for a valid match
    threshold = 0.6
    if BestScore >= threshold:
        response = data.iloc[BestIndex]['answer']
    else:
        response = get_llm_response(UserInput)

    return jsonify({"response": response})

# ================== Health Score and Recommendation Functionality ==================

model_score = joblib.load('health_score_model.pkl') # Health Score Pickle file

# Loading GPT-2 model along with tokenizer
model_name = "gpt2-medium"  
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_health_recommendations(prompt):
    input_ids = tokenizer.encode(prompt,return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=220,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    text = tokenizer.decode(output[0],skip_special_tokens=True)

    # Spliting into lines
    if text.startswith(prompt):
        text = text[len(prompt):].strip()

    import re
    Recommendations = re.split(r'[.;]',text)  # Now Spliting sentences
    Formatted_Recommendations = "\n".join(
        f"- {rec.strip()}" for rec in Recommendations if rec.strip() # Formatted Using Bullet Points
    )
    return Formatted_Recommendations

@app.route('/submit', methods=['POST'])
def submit():
    # Getting all input values from the I/P Form
    heart_rate = request.form.get('heart-rate-text')
    respiratory_rate = request.form.get('respiratory-rate-text')
    body_temperature = request.form.get('body-temperature-text')
    systolic_bp = request.form.get('systolic-text')
    diastolic_bp = request.form.get('diastolic-text')
    oxygen_saturation = request.form.get('oxygen-saturation-text')
    age = request.form.get('age-text')
    weight = float(request.form.get('weight-text'))
    height = float(request.form.get('height-text'))

    # Structured Input for the Health score Calculation
    model_input = [[heart_rate, respiratory_rate, body_temperature, systolic_bp, diastolic_bp, oxygen_saturation, age, weight, height]]
    
    # Predicting Health Score
    health_score = int(model_score.predict(model_input)[0])  
    
    # Formatting Prompt for Recommendations
    patient_vitals = f"""
    Patient vital signs are as follows:
    Blood Pressure: {systolic_bp}/{diastolic_bp} mmHg,
    Heart Rate: {heart_rate} bpm,
    BMI: {weight / (float(height)**2):.2f},
    Respiratory Rate: {respiratory_rate} breaths/min,
    Body Temperature: {body_temperature}\u00b0C,
    Oxygen Saturation: {oxygen_saturation}%.
    
    Based on these values, provide detailed health recommendations for the patient:
    """

    # Now Generating health recommendations using the GPT-2 model
    recommendations = generate_health_recommendations(patient_vitals)
    if not recommendations:
        recommendations = "N/A"
    return render_template('index.html', score=health_score, recommendation=recommendations)

# Common Route
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
