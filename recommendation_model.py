from transformers import GPT2LMHeadModel, GPT2Tokenizer 

# Loading GPT-2 model along with tokenizer
model_name = "gpt2-medium"  
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Formatting Prompt for Recommendations
patient_vitals = """
Patient vital signs are as follows:
Blood Pressure: 150/95 mmHg,
Heart Rate: 88 bpm,
BMI: 32.5,
Respiratory Rate: 22 breaths/min,
Body Temperature: 38°C,
Oxygen Saturation: 94%.

Based on these values, provide detailed health recommendations for the patient:
"""

def generate_health_recommendations(prompt):
    """
    Generates health recommendations using GPT-2.

    Parameters:
        prompt (str): Input text containing vital signs.

    Returns:
        str: Generated health recommendations.
    """
    # Encode the input text
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate output using the GPT-2 model
    output = model.generate(
        input_ids,
        max_length=1000,           # Limit the length of output
        num_beams=5,              # Beam search for better results
        no_repeat_ngram_size=2,   # Avoid repeating phrases
        early_stopping=True       # Stop early for better coherence
    )

    # Decode and return the generated text
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Generate recommendations
recommendations = generate_health_recommendations(patient_vitals)
print("Generated Health Recommendations:")
print(recommendations)
