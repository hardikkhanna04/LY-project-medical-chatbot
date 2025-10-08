import pandas as pd
from flask import Flask, request, render_template_string, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import numpy as np
import re

app = Flask(__name__)

# --- Project Dataset ---
DATA = {
    'symptoms': [
        "runny nose, sneezing, mild sore throat, cough",
        "high fever, severe cough, body aches, chills, fatigue",
        "stomach pain, diarrhea, nausea, vomiting",
        "sharp chest pain, shortness of breath, left arm pain",
        "itchy eyes, sneezing, runny nose, congestion during spring",
        "joint pain, swelling, stiffness, redness in joints",
        "sudden severe headache, vomiting, sensitivity to light",
        "skin rash, hives, difficulty breathing after eating nuts",
        "burning sensation when urinating, frequent urination, pelvic pain",
        "muscle cramps, fatigue, weakness, dehydration",
        "coughing up blood, persistent cough, unexplained weight loss",
        "frequent headaches, neck stiffness, fever",
        "blurry vision, frequent urination, increased thirst",
        "sore throat, white patches on tonsils, fever",
        "difficulty sleeping, sweating, rapid heart rate, anxiety",
    ],
    'condition': [
        "Common Cold",
        "Flu",
        "Gastroenteritis",
        "Emergency",
        "Allergy",
        "Arthritis",
        "Emergency",
        "Emergency",
        "Urinary Tract Infection (UTI)",
        "Dehydration/Electrolyte Imbalance",
        "Emergency",
        "Meningitis",
        "Diabetes",
        "Strep Throat",
        "Anxiety/Thyroid Issues",
    ],
    'advice': [
        "Rest, stay hydrated, and use over-the-counter medication for symptom relief.",
        "Get plenty of rest and fluids. An antiviral may be prescribed by a doctor if you are in a high-risk group.",
        "Focus on rehydration with oral rehydration solutions. Avoid solid food for a few hours.",
        "This is a medical emergency. Seek immediate medical attention or call emergency services.",
        "Avoid allergens. Over-the-counter antihistamines can help. If severe, see an allergist.",
        "Use hot or cold packs. Anti-inflammatory drugs can help with pain and swelling. Consult a doctor.",
        "This is a medical emergency. Seek immediate medical attention or call emergency services.",
        "This is a medical emergency. Use an EpiPen if available and call emergency services immediately.",
        "Drink plenty of water and cranberry juice. A doctor will likely prescribe antibiotics.",
        "Drink water or a sports drink with electrolytes. If symptoms persist, consult a doctor.",
        "This is a medical emergency. Seek immediate medical attention.",
        "Seek immediate medical attention. These could be signs of a serious infection.",
        "See a doctor for a blood glucose test. Lifestyle changes and medication may be needed.",
        "A rapid strep test from a doctor is recommended. Antibiotics are typically prescribed.",
        "Consult a healthcare professional to determine the cause. Stress management or medication may be needed.",
    ]
}

df = pd.DataFrame(DATA)

# --- Preprocessing and Model Training ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['clean_symptoms'] = df['symptoms'].apply(clean_text)
pipeline = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)
pipeline.fit(df['clean_symptoms'], df['condition'])

CONF_THRESH = 0.55

def get_advice(condition):
    advice_row = df[df['condition'] == condition]
    return advice_row['advice'].iloc[0] if not advice_row.empty else "No specific advice found."

def generate_reply(user_text):
    clean_input = clean_text(user_text)
    if not clean_input:
        return {
            "predicted_condition": "Unknown",
            "confidence": 0.0,
            "advice": "Please enter some symptoms to get a response.",
        }

    probs = pipeline.predict_proba([clean_input])[0]
    classes = pipeline.classes_
    best_idx = np.argmax(probs)
    best_prob = probs[best_idx]
    predicted_condition = classes[best_idx]

    if best_prob < CONF_THRESH:
        predicted_condition = "Uncertain/General"
        advice = "I am not confident in a specific diagnosis. Please provide more details or consult a healthcare professional."
    elif predicted_condition == "Emergency":
        advice = "Your symptoms may indicate a medical emergency. Please seek immediate medical attention or call emergency services."
    else:
        advice = get_advice(predicted_condition)

    reply = {
        "predicted_condition": predicted_condition,
        "confidence": float(best_prob),
        "advice": advice,
    }
    return reply

# --- Flask Routes and HTML Interface (no changes needed here) ---
HOME_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Chatbot (LY Project)</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(to right, #e8f5e9, #c8e6c9); margin: 0; padding: 2rem; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
        .chat-container { width: 100%; max-width: 800px; background: #fff; padding: 2rem; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); display: flex; flex-direction: column; }
        h1 { color: #2e7d32; text-align: center; margin-bottom: 1rem; }
        .disclaimer { background: #fff3e0; color: #e65100; padding: 1rem; border-radius: 8px; border: 1px solid #ffb74d; margin-bottom: 1.5rem; font-weight: 500; }
        .chatbox { flex: 1; height: 400px; overflow-y: auto; padding: 1rem; border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 1rem; display: flex; flex-direction: column-reverse; }
        .message { margin-bottom: 0.8rem; padding: 0.7rem 1rem; border-radius: 20px; max-width: 80%; word-wrap: break-word; }
        .user-msg { background-color: #e3f2fd; color: #1565c0; align-self: flex-end; }
        .bot-msg { background-color: #e8f5e9; color: #2e7d32; align-self: flex-start; }
        .message strong { font-weight: bold; }
        form { display: flex; gap: 0.5rem; }
        input[type="text"] { flex: 1; padding: 0.8rem 1.2rem; border: 1px solid #ccc; border-radius: 20px; font-size: 1rem; }
        button { padding: 0.8rem 1.5rem; border: none; background: #4caf50; color: white; border-radius: 20px; cursor: pointer; transition: background-color 0.3s; }
        button:hover { background-color: #388e3c; }
    </style>
</head>
<body>
<div class="chat-container">
    <h1>Intelligent Medical Chatbot</h1>
    <div class="disclaimer">
        ⚠️ **Disclaimer:** This is a prototype for an academic project. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider for any medical questions.
    </div>
    <div id="chatbox" class="chatbox"></div>
    <form id="chatform" onsubmit="return sendMessage()">
        <input type="text" id="user-input" placeholder="Describe your symptoms (e.g., runny nose, fever)..." autocomplete="off" />
        <button type="submit">Send</button>
    </form>
</div>
<script>
    async function sendMessage() {
        const userInput = document.getElementById('user-input').value.trim();
        if (!userInput) return false;
        
        appendMessage(userInput, 'user-msg');
        document.getElementById('user-input').value = '';

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: userInput })
            });
            const data = await response.json();
            
            let botMessage = `**Condition:** ${data.predicted_condition} (Confidence: ${(data.confidence * 100).toFixed(1)}%)`;
            if (data.advice) {
                botMessage += `<br/>**Advice:** ${data.advice}`;
            }
            
            appendMessage(botMessage, 'bot-msg');
        } catch (error) {
            console.error('Error:', error);
            appendMessage('Sorry, something went wrong. Please try again later.', 'bot-msg');
        }
        return false;
    }

    function appendMessage(text, className) {
        const chatbox = document.getElementById('chatbox');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${className}`;
        messageDiv.innerHTML = text;
        chatbox.insertBefore(messageDiv, chatbox.firstChild);
    }
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HOME_HTML)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        data = request.get_json() or {}
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        reply = generate_reply(text)
        return jsonify(reply)
    except Exception as e:
        # Log the error to the terminal
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8501)