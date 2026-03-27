# 🏥 Medical Chatbot (AI + Rule-Based Hybrid)

A lightweight **medical symptom assistant chatbot** built using **Streamlit + Transformers**, designed to provide **safe, basic health guidance** while strictly following guardrails to avoid harmful or misleading advice.

---

## 🚀 Features

- 🧠 **Hybrid Intelligence**
  - Template-based symptom matching
  - AI-generated responses (fallback if model available)

- ⚡ **Automatic Model Selection**
  - Tries multiple models:
    - `Qwen/Qwen2.5-3B-Instruct`
    - `distilgpt2`
  - Falls back to safe template mode if models fail

- 🛡️ **Strong Safety Guardrails**
  - Detects emergency symptoms → redirects to urgent care
  - Blocks:
    - Dosage advice
    - Antibiotics/steroids suggestions
  - Extra warnings for:
    - Pregnancy
    - Infants
    - Elderly users

- 🧹 **Input Cleaning System**
  - Removes noise & irrelevant text
  - Extracts meaningful symptom keywords

- 🔍 **Smart Symptom Matching**
  - Matches user input with predefined medical templates
  - Uses keyword overlap scoring

- 💬 **Conversational UI**
  - Built using Streamlit chat interface
  - Maintains session history


## ⚙️ How It Works

### 1. Input Processing
- Cleans user input  
- Extracts relevant symptom keywords  

### 2. Guardrails Check
- Detects:
  - Emergency conditions  
  - Dosage-related queries  
  - High-risk categories  

### 3. Template Matching
- Compares symptoms with predefined conditions  
- Selects best match using keyword scoring  

### 4. Response Generation
- If model available → AI response  
- Else → rule-based fallback response  

---

## 🛡️ Safety Design

This chatbot is intentionally **restricted** to avoid misuse:

- ❌ No diagnosis  
- ❌ No prescription  
- ❌ No exact dosage  
- ❌ No antibiotics/steroids recommendation  

✔ Always:
- Encourages consulting a real doctor  
- Highlights emergency warning signs  

---

## 🧪 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/pranjalmishra27/Medicalchatbot.git
cd Medicalchatbot
2. Install Dependencies
pip install streamlit transformers
3. Run the App
streamlit run app.py
