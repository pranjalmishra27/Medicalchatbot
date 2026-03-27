# Medical Chatbot

This is a lightweight Streamlit chatbot that uses:

- a stronger Hugging Face instruct model: `Qwen/Qwen2.5-3B-Instruct`
- an optional 4B model choice in the sidebar: `google/gemma-3-4b-it`
- a separate JSON symptom template file
- basic symptom matching to make the chatbot a little more useful on weak systems
- expanded condition coverage with many common respiratory, stomach, skin, pain, endocrine, and urinary symptom profiles
- locked safer generation settings tuned for stable low-hallucination responses
- automatic weak-output retry with fallback to template guidance

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py`: Streamlit chatbot UI and model logic
- `medical_templates.json`: symptom document/template data
- `requirements.txt`: Python packages

## Notes

- This is not real medical advice.
- You can switch between 3B and 4B model options from the sidebar.
- Generation settings are fixed in code for safer, more stable output quality.
- You can expand `medical_templates.json` with more symptoms and suggestions later.
