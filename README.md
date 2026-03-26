# Tiny Medical Chatbot

This is a lightweight Streamlit chatbot that uses:

- a small Hugging Face model: `sshleifer/tiny-gpt2`
- a separate JSON symptom template file
- basic symptom matching to make the chatbot a little more useful on weak systems

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
- The model is intentionally tiny, so answers may be weak but still interesting.
- You can expand `medical_templates.json` with more symptoms and suggestions later.
