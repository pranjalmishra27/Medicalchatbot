import json
from pathlib import Path

import streamlit as st
from transformers import pipeline


BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = BASE_DIR / "medical_templates.json"
# Automatic model selection (best effort). If all fail, app uses template-only mode.
MODEL_CANDIDATES = [
    "Qwen/Qwen2.5-3B-Instruct",
    "distilgpt2",
]

# Locked generation settings chosen for stable, low-hallucination answers.
GENERATION_SETTINGS = {
    "temperature": 0.2,
    "top_p": 0.75,
    "max_new_tokens": 140,
    "repetition_penalty": 1.15,
}
AUTO_RETRY_ATTEMPTS = 1

EMERGENCY_KEYWORDS = {
    "chest pain",
    "can not breathe",
    "cannot breathe",
    "breathing trouble",
    "breathless",
    "shortness of breath",
    "stroke",
    "seizure",
    "passed out",
    "unconscious",
    "fainted",
    "vomiting blood",
    "blood in stool",
    "suicide",
    "kill myself",
    "overdose",
}

DOSAGE_KEYWORDS = {
    "dose",
    "dosage",
    "how many tablets",
    "how much medicine",
    "mg",
    "antibiotic",
    "prescription",
    "steroid",
}

HIGH_RISK_KEYWORDS = {
    "pregnant",
    "pregnancy",
    "baby",
    "infant",
    "newborn",
    "toddler",
    "elderly",
    "old age",
}

GENERIC_SYMPTOM_WORDS = {
    "fever",
    "cough",
    "cold",
    "pain",
    "rash",
    "itching",
    "vomiting",
    "nausea",
    "diarrhea",
    "loose",
    "motion",
    "throat",
    "headache",
    "dizzy",
    "weakness",
    "burning",
    "urine",
    "breathing",
    "sneezing",
    "swelling",
    "bodyache",
}


@st.cache_resource(show_spinner=True)
def load_generator():
    """
    Try models in priority order and return first successful generator.
    Returns (generator, model_name) or (None, None).
    """
    for model_name in MODEL_CANDIDATES:
        try:
            gen = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                max_new_tokens=90,
            )
            return gen, model_name
        except Exception:
            continue
    return None, None


@st.cache_data(show_spinner=False)
def load_templates():
    with TEMPLATE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_words(text):
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
    return {word for word in cleaned.split() if len(word) > 2}


def normalize_text(text):
    return " ".join(text.lower().split())


def build_keyword_bank(templates):
    keywords = set(GENERIC_SYMPTOM_WORDS)
    for item in templates["conditions"]:
        keywords.update(word.lower() for word in item["keywords"])
        keywords.update(normalize_words(item["name"]))
    return keywords


def extract_relevant_symptoms(user_text, templates):
    keyword_bank = build_keyword_bank(templates)
    cleaned_lines = []

    for raw_line in user_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        words = normalize_words(line)
        if not words:
            continue

        alpha_chars = sum(char.isalpha() for char in line)
        visible_chars = sum(not char.isspace() for char in line)
        alpha_ratio = alpha_chars / visible_chars if visible_chars else 0
        keyword_hits = len(words & keyword_bank)

        if alpha_ratio < 0.55:
            continue
        if keyword_hits == 0:
            continue
        if len(words) > 12 and keyword_hits < 2:
            continue

        cleaned_lines.append(line)

    if cleaned_lines:
        return " | ".join(cleaned_lines[:3])

    return user_text.strip()


def looks_like_gibberish(text):
    words = text.split()
    if len(words) < 6:
        return True

    unique_ratio = len(set(words)) / len(words)
    long_words = [word for word in words if len(word) > 14]
    non_ascii = sum(1 for char in text if ord(char) > 127)

    if unique_ratio < 0.45:
        return True
    if len(long_words) > max(3, len(words) // 5):
        return True
    if non_ascii > 2:
        return True

    weird_tokens = 0
    for word in words:
        has_lower = any(char.islower() for char in word)
        has_upper = any(char.isupper() for char in word)
        if has_lower and has_upper and len(word) > 7:
            weird_tokens += 1
    if weird_tokens > max(2, len(words) // 6):
        return True

    return False


def find_best_template(user_text, templates):
    user_words = normalize_words(user_text)
    best_item = None
    best_score = 0

    for item in templates["conditions"]:
        keywords = {word.lower() for word in item["keywords"]}
        score = len(user_words & keywords)
        if score > best_score:
            best_score = score
            best_item = item

    return best_item, best_score


def check_guardrails(user_text):
    normalized = normalize_text(user_text)

    if any(phrase in normalized for phrase in EMERGENCY_KEYWORDS):
        return (
            "I cannot safely handle emergency symptoms in this chatbot.\n\n"
            "Please get urgent medical help now or contact local emergency services immediately.\n\n"
            "Red flag examples include chest pain, breathing trouble, fainting, seizure, overdose, vomiting blood, or thoughts of self-harm."
        )

    if any(phrase in normalized for phrase in DOSAGE_KEYWORDS):
        return (
            "I should not give exact medicine dosage, antibiotic, steroid, or prescription instructions.\n\n"
            "For safety, please ask a licensed doctor or pharmacist for exact medicine and dose guidance."
        )

    if any(phrase in normalized for phrase in HIGH_RISK_KEYWORDS):
        return (
            "Symptoms in pregnancy, newborns, infants, or older adults need extra care.\n\n"
            "I can give only very general guidance here, so please contact a real doctor for tailored advice."
        )

    return None


def build_context(template):
    if not template:
        return (
            "No matching symptom document was found. Give only general wellness advice, "
            "ask for more details, and avoid naming strong medicines."
        )

    advice = ", ".join(template["basic_care"])
    warnings = ", ".join(template["seek_help_now"])
    otc = ", ".join(template["common_otc"])
    return (
        f"Condition hint: {template['name']}. "
        f"Possible explanation: {template['summary']}. "
        f"Basic care: {advice}. "
        f"Common OTC options: {otc}. "
        f"Urgent help signs: {warnings}. "
        "Always remind that this is not a real medical diagnosis."
    )


def build_fallback_reply(user_text, template, score):
    if template and score > 0:
        care = "\n".join(f"- {item}" for item in template["basic_care"])
        otc = "\n".join(f"- {item}" for item in template["common_otc"])
        urgent = "\n".join(f"- {item}" for item in template["seek_help_now"])
        return (
            f"I found a possible match for **{template['name']}**.\n\n"
            f"{template['summary']}\n\n"
            f"Try basic care:\n{care}\n\n"
            f"Common OTC or simple support options:\n{otc}\n\n"
            f"Get urgent medical help if you notice:\n{urgent}\n\n"
            "Avoid antibiotics, steroids, or exact dosage decisions without a real clinician.\n\n"
            "This is only a lightweight symptom guide, not a doctor."
        )

    return (
        "I could not match your symptoms strongly, but I can still help a little.\n\n"
        "Please describe:\n"
        "- Your main symptoms\n"
        "- How long they have been happening\n"
        "- Your age group\n"
        "- Fever, breathing trouble, or severe pain if any\n\n"
        "If symptoms are severe or sudden, contact a real doctor quickly."
    )


def extract_answer(generated_text):
    answer = generated_text.split("Assistant:", 1)[-1].strip()
    # Some models echo speaker tags repeatedly; keep only the first assistant block.
    answer = answer.split("User:", 1)[0].strip()
    return answer


def is_off_topic_response(answer, user_text, template):
    lower_answer = answer.lower()

    blocked_phrases = [
        "no response from your assistant",
        "contact us at",
        "help center",
        "support@",
        "within 24 hours",
    ]
    if any(phrase in lower_answer for phrase in blocked_phrases):
        return True

    answer_words = normalize_words(answer)
    if len(answer_words) < 8:
        return True

    topic_words = normalize_words(user_text)
    medical_anchor_words = {
        "symptom",
        "fever",
        "cough",
        "pain",
        "care",
        "rest",
        "hydration",
        "doctor",
        "medical",
        "urgent",
    }
    if template:
        topic_words |= {word.lower() for word in template.get("keywords", [])}
        topic_words |= normalize_words(template.get("name", ""))

    overlap = len(answer_words & topic_words)
    anchor_overlap = len(answer_words & medical_anchor_words)

    # Reject responses that do not stay on medical/symptom topic.
    if overlap == 0 and anchor_overlap < 2:
        return True

    return False


def generate_reply(user_text, generator, template, score):
    guardrail_message = check_guardrails(user_text)
    if guardrail_message:
        return guardrail_message

    if generator is None:
        return build_fallback_reply(user_text, template, score)

    context = build_context(template)
    prompt = (
        "You are a medical helper chatbot for educational use.\n"
        "Use simple English.\n"
        "Do not claim to be a doctor.\n"
        "Do not prescribe dangerous medicine.\n"
        "Do not give exact dosage.\n"
        "Do not suggest antibiotics, steroids, or prescription medicines.\n"
        "If emergency symptoms appear, tell the user to seek urgent care immediately.\n"
        "Keep the answer short and practical.\n"
        "Stay consistent with the provided condition context.\n"
        "If unsure, ask for details instead of inventing facts.\n\n"
        f"{context}\n\n"
        f"User symptoms: {user_text}\n"
        "Assistant:"
    )

    try:
        for attempt in range(AUTO_RETRY_ATTEMPTS + 1):
            if attempt == 0:
                result = generator(
                    prompt,
                    do_sample=False,
                    max_new_tokens=GENERATION_SETTINGS["max_new_tokens"],
                    repetition_penalty=GENERATION_SETTINGS["repetition_penalty"],
                    num_return_sequences=1,
                )[0]["generated_text"]
            else:
                result = generator(
                    prompt,
                    do_sample=True,
                    temperature=GENERATION_SETTINGS["temperature"],
                    top_p=GENERATION_SETTINGS["top_p"],
                    max_new_tokens=GENERATION_SETTINGS["max_new_tokens"],
                    repetition_penalty=GENERATION_SETTINGS["repetition_penalty"],
                    num_return_sequences=1,
                )[0]["generated_text"]

            answer = extract_answer(result)
            if (
                len(answer) >= 20
                and not looks_like_gibberish(answer)
                and not is_off_topic_response(answer, user_text, template)
            ):
                return answer

        raise ValueError("Weak output after retries")
    except Exception:
        return build_fallback_reply(user_text, template, score)


def main():
    st.set_page_config(page_title="Medical Chatbot", layout="centered")

    st.title("Medical Chatbot")
    st.warning(
        "This project is only for learning and basic guidance. It is not medical advice or a diagnosis."
    )

    with st.sidebar:
        st.subheader("Inference Mode")
        st.write("Automatic model selection with strict safety fallback")

        st.subheader("How It Works")
        st.write(
            "1. Reads symptom templates from `medical_templates.json`\n"
            "2. Matches user symptoms by keywords\n"
            "3. Tries an automatic model and strictly validates output\n"
            "4. Falls back to trusted template guidance when output is weak/off-topic"
        )

        st.subheader("Guardrails")
        st.write(
            "- Blocks emergency handling\n"
            "- Avoids exact dosage advice\n"
            "- Avoids antibiotics and prescription guidance\n"
            "- Warns for pregnancy, infants, and older adults"
        )

    templates = load_templates()
    generator, active_model = load_generator()

    if generator is None:
        st.info(
            "Model is unavailable right now. Running in trusted template-guidance mode."
        )
    else:
        st.caption(f"Auto-selected model: {active_model}")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Tell me your symptoms in simple words, for example: "
                    "`fever, cough, sore throat for 2 days`."
                ),
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_text = st.chat_input("Describe symptoms here...")
    if not user_text:
        return

    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    cleaned_user_text = extract_relevant_symptoms(user_text, templates)
    template, score = find_best_template(cleaned_user_text, templates)
    reply = generate_reply(
        cleaned_user_text,
        generator,
        template,
        score,
    )

    with st.chat_message("assistant"):
        if cleaned_user_text != user_text.strip():
            st.caption(f"Using cleaned symptoms: {cleaned_user_text}")
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
