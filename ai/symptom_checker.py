
import re
import pandas as pd
import streamlit as st
from transformers import pipeline
import google.generativeai as genai

API_KEY = "#"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')



# Load zero-shot classification pipeline
@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_model()
labels = ["Emergency", "Routine", "Self-care"]

st.set_page_config(page_title="AI Health Assistant", page_icon="🩺", layout="centered")
st.title("🩺 AI Health Assistant")

user_input = st.text_input("Describe your symptoms:")

def clean_list(text):
    items = []
    for line in text.splitlines():
        line = line.strip()
        # Remove leading bullets, numbers, whitespace
        line = re.sub(r'^[\-\*\u2022\d\.\)\s]+', '', line)
        # Remove all Markdown bold ** and italic * and underscores _
        line = re.sub(r'(\*\*|\*|_)', '', line)
        # Remove trailing colons if any
        line = re.sub(r':$', '', line)
        if line:
            items.append(line)
    return items

def clean_disease(text):
    text = text.strip()
    # Remove leading markdown bold "**" or "*"
    text = re.sub(r'^\*\*\s*', '', text)
    text = re.sub(r'^\*\s*', '', text)
    # Remove trailing markdown bold "**" or "*"
    text = re.sub(r'\s*\*\*$','', text)
    text = re.sub(r'\s*\*$','', text)
    # Remove trailing colons if any
    text = re.sub(r':$', '', text)
    return text

if user_input:
    with st.spinner("Analyzing your symptoms..."):
        # Step 1: Classify urgency
        result = classifier(user_input, labels)
        pred_label = result['labels'][0]
        confidence = result['scores'][0]

    st.markdown(f"### Predicted Urgency: **{pred_label}** (confidence: {confidence * 100:.0f}%)")

    if pred_label == "Emergency":
        st.error("Your symptoms may require immediate medical attention. Please seek emergency care right away. Stay calm and safe.")
    elif pred_label == "Routine":
        st.info("Your symptoms suggest you should schedule an appointment with your healthcare provider soon.")
    else:
        st.success("Your symptoms appear mild and may be managed with self-care. Monitor your condition and consult a healthcare professional if symptoms worsen.")

    # Step 2: Use Gemini to generate disease, food, and medicine advice
    prompt = f"""
    You are a compassionate medical assistant.

    A user has described these symptoms:
    "{user_input}"

    Based on these symptoms, please provide:

    1. A short statement predicting the most likely disease or condition.
    2. A list of suitable foods to eat (as bullet points or numbered list).
    3. A list of general over-the-counter medicines or remedies (as bullet points or numbered list).
       Avoid prescription drugs.

    Present the answer clearly separated into three sections titled 'Disease:', 'Foods:', and 'Medicines:'.

    Use empathetic and clear language.

    End with this disclaimer:
    "This is not a medical diagnosis. Please consult a licensed healthcare professional for an accurate diagnosis and treatment."
    """

    with st.spinner("Generating disease, food, and medicine advice..."):
        try:
            response = model.generate_content(prompt)
            advice_text = response.text
        except Exception as e:
            advice_text = f"❌ API Error: {e}"

    # Parse Gemini response
    disease_match = re.search(r"Disease?:\s*(.+?)(Foods?:|$)", advice_text, re.DOTALL | re.IGNORECASE)
    foods_match = re.search(r"Foods?:\s*(.+?)(Medicines?:|$)", advice_text, re.DOTALL | re.IGNORECASE)
    medicines_match = re.search(r"Medicines?:\s*(.+)", advice_text, re.DOTALL | re.IGNORECASE)

    disease = clean_disease(disease_match.group(1)) if disease_match else "No disease prediction available."
    foods = clean_list(foods_match.group(1)) if foods_match else []
    medicines = clean_list(medicines_match.group(1)) if medicines_match else []

    st.markdown("### Predicted Disease / Condition")
    st.markdown(disease)

    st.markdown("### Recommended Foods")
    if foods:
        df_foods = pd.DataFrame(foods, columns=["Foods to Eat"])
        st.table(df_foods)
    else:
        st.write("No food suggestions available.")

    st.markdown("### Suggested Medicines / Remedies")
    if medicines:
        df_medicines = pd.DataFrame(medicines, columns=["Medicines / Remedies"])
        st.table(df_medicines)
    else:
        st.write("No medicine suggestions available.")