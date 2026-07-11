from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import sqlite3
import os
import requests
import time
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
import google.generativeai as genai
import json

# -------------------- APP SETUP --------------------
app = Flask(__name__)

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

DB_PATH = 'data/healthcare.db'

# Load trained ML model
model = joblib.load('data/disease_model.pkl')
symptom_columns = joblib.load('data/symptom_columns.pkl')

# -------------------- DATABASE INIT --------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Doctors table
    c.execute('''CREATE TABLE IF NOT EXISTS doctors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        specialization TEXT,
        location TEXT,
        contact TEXT
    )''')

    # Pharmacies table
    c.execute('''CREATE TABLE IF NOT EXISTS pharmacies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        location TEXT,
        contact TEXT
    )''')

    # Patient history
    c.execute('''CREATE TABLE IF NOT EXISTS patient_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        symptoms TEXT,
        predicted_disease TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')

    # Disease info cache
    c.execute('''CREATE TABLE IF NOT EXISTS disease_info (
        disease TEXT PRIMARY KEY,
        description TEXT,
        selfcare TEXT
    )''')

    conn.commit()
    conn.close()

init_db()

# -------------------- GEMINI HELPERS --------------------
def extract_symptoms(user_text):
    """Extract symptoms from free-text user input using Gemini"""
    try:
        model_ai = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = f"""
        Extract all medically relevant symptoms mentioned in this text.
        Return them as a JSON array of lowercase short terms like ["fever","headache","nausea"].
        Text: {user_text}
        """
        response = model_ai.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        print("Gemini extraction error:", e)
        return []

def disease_description(disease):
    """Fetch disease info from cache or generate with Gemini."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # ✅ Step 1: Check cache
    c.execute("SELECT description, selfcare FROM disease_info WHERE disease=?", (disease,))
    row = c.fetchone()
    if row:
        conn.close()
        return {"description": row[0], "self_care": json.loads(row[1])}

    # ✅ Step 2: Call Gemini API
    try:
        print(f"🔎 Querying Gemini for: {disease}")
        model_ai = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = f"""
        Provide a short, medically accurate explanation for the disease "{disease}" in simple terms.
        Then list 3–5 practical self-care or lifestyle tips.
        Respond strictly in JSON like this:
        {{
          "description": "short explanation...",
          "self_care": ["tip1", "tip2", "tip3"]
        }}
        """
        response = model_ai.generate_content(prompt)

        # ✅ Step 3: Extract text safely
        text = getattr(response, "text", "")
        if not text and hasattr(response, "candidates") and response.candidates:
            text = response.candidates[0].content.parts[0].text or ""

        print("🧾 Raw Gemini Output:", text)  # <— This will show what Gemini actually sends back

        # ✅ Step 4: Clean the response
        text = text.strip().replace("```json", "").replace("```", "")

        # ✅ Step 5: Try parsing JSON
        try:
            info = json.loads(text)
        except json.JSONDecodeError:
            # fallback if it's plain text
            info = {
                "description": text.strip(),
                "self_care": []
            }

        # ✅ Step 6: Save result in DB cache
        c.execute(
            "INSERT OR REPLACE INTO disease_info (disease, description, selfcare) VALUES (?, ?, ?)",
            (disease, info.get("description"), json.dumps(info.get("self_care", [])))
        )
        conn.commit()
        conn.close()
        return info

    except Exception as e:
        print("🚨 Gemini description error:", e)
        conn.close()
        return {"description": "No description available.", "self_care": []}


# -------------------- OSM HELPERS --------------------
geolocator = Nominatim(user_agent="healthcare_ai_demo")

def geocode_city(city):
    try:
        loc = geolocator.geocode(city, timeout=10)
        time.sleep(1)
        if loc:
            return float(loc.latitude), float(loc.longitude)
    except Exception as e:
        print("Geocode error:", e)
    return None, None

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def overpass_nearby(lat, lon, place_type="pharmacy", radius=4000, max_results=5):
    if place_type == "pharmacy":
        query = f"""
        [out:json][timeout:25];
        (
          node["amenity"="pharmacy"](around:{radius},{lat},{lon});
          node["shop"="chemist"](around:{radius},{lat},{lon});
        );
        out center {max_results};
        """
    else:
        query = f"""
        [out:json][timeout:25];
        (
          node["healthcare"="doctor"](around:{radius},{lat},{lon});
          node["amenity"="clinic"](around:{radius},{lat},{lon});
        );
        out center {max_results};
        """
    try:
        r = requests.post(OVERPASS_URL, data={"data": query}, timeout=30)
        data = r.json()
        results = []
        for el in data.get("elements", [])[:max_results]:
            tags = el.get("tags", {})
            name = tags.get("name") or tags.get("operator") or "Unknown"
            addr = ", ".join(filter(None, [
                tags.get("addr:street"),
                tags.get("addr:city"),
                tags.get("addr:postcode")
            ]))
            results.append({"name": name, "address": addr})
        return results
    except Exception as e:
        print("Overpass error:", e)
        return []

# -------------------- ROUTES --------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    text_input = request.form['symptoms']
    city = request.form.get('city', '').strip()
    lat = request.form.get('lat')
    lon = request.form.get('lon')

    # Step 1: Extract symptoms from text
    symptoms = extract_symptoms(text_input)
    input_data = [1 if col in symptoms else 0 for col in symptom_columns]
    pred = model.predict([input_data])[0]

    # Step 2: Get description + self-care from Gemini (with caching)
    disease_info = disease_description(pred)

    # Step 3: Get location coordinates
    if not (lat and lon):
        lat, lon = geocode_city(city) if city else (None, None)

    # Step 4: Get nearby results
    pharmacies = overpass_nearby(lat, lon, "pharmacy") if lat and lon else []
    doctors = overpass_nearby(lat, lon, "doctor") if lat and lon else []

    # Step 5: Log patient interaction
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO patient_history (name, symptoms, predicted_disease)
                 VALUES (?, ?, ?)""", (name, ', '.join(symptoms), pred))
    conn.commit()
    conn.close()

    return render_template(
        'index.html',
        name=name,
        prediction=pred,
        description=disease_info.get("description"),
        selfcare=disease_info.get("self_care", []),
        doctors=doctors,
        pharmacies=pharmacies
    )
5
# -------------------- MAIN --------------------
if __name__ == '__main__':
    app.run(debug=True, port=5009)

