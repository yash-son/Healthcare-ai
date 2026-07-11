# app.py
from flask import Flask, render_template, request, jsonify
import joblib
import sqlite3
import os
import requests
import time
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
import google.generativeai as genai
import json
import re
from datetime import date

# -------------------- APP SETUP --------------------
app = Flask(__name__)

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
print("✅ GEMINI_API_KEY loaded" if os.getenv("GEMINI_API_KEY") else "⚠️ GEMINI_API_KEY missing")

DB_PATH = 'data/healthcare.db'

# Load trained ML model and symptom columns
model = joblib.load('data/disease_model.pkl')
symptom_columns = joblib.load('data/symptom_columns.pkl')

# -------------------- DATABASE INIT --------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS doctors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        specialization TEXT,
        location TEXT,
        contact TEXT
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS pharmacies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        location TEXT,
        contact TEXT
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS patient_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        symptoms TEXT,
        predicted_disease TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS disease_info (
        disease TEXT PRIMARY KEY,
        description TEXT,
        selfcare TEXT
    )''')

    conn.commit()
    conn.close()

init_db()

# -------------------- GEMINI HELPERS --------------------
def _safe_extract_text(response):
    """Return text from a gemini response object (safe)."""
    text = getattr(response, "text", "") or ""
    if not text and hasattr(response, "candidates") and response.candidates:
        try:
            text = response.candidates[0].content.parts[0].text or ""
        except Exception:
            text = ""
    return text.strip()

# (extract_symptoms and disease_description unchanged from your previous version)
def extract_symptoms(user_text):
    try:
        model_ai = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = f"""
Extract medically relevant symptoms from the following text and return them
strictly as a JSON array of short lowercase terms. Example:
["fever","headache","back pain"]

Text: {user_text}
"""
        response = model_ai.generate_content(prompt)
        raw = _safe_extract_text(response)
        print("🧾 Raw Gemini extract output:", raw)

        cleaned = raw.replace("```json", "").replace("```", "").strip()
        m = re.search(r"\[.*?\]", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(0)

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return [str(s).lower().strip() for s in parsed if str(s).strip()]
        except Exception:
            tokens = re.split(r"[,\n;]+", cleaned)
            tokens = [t.strip().lower() for t in tokens if len(t.strip()) > 2]
            if tokens:
                return list(dict.fromkeys(tokens))

        print("⚠️ Falling back to simple keyword extraction for symptoms.")
        words = [w.lower().strip(".,") for w in re.findall(r"\b[a-zA-Z\-]{4,}\b", user_text)]
        return list(dict.fromkeys(words))[:10]

    except Exception as e:
        print("🚨 Gemini extraction error:", e)
        return []

def disease_description(disease):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT description, selfcare FROM disease_info WHERE disease=?", (disease,))
    row = c.fetchone()
    if row:
        conn.close()
        try:
            return {"description": row[0], "self_care": json.loads(row[1])}
        except Exception:
            return {"description": row[0], "self_care": []}

    try:
        print(f"🔎 Querying Gemini for disease info: {disease}")
        model_ai = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = f"""
Provide a concise, medically-accurate explanation for the disease "{disease}" in simple terms,
then list 3-5 practical self-care tips. Return strictly JSON:

{{ "description": "short explanation...", "self_care": ["tip1","tip2","tip3"] }}
"""
        response = model_ai.generate_content(prompt)
        raw = _safe_extract_text(response)
        print("🧾 Raw Gemini disease output:", raw)

        cleaned = raw.replace("```json", "").replace("```", "").strip()
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(0)
        try:
            info = json.loads(cleaned)
        except Exception:
            info = {"description": cleaned, "self_care": []}

        try:
            c.execute(
                "INSERT OR REPLACE INTO disease_info (disease, description, selfcare) VALUES (?, ?, ?)",
                (disease, info.get("description", ""), json.dumps(info.get("self_care", [])))
            )
            conn.commit()
        except Exception as db_e:
            print("⚠️ DB cache error:", db_e)
        conn.close()
        return info
    except Exception as e:
        print("🚨 Gemini description error:", e)
        conn.close()
        return {"description": "No description available.", "self_care": []}

# -------------------- LOCATION HELPERS --------------------
geolocator = Nominatim(user_agent="healthcare_ai_demo")

def geocode_city(city, max_retries=3):
    if not city:
        print("⚠️ No city provided for geocoding.")
        return None, None

    geolocator = Nominatim(user_agent="AI_Healthcare_App/1.0 (contact: test@example.com)")
    for attempt in range(max_retries):
        try:
            print(f"🌍 Attempting to geocode: {city} (try {attempt+1})")
            loc = geolocator.geocode(city, timeout=15)
            if loc:
                print(f"✅ Geocoded {city} → lat={loc.latitude}, lon={loc.longitude}")
                return float(loc.latitude), float(loc.longitude)
        except Exception as e:
            print(f"⚠️ Geocode error on attempt {attempt+1}: {e}")
            time.sleep(2)
    print(f"❌ Failed to geocode {city} after {max_retries} attempts.")
    return None, None

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def overpass_nearby(lat, lon, place_type="pharmacy", radius=8000, max_results=10, max_retries=2):
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except Exception:
        print("⚠️ overpass_nearby: invalid coordinates:", lat, lon)
        return []

    if place_type == "pharmacy":
        query = f"""
        [out:json][timeout:25];
        (
          node["amenity"="pharmacy"](around:{radius},{lat_f},{lon_f});
          node["shop"="chemist"](around:{radius},{lat_f},{lon_f});
        );
        out center {max_results};
        """
    else:
        query = f"""
        [out:json][timeout:25];
        (
          node["healthcare"="doctor"](around:{radius},{lat_f},{lon_f});
          node["amenity"="clinic"](around:{radius},{lat_f},{lon_f});
        );
        out center {max_results};
        """

    headers = {
        "User-Agent": "AI-Healthcare-App/1.0 (+https://example.com)",
        "Accept": "application/json"
    }

    backoff = 1
    for attempt in range(1, max_retries + 2):
        try:
            print(f"🌍 Overpass attempt {attempt}: searching {place_type} near ({lat_f},{lon_f}) radius={radius}")
            resp = requests.post(OVERPASS_URL, data={"data": query}, headers=headers, timeout=40)
            print(f"🌐 Overpass HTTP {resp.status_code}")

            text_start = resp.text[:1000].strip()
            if text_start.lower().startswith("<!doctype") or text_start.lower().startswith("<html"):
                print("⚠️ Overpass returned HTML (likely an error page). Response preview:")
                print(text_start)
                return []

            if resp.status_code != 200:
                print("⚠️ Overpass non-200 response body (preview):")
                print(text_start)
                if 500 <= resp.status_code < 600:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                return []

            try:
                data = resp.json()
            except ValueError:
                print("⚠️ Could not parse Overpass JSON. Response preview:")
                print(text_start)
                return []

            results = []
            for el in data.get("elements", [])[:max_results]:
                tags = el.get("tags", {}) or {}
                name = tags.get("name") or tags.get("operator") or "Unknown"
                addr = ", ".join(filter(None, [
                    tags.get("addr:street"),
                    tags.get("addr:city"),
                    tags.get("addr:postcode")
                ]))
                results.append({
                    "name": name,
                    "address": addr or "Address not available",
                    "lat": el.get("lat") or el.get("center", {}).get("lat"),
                    "lon": el.get("lon") or el.get("center", {}).get("lon")
                })

            print(f"✅ Parsed {len(results)} results from Overpass.")
            return results

        except requests.exceptions.RequestException as req_e:
            print(f"⚠️ Overpass request exception (attempt {attempt}):", req_e)
            time.sleep(backoff)
            backoff *= 2

    print("❌ All Overpass attempts failed.")
    return []

# -------------------- NEW: Gen-Z FEATURES --------------------

# Simple rotating wellness tips (fallback list)
WELLNESS_TIPS = [
    "Take 5 deep breaths — reset your nervous system.",
    "Stand up & stretch every hour — your back will thank you.",
    "Drink a glass of water and smile — small wins matter.",
    "Try 2 minutes of fresh air — sunlight helps mood.",
    "Write down one thing you're grateful for today.",
    "Do a gentle neck roll — relieve tension from screens.",
    "Swap one sugary drink for water today.",
    "Take 60 seconds to close your eyes and rest."
]

@app.route('/wellness_tip', methods=['GET'])
def wellness_tip():
    """Return a daily wellness tip. Rotates predictably by date (no DB needed)."""
    try:
        idx = date.today().toordinal() % len(WELLNESS_TIPS)
        tip = WELLNESS_TIPS[idx]
        return jsonify({"tip": tip})
    except Exception as e:
        print("⚠️ wellness_tip error:", e)
        return jsonify({"tip": WELLNESS_TIPS[0]})

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Smart recommendations endpoint.
    Expects JSON form data with 'disease' (optional) and returns short lifestyle tips or recipes.
    Uses Gemini with a short prompt; falls back to canned suggestions on failure.
    """
    try:
        disease = request.form.get('disease', '').strip()
        prompt = f"""
You are a friendly health assistant. Give 4 short, practical recommendations (1-2 lines each)
for recovery / lifestyle improvements tailored for someone with: "{disease}".
Return strictly as a JSON array of short strings.
"""
        model_ai = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model_ai.generate_content(prompt)
        raw = _safe_extract_text(response)
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        # try to extract array
        m = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if m:
            arr_text = m.group(0)
            try:
                arr = json.loads(arr_text)
                if isinstance(arr, list) and arr:
                    return jsonify({"recommendations": arr})
            except Exception:
                pass

        # fallback: simple heuristic parsing (commas/newlines)
        tokens = re.split(r"[\n\r]+", cleaned)
        tokens = [t.strip(" \t\"-•") for t in tokens if len(t.strip()) > 10]
        if tokens:
            return jsonify({"recommendations": tokens[:4]})

        # ultimate fallback canned tips
        fallback = [
            "Stay hydrated: drink small sips every 30 minutes.",
            "Prioritize rest — sleep helps recovery.",
            "Try warm lemon water and honey for throat discomfort.",
            "Light walks & fresh air, if you feel up to it."
        ]
        return jsonify({"recommendations": fallback})

    except Exception as e:
        print("🚨 recommend error:", e)
        return jsonify({"recommendations": [
            "Stay hydrated.",
            "Get rest and avoid heavy activity.",
            "Consult a local doctor if symptoms worsen.",
            "Simple home care: fluids, rest, light nutrition."
        ]})

# -------------------- ROUTES (existing) --------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form.get('name', '').strip()
        text_input = request.form.get('symptoms', '').strip()
        city = request.form.get('city', '').strip()
        lat = request.form.get('lat')
        lon = request.form.get('lon')

        print(f"🔹 Input | name:{name} city:{city} lat:{lat} lon:{lon}")
        print("🩺 Symptoms text:", text_input)

        symptoms = extract_symptoms(text_input)
        print("🔍 Extracted symptoms:", symptoms)

        if not symptoms:
            return render_template('index.html', error="Could not extract symptoms. Try rephrasing.")

        input_data = [1 if col in symptoms else 0 for col in symptom_columns]
        pred = model.predict([input_data])[0]
        print("✅ Predicted disease:", pred)

        disease_info = disease_description(pred)
        print("📄 Disease info fetched.")

        if not (lat and lon):
            lat, lon = geocode_city(city) if city else (None, None)

        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""INSERT INTO patient_history (name, symptoms, predicted_disease)
                         VALUES (?, ?, ?)""", (name, ', '.join(symptoms), pred))
            conn.commit()
            conn.close()
        except Exception as log_e:
            print("⚠️ Could not log history:", log_e)

        return render_template(
            'index.html',
            name=name,
            prediction=pred,
            description=disease_info.get("description"),
            selfcare=disease_info.get("self_care", []),
            city=city,
            lat=lat,
            lon=lon
        )

    except Exception as e:
        print("🚨 Prediction Error:", e)
        return render_template('index.html', error=str(e))


@app.route('/find', methods=['POST'])
def find_nearby():
    try:
        place_type = request.form.get('type')
        disease = request.form.get('disease')
        city = request.form.get('city', '').strip()
        lat = request.form.get('lat')
        lon = request.form.get('lon')

        if not city:
            city = "Mumbai"

        print(f"🔎 Find request: type={place_type}, disease={disease}, city={city}, lat={lat}, lon={lon}")

        disease_info = disease_description(disease)

        if not (lat and lon):
            lat, lon = geocode_city(city) if city else (None, None)

        print(f"📍 Geocoded {city} → lat={lat}, lon={lon}")

        if not lat or not lon:
            return render_template(
                'index.html',
                prediction=disease,
                description=disease_info.get("description"),
                selfcare=disease_info.get("self_care", []),
                error="Please enable location in browser or enter a valid city."
            )

        results = overpass_nearby(lat, lon, place_type)
        print(f"✅ Found {len(results)} results for {place_type}")

        return render_template(
            'index.html',
            prediction=disease,
            description=disease_info.get("description"),
            selfcare=disease_info.get("self_care", []),
            results=results,
            place_type=place_type,
            city=city,
            lat=lat,
            lon=lon,
            success=f"Found {len(results)} nearby {place_type}s.",
            show_results=True
        )

    except Exception as e:
        print("🚨 Find Nearby Error:", e)
        return render_template('index.html', prediction=None, error=str(e))


@app.route('/ask', methods=['POST'])
def ask_about_disease():
    try:
        disease = request.form.get('disease', '').strip()
        question = request.form.get('question', '').strip()

        if not disease or not question:
            return jsonify({"error": "Missing disease or question"}), 400

        prompt = f"""
        You are a friendly medical assistant.
        The patient has been diagnosed with "{disease}".
        They asked: "{question}".
        Answer in 2–4 sentences maximum, using simple, reassuring, and factual language.
        Avoid medical jargon or complex terminology.
        """

        model_ai = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model_ai.generate_content(prompt)
        answer = getattr(response, "text", "").strip() or "Sorry, I couldn't find an answer right now."

        return jsonify({"answer": answer})

    except Exception as e:
        print("🚨 Chat AI Error:", e)
        return jsonify({"error": str(e)})


# -------------------- MAIN --------------------
if __name__ == '__main__':
    app.run(debug=True, port=5027)
