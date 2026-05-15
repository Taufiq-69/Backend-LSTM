from flask import Flask, jsonify, request
import time
import numpy as np
import requests
import joblib
from datetime import datetime
import os

print("[INFO] Backend starting...")

app = Flask(__name__)

# ================= CONFIG =================
FIREBASE_URL = "https://smart-agriculture-berbasis-ai-default-rtdb.asia-southeast1.firebasedatabase.app"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "lstm_soil_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.save")

WINDOW = 20
PUMP_WINDOW = 240
TIMEOUT = 10

SECRET_KEY = "12345"

# ================= SESSION =================
session = requests.Session()

# ================= DEBUG FILE =================
print("[INFO] MODEL_PATH :", MODEL_PATH)
print("[INFO] SCALER_PATH:", SCALER_PATH)

print("[INFO] Model exists :", os.path.exists(MODEL_PATH))
print("[INFO] Scaler exists:", os.path.exists(SCALER_PATH))

# ================= LOAD MODEL =================
try:
    print("[INFO] Importing TensorFlow...")
    from tensorflow.keras.models import load_model

    print("[INFO] Loading model...")
    model = load_model(MODEL_PATH)

    print("[INFO] Loading scaler...")
    scaler = joblib.load(SCALER_PATH)

    print("[INFO] Model & Scaler loaded successfully")

except Exception as e:
    print("[FATAL ERROR]", str(e))
    raise e

# ================= STATE =================
last_pump_on_time = 0
sequence = []

# ================= FIREBASE =================
def fb_get(path):
    try:
        url = f"{FIREBASE_URL}/{path}.json"

        res = session.get(url, timeout=TIMEOUT)

        if res.status_code == 200:
            return res.json()

        return None

    except Exception as e:
        print(f"[ERROR] fb_get({path}): {e}")
        return None


def fb_set(path, data):
    try:
        url = f"{FIREBASE_URL}/{path}.json"

        session.put(url, json=data, timeout=TIMEOUT)

    except Exception as e:
        print(f"[ERROR] fb_set({path}): {e}")

# ================= WINDOW =================
def load_window():
    try:
        data = fb_get("lstm/window")

        # ===== Firebase kadang simpan array jadi dict =====
        if isinstance(data, dict):
            data = list(data.values())

        if isinstance(data, list):

            # ===== Validasi item =====
            clean_data = []

            for item in data:
                if isinstance(item, dict):
                    if all(k in item for k in [
                        "soil",
                        "temp",
                        "hum",
                        "pump",
                        "time_sin",
                        "time_cos",
                        "ts"
                    ]):
                        clean_data.append(item)

            # ===== Sort berdasarkan timestamp =====
            clean_data.sort(key=lambda x: x["ts"])

            # ===== Ambil maksimal WINDOW terakhir =====
            clean_data = clean_data[-WINDOW:]

            print(f"[INFO] Loaded window: {len(clean_data)} data")

            return clean_data

        return []

    except Exception as e:
        print("[ERROR] load_window:", e)
        return []


def save_window(window):
    try:
        url = f"{FIREBASE_URL}/lstm/window.json"

        session.put(url, json=window, timeout=TIMEOUT)

    except Exception as e:
        print(f"[ERROR] save_window: {e}")


def update_window(new_data):
    global sequence

    sequence.append(new_data)

    if len(sequence) > WINDOW:
        sequence.pop(0)

    save_window(sequence)

    return sequence

# ================= PUMP =================
def update_pump_state():
    global last_pump_on_time

    pump_now = fb_get("aktuator/pompa")

    if pump_now:
        last_pump_on_time = time.time()


def get_pump_feature():
    if (time.time() - last_pump_on_time) <= PUMP_WINDOW:
        return 1

    return 0

# ================= SENSOR =================
def get_sensor_data():
    sensor = fb_get("sensor")

    if not sensor:
        return None

    try:
        soil = float(sensor.get("soil", 0))
        temp = float(sensor.get("temperature", 0))
        hum = float(sensor.get("humidity", 0))

        return soil, temp, hum

    except Exception as e:
        print("[ERROR] sensor parse:", e)
        return None

# ================= FEATURE =================
def build_feature():
    sensor = get_sensor_data()

    if not sensor:
        return None

    soil, temp, hum = sensor

    pump = get_pump_feature()

    now = datetime.now()

    minute_of_day = now.hour * 60 + now.minute

    time_sin = np.sin(2 * np.pi * minute_of_day / 1440)
    time_cos = np.cos(2 * np.pi * minute_of_day / 1440)

    return {
        "soil": soil,
        "temp": temp,
        "hum": hum,
        "pump": pump,
        "time_sin": float(time_sin),
        "time_cos": float(time_cos),
        "ts": int(time.time())
    }

# ================= PREDICT =================
def predict(sequence_data):

    print("[DEBUG] sequence_data len:", len(sequence_data))

    input_data = np.array(sequence_data, dtype=np.float32)

    print("[DEBUG] input_data shape:", input_data.shape)

    # ===== VALIDASI SHAPE =====
    if input_data.shape != (WINDOW, 6):
        raise ValueError(
            f"Invalid input shape {input_data.shape}, expected {(WINDOW, 6)}"
        )

    # ===== SCALE =====
    input_scaled = scaler.transform(input_data)

    print("[DEBUG] input_scaled shape:", input_scaled.shape)

    # ===== RESHAPE LSTM =====
    input_scaled = input_scaled.reshape(1, WINDOW, 6)

    print("[DEBUG] reshape:", input_scaled.shape)

    # ===== PREDICT =====
    pred_scaled = model.predict(input_scaled, verbose=0)[0][0]

    print("[DEBUG] pred_scaled:", pred_scaled)

    # ===== INVERSE SCALE =====
    dummy = np.zeros((1, 6))

    dummy[0, 0] = pred_scaled

    real_pred = scaler.inverse_transform(dummy)[0][0]

    real_pred = float(np.clip(real_pred, 0, 100))

    print("[DEBUG] real_pred:", real_pred)

    return real_pred

# ================= SEND =================
def send_prediction(pred):

    payload = {
        "prediction": pred,
        "timestamp": int(time.time())
    }

    fb_set("lstm", payload)

# ================= INIT =================
sequence = load_window()

print(f"[INIT] window size: {len(sequence)}")

# ================= ROUTE =================
@app.route("/")
def home():

    return jsonify({
        "status": "LSTM backend running"
    })


@app.route("/run", methods=["GET"])
def run():

    global sequence

    # ===== AUTH =====
    key = request.args.get("key")

    if key != SECRET_KEY:

        return jsonify({
            "status": "unauthorized"
        })

    print("\n==============================")
    print("[CRON] Triggered:", datetime.now())
    print("==============================")

    try:

        # ===== UPDATE PUMP =====
        update_pump_state()

        # ===== BUILD FEATURE =====
        data = build_feature()

        if not data:

            return jsonify({
                "status": "no_data"
            })

        print("[DEBUG] New data:", data)

        # ===== UPDATE WINDOW =====
        sequence = update_window(data)

        print("[DEBUG] Current window:", len(sequence))

        # ===== WARMUP =====
        if len(sequence) < WINDOW:

            return jsonify({
                "status": "warming_up",
                "buffer": len(sequence)
            })

        # ===== BUILD ARRAY =====
        seq_array = []

        for d in sequence:

            seq_array.append([
                float(d["soil"]),
                float(d["temp"]),
                float(d["hum"]),
                float(d["pump"]),
                float(d["time_sin"]),
                float(d["time_cos"])
            ])

        print("[DEBUG] seq_array shape:", np.array(seq_array).shape)

        print("[DEBUG] first data:", seq_array[0])

        print("[DEBUG] last data:", seq_array[-1])

        # ===== PREDICT =====
        pred = predict(seq_array)

        # ===== SEND =====
        send_prediction(pred)

        print("[SUCCESS] Prediction sent")

        return jsonify({
            "status": "success",
            "prediction": pred
        })

    except Exception as e:

        print("[ERROR]", str(e))

        return jsonify({
            "status": "error",
            "message": str(e)
        })

# ================= RUN =================
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    print(f"[INFO] Running on port {port}")

    app.run(
        host="0.0.0.0",
        port=port
    )
