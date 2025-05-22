import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model
mpath = os.path.join(".", "model", "pawdoct.joblib")
model = joblib.load(mpath)

# Fitur yang digunakan (harus sama dengan saat training)
FEATURES = [
    "anorexia",
    "muntah",
    "lemah",
    "kurang respon",
    "dehidrasi",
    "demam",
    "diare",
    "hipersevalis",
    "radang telinga",
    "batuk",
    "hidung meler",
    "gatal",
    "telinga keropeng",
    "pilek",
    "bersin2",
    "mata berair",
]

# Saran penanganan statik berdasarkan penyakit dan gejala
SUGGESTIONS = {
    "scabies": {
        "gatal": "Gunakan salep antiparasit seperti ivermectin.",
        "telinga keropeng": "Bersihkan telinga dengan antiseptik dan aplikasikan salep antiparasit.",
        "radang telinga": "Konsultasikan dengan dokter hewan untuk antibiotik dan antiparasit.",
        "batuk": "Gunakan obat batuk yang aman untuk hewan dan konsultasi lebih lanjut.",
        "hidung meler": "Bersihkan hidung dengan larutan saline untuk mencegah infeksi.",
        "demam": "Berikan obat penurun panas sesuai rekomendasi dokter hewan.",
    },
    "panleukopenia": {
        "diare": "Berikan cairan rehidrasi oral untuk mengatasi dehidrasi.",
        "demam": "Kompres hangat atau beri obat penurun panas.",
        "lemah": "Pastikan hewan cukup istirahat dan diberikan makanan bernutrisi.",
        "anorexia": "Berikan makanan cair atau suplemen untuk merangsang nafsu makan.",
        "muntah": "Kontrol muntah dengan pemberian obat anti-muntah sesuai dosis.",
    },
    "enteritis": {
        "diare": "Berikan cairan rehidrasi oral untuk mengatasi dehidrasi.",
        "muntah": "Kontrol muntah dengan obat anti-muntah dan pemberian makanan sedikit-sedikit.",
        "demam": "Pemberian obat antipiretik sesuai dosis dokter hewan.",
        "lemah": "Pastikan hewan mendapat istirahat yang cukup.",
        "kurang respon": "Konsultasi dengan dokter hewan untuk pemeriksaan lebih lanjut.",
    },
    "fcv": {
        "pilek": "Berikan obat anti-inflamasi dan pastikan hewan tetap hangat.",
        "batuk": "Gunakan obat batuk yang tepat untuk hewan dan pastikan hewan tetap terhidrasi.",
        "mata berair": "Bersihkan mata dengan larutan saline atau cairan antiseptik.",
        "bersin2": "Jaga kebersihan lingkungan hewan dan beri obat sesuai anjuran dokter hewan.",
        "hidung meler": "Bersihkan hidung dengan saline dan pastikan hewan tetap hangat.",
    },
    "_default": {
        "scabies": "Gunakan obat antiparasit topikal dan bersihkan telinga jika ada radang.",
        "panleukopenia": "Bawa ke dokter hewan untuk perawatan intensif, termasuk cairan dan antibiotik.",
        "enteritis": "Berikan cairan elektrolit dan konsultasi ke dokter hewan untuk penanganan lebih lanjut.",
        "fcv": "Pastikan hewan mendapatkan terapi suportif dan isolasi dari hewan lain.",
    },
}


def get_suggestions_by_symptoms(prediction, features):
    saran = []
    rules = SUGGESTIONS.get(prediction, {})
    for f, val in features.items():
        if val == 1 and f in rules:
            saran.append(rules[f])
    if not saran:
        saran.append(SUGGESTIONS["_default"].get(prediction, "Tidak ada saran."))
    return saran


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Validasi input
    if not data or not all(f in data for f in FEATURES):
        return jsonify({"error": f"Input harus berisi semua fitur: {FEATURES}"}), 400

    # Konversi Yes/No ke 1/0
    input_values = []
    converted_data = {}
    for f in FEATURES:
        v = data[f]
        bin_val = 1 if isinstance(v, str) and v.lower() == "yes" else 0
        input_values.append(bin_val)
        converted_data[f] = bin_val

    # Prediksi
    input_df = pd.DataFrame([converted_data])
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    # Saran
    suggestions = get_suggestions_by_symptoms(prediction, converted_data)

    response = jsonify(
        {
            "prediction": prediction,
            "probabilities": dict(zip(model.classes_, proba)),
            "features_used": converted_data,
            "suggestions": suggestions,
        }
    )

    return response


if __name__ == "__main__":
    app.run(debug=True)
