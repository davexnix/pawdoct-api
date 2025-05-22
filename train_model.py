# Import Library
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Menonaktifkan silent downcasting warning
pd.set_option("future.no_silent_downcasting", True)

# Definisikan dataset dan save model
dataset = os.path.join(".", "data", "dataset.csv")
savemodel = os.path.join(".", "model", "pawdoct.joblib")

# Load dataset
df = pd.read_csv(dataset)
print(f"Dataset: {dataset}")

# Konversi 'Yes' dan 'No' ke 1 dan 0
df = df.replace({"Yes": 1, "No": 0}).infer_objects(copy=False)

# Pisahkan fitur dan label
X = df.drop("penyakit", axis=1)
y = df["penyakit"]

# Split data untuk evaluasi akhir
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediksi dan akurasi
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAkurasi: {acc:.2f}")

# Laporan klasifikasi
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("\nConfusion Matrix:")
print("Labels:", list(model.classes_))
print(cm)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Rata-rata cross-val: {cv_scores.mean():.2f}")

# Simpan model
joblib.dump(model, savemodel)
print(f"\n✅ Model disimpan sebagai {savemodel}")

# Fitur penting
print("\nFeature Importance:")
importances = model.feature_importances_
for feat, imp in sorted(zip(X.columns, importances), key=lambda x: -x[1]):
    print(f"{feat}: {imp:.3f}")
