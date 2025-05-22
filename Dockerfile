# Gunakan base image Python
FROM python:3.11-slim

# Atur direktori kerja di container
WORKDIR /app

# Salin file ke container
COPY requirements.txt .
COPY app.py .
COPY pawdoct.joblib .

# Install dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Jalankan aplikasi Flask
EXPOSE 5000
CMD ["python3", "app.py"]
