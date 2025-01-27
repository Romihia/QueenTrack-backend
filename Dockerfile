# Dockerfile 
FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# העתקת requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# העתקת קבצי הפרויקט
COPY . .

# הפעלת uvicorn (ללא reload בסביבת ייצור)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
