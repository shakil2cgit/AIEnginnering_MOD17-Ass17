# Module 17 Assignment: YOLOv11 REST API & Dockerization

**Course:** Ostad AI Engineers Mastercourse  
**Topic:** Object Detection Inference Pipeline, FastAPI REST API, Automated Testing & Docker Containerization  
**Model:** YOLOv11 (Ultralytics)

---

## 📌 Project Overview (প্রজেক্ট সামারি)

এই প্রজেক্টে **YOLOv11** অবজেক্ট ডিটেকশন মডেলকে সার্ভিস হিসেবে সার্ভ করার জন্য **FastAPI** দিয়ে একটি হাই-পারফরম্যান্স **REST API** তৈরি করা হয়েছে এবং এটিকে সম্পূর্ণভাবে **Dockerize** করা হয়েছে। 

### 🌟 Key Features (প্রধান সুবিধাসমূহ):
1. **YOLOv11 Inference Pipeline:** ডিনোমিনেশন/অবজেক্ট ডিটেকশন, কনফিডেন্স স্কোর ও বাউন্ডিং বক্স `[xmin, ymin, xmax, ymax]` এক্সট্র্যাক্ট করে।
2. **FastAPI REST API:** `/predict` (POST) এন্ডপয়েন্ট তৈরি করা হয়েছে যা Multipart Image Upload সমর্থন করে।
3. **Robust Input Validation & Error Handling:** কারাপ্টেড পিকচার, ভুল ফাইল ফরম্যাট বা খালি ফাইল পাঠানোর ক্ষেত্রে যথোপযুক্ত HTTP Status Code (400, 415, 500) প্রদান করে।
4. **Automated Testing Suite:** ৫টি টেস্ট ইমেজের ওপর এন্ড-টু-এন্ড অটোমেটেড টেস্ট এবং নেগেটিভ এজ কেস ভ্যালিডেশন (`test_api.py`)।
5. **Production-Ready Dockerization:** রিমোট সার্ভার বা ক্লাউডে এক ক্লিকে রান করার জন্য Docker Containerization।
6. **Google Colab Ready:** কোনো লোকাল সেটআপ ছাড়াই ক্লাউডে ইন্সট্যান্ট চালনার জন্য `.ipynb` নোটবুক।

---

## 📂 Project Structure (ফোল্ডার স্ট্রাকচার)

```text
mod17/
├── app/
│   ├── __init__.py                  # Package Initializer
│   ├── main.py                      # FastAPI REST Server & Route Handlers
│   └── inference.py                 # YOLOv11 Model Inference Core Logic
├── models/
│   └── yolo11n.pt                   # Pretrained YOLOv11 Model Weights
├── test_images/                     # 5 Sample Test Images for Validation
│   ├── test1.jpg
│   ├── test2.jpg
│   ├── test3.jpg
│   ├── test4.jpg
│   └── test5.jpg
├── Dockerfile                       # Container definition for API deployment
├── requirements.txt                 # Python dependencies
├── run_demo.py                      # Standalone Single Image Inference Demo Script
├── test_api.py                      # Automated Test Runner for REST API
├── create_test_images.py            # Utility script to generate test images
├── generate_notebook.py             # Script to generate Google Colab Notebook
├── Assignment17_YOLOv11_FastAPI_Docker.ipynb # Standalone Google Colab Notebook
└── README.md                        # Project Documentation
```

---

## 🚀 Quick Start Guide (লোকাল এনভাইরনমেন্ট সেটআপ)

### ১. ডিপেন্ডেন্সি ইনস্টল করা:
```bash
pip install -r requirements.txt
```

### ২. সিঙ্গেল ইমেজ ইনফারেন্স টেস্ট (Task 1):
```bash
python run_demo.py
```

### ৩. FastAPI REST API রান করা (Task 2):
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
- API Documentation (Swagger UI): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### ৪. অটোমেটেড API টেস্ট চালানো (Task 3):
অনুরোধ করা হচ্ছে সার্ভার রান থাকা অবস্থায় নতুন টার্মিনালে রান করুন:
```bash
python test_api.py
```

---

## 🐳 Dockerization Guide (ডকার কনটেইনার চালনা - Task 4)

### ১. ডকার ইমেজ বিল্ড করা:
```bash
docker build -t yolo11-api .
```

### ২. ডকার কনটেইনার রান করা:
```bash
docker run -d -p 8000:8000 --name yolo11-container yolo11-api
```

### ৩. কনটেইনারের API অ্যাক্সেস ও টেস্ট:
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@test_images/test1.jpg"
```

---

## 💻 Google Colab Execution (গুগল কোলাব টিউটোরিয়াল)

1. `Assignment17_YOLOv11_FastAPI_Docker.ipynb` ফাইলটি Google Colab এ আপলোড করুন।
2. **Runtime -> Run all** নির্বাচন করুন। নোটবুকটিতে প্রতিটি সেলে বাংলা হেডিং ও বিস্তারিত ব্যাখ্যা দেওয়া আছে।

---

## 📡 API Specification & Sample Curl Requests

### Endpoint: `/predict`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Body Field:** `file` (Image: JPEG / PNG / WEBP)

#### Sample Curl Request:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@test_images/test1.jpg;type=image/jpeg'
```

#### Sample Successful JSON Response (HTTP 200 OK):
```json
{
  "success": true,
  "filename": "test1.jpg",
  "processing_time_sec": 0.0452,
  "total_detections": 1,
  "predictions": [
    {
      "class_name": "person",
      "confidence": 0.8912,
      "bounding_box": {
        "xmin": 52.1,
        "ymin": 48.3,
        "xmax": 248.5,
        "ymax": 245.0
      }
    }
  ]
}
```

#### Sample Error Response (HTTP 415 Unsupported Media Type):
```json
{
  "detail": "অসমর্থিত কন্টেন্ট টাইপ 'text/plain'! শুধুমাত্র JPEG, PNG বা WEBP সমর্থন করে।"
}
```

---

## ☁️ Cloud Deployment Instructions (Bonus Task)

### Deploying on Render / Railway / GCP Cloud Run:
1. প্রজেক্টের সোর্স কোড এবং `Dockerfile` গিটহাব (GitHub) রিপোজিটরিতে পুশ করুন।
2. Render (render.com) বা Railway (railway.app) এ নতুন **Web Service** ক্রিয়েট করে রিপোজিটরি কানেক্ট করুন।
3. **Environment:** Docker সিলেক্ট করুন।
4. **Port:** `8000` সেট করুন এবং Deploy চাপুন।
5. সিস্টেম পাবলিকলি অ্যাক্সেসযোগ্য API Endpoint (যেমন `https://yolo11-api.onrender.com/predict`) প্রোভাইড করবে।
