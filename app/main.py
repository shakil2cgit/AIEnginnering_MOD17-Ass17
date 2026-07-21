# ==============================================================================
# ধাপ ১: প্রয়োজনীয় প্যাকেজ এবং মডিউল ইমপোর্ট করা (Import Dependencies)
# ==============================================================================
import io                                        # ইমেজ সোর্স স্ট্রিম পড়ার জন্য io
import time                                      # রিকোয়েস্ট প্রসেসিং টাইম পরিমাপ করার জন্য
from typing import Dict, Any                     # টাইপিং হিন্টস ব্যবহারের জন্য
from PIL import Image                            # ইমেজ ভ্যালিডেশনের জন্য PIL
from fastapi import FastAPI, File, UploadFile, HTTPException, status # FastAPI কোর মডিউলসমূহ
from fastapi.responses import JSONResponse       # কাস্টম JSON রেসপন্স পাঠানোর জন্য
from fastapi.middleware.cors import CORSMiddleware # ফ্রন্টএন্ড/ওয়েব ক্লায়েন্ট অ্যাক্সেসের জন্য CORS
from app.inference import run_yolo_inference     # YOLOv11 ইনফারেন্স পাইপলাইন ফাংশন

# ==============================================================================
# ধাপ ২: FastAPI অ্যাপ্লিকেশন তৈরি এবং CORS কনফিগারেশন (FastAPI Setup & CORS)
# ==============================================================================
app = FastAPI(
    title="YOLOv11 Object & Denomination Detection API",
    description="FastAPI REST Service for YOLOv11 Model Inference (Module 17 Assignment)",
    version="1.0.0"
)

# সকল ডোমেইন থেকে রিকোয়েস্ট অনুমতির জন্য CORS মিডিওয়্যার যোগ করা
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                         # যেকোনো অরিজিন থেকে API অ্যাক্সেস করার অনুমতি
    allow_credentials=True,                      # ক্রেডেনশিয়াল অনুমোদিত
    allow_methods=["*"],                         # সকল HTTP মেথড (GET, POST ইত্যাদি) অনুমোদিত
    allow_headers=["*"],                         # সকল HTTP হেডার অনুমোদিত
)

# ==============================================================================
# ধাপ ৩: হেলপার ও গ্লোবাল এক্সেপশন হ্যান্ডলার (Exception Handling & Middleware)
# ==============================================================================
ALLOWED_EXTENSIONS = {"image/jpeg", "image/jpg", "image/png", "image/webp"} # সমর্থিত ইমেজ মাইম টাইপসমূহ

@app.exception_handler(Exception)
def global_exception_handler(request, exc):
    """
    যেকোনো অপ্রত্যাশিত সার্ভার ইন্টারনাল এরর ক্যাচ করে ৫০০ রেসপন্স ফেরত দেয়।
    """
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_type": "InternalServerError",
            "message": f"সার্ভারে একটি অনাকাঙ্ক্ষিত ত্রুটি ঘটেছে: {str(exc)}"
        }
    )

# ==============================================================================
# ধাপ ৪: হেলথ এবং রুট এন্ডপয়েন্টসমূহ (Health Check & Info Endpoints)
# ==============================================================================
@app.get("/", status_code=status.HTTP_200_OK)
def root_info() -> Dict[str, Any]:
    """
    Root endpoint: API সম্পর্কিত সাধারণ তথ্য প্রদান করে।
    """
    return {
        "service": "YOLOv11 Object Detection REST API", # সার্ভিসের নাম
        "version": "1.0.0",                              # ভার্সন
        "status": "online",                              # অনলাইন স্ট্যাটাস
        "docs_url": "/docs",                             # Swagger API ডকুমেন্টেশন ইউআরএল
        "predict_endpoint": "/predict (POST)"            # প্রেডিকশন এন্ডপয়েন্ট বার্তা
    }

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, Any]:
    """
    Health check endpoint: সিস্টেম হেলথ যাচাইয়ের জন্য।
    """
    return {"status": "healthy", "model": "YOLOv11"}    # সিস্টেম স্ট্যাটাস মেসেজ

# ==============================================================================
# ধাপ ৫: অবজেক্ট ডিটেকশন মূল এন্ডপয়েন্ট /predict (Main Detection Endpoint)
# ==============================================================================
@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict_object(
    file: UploadFile = File(...)                        # ক্লায়েন্ট থেকে পাঠানো ইমেজ ফাইল গ্রহণ
) -> Dict[str, Any]:
    """
    ইমেজ ফাইল গ্রহণ করে YOLOv11 এর সাহায্যে ডিটেক্ট করা অবজেক্ট,
    কনফিডেন্স স্কোর এবং বাউন্ডিং বক্স কোঅর্ডিনেট রিটার্ন করে।
    """
    start_time = time.time()                             # রিকোয়েস্ট শুরুর সময় রেকর্ড

    # ৫.১: ইনপুট ভ্যালিডেশন - ফাইল উপস্থিত আছে কিনা চেক
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,     # ৪-শত ব্যাড রিকোয়েস্ট স্ট্যাটাস
            detail="কোনো ইমেজ ফাইল প্রদান করা হয়নি। অনুগ্রহ করে একটি ছবি আপলোড করুন।"
        )

    # ৫.২: ইনপুট ভ্যালিডেশন - ফাইল ফরম্যাট/কন্টেন্ট টাইপ চেক
    if file.content_type and file.content_type.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, # ৪১৫ মিডিয়া টাইপ এরর
            detail=f"অসমর্থিত কন্টেন্ট টাইপ '{file.content_type}'! শুধুমাত্র JPEG, PNG বা WEBP সমর্থন করে।"
        )

    # ৫.৩: ইনপুট বাইট পড়া এবং ইমেজ ভ্যালিডেশন
    try:
        contents = await file.read()                     # ইমেজের সমস্ত বাইট মেমরিতে পড়া
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, # খালি ফাইলের জন্য এরর
                detail="আপলোড করা ইমেজ ফাইলটি সম্পূর্ণ খালি (0 bytes)।"
            )

        # PIL দিয়ে ইমেজ ওপেন করে চেক করা (ছবি কারাপ্টেড কিনা জানার জন্য)
        img = Image.open(io.BytesIO(contents))
        img.verify()                                     # ইমেজের ইন্টিগ্রিটি ভ্যালিডেশন
    except Exception as err:
        if isinstance(err, HTTPException):
            raise err
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,     # ইমেজ লোড হতে ব্যর্থ হলে ৪০০ স্ট্যাটাস
            detail=f"ত্রুটিপূর্ণ বা কারাপ্টেড ইমেজ ফাইল! সঠিক ইমেজ আপলোড করুন। বিস্তারিত: {str(err)}"
        )

    # ৫.৪: YOLOv11 ইনফারেন্স পাইপলাইন চালানো
    try:
        result = run_yolo_inference(contents, conf_threshold=0.25) # ইনফারেন্স ফাংশন কল
        processing_time = round(time.time() - start_time, 4)     # প্রসেসিং সময় হিসাব (সেকেন্ডে)

        # ৫.৫: সফল JSON রেসপন্স তৈরি
        response_data = {
            "success": True,
            "filename": file.filename,                   # ইমেজের নাম
            "processing_time_sec": processing_time,     # প্রসেসিং টাইম
            "total_detections": result["total_detections"], # মোট ডিটেকশন সংখ্যা
            "predictions": result["predictions"]         # ডিটেকশন রেজাল্ট লিস্ট
        }
        return response_data

    except Exception as inference_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, # ইনফারেন্স ফেইল করলে ৫০০
            detail=f"মডেল ইনফারেন্স চলাকালে ত্রুটি ঘটেছে: {str(inference_err)}"
        )
