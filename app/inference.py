# ==============================================================================
# ধাপ ১: প্রয়োজনীয় প্যাকেজ এবং লাইব্রেরি ইমপোর্ট করা (Import Libraries)
# ==============================================================================
import os                                        # ফাইল এবং ডিরেক্টরি পাথ হ্যান্ডেল করার জন্য
import io                                        # বাইট স্ট্রিম প্রসেস করার জন্য
from typing import List, Dict, Any, Union        # টাইপ হিস্ট ব্যবহারের জন্য
from PIL import Image                            # ইমেজ ওপেন ও প্রসেস করার জন্য PIL লাইব্রেরি
import numpy as np                               # নিউমেরিক্যাল ডাটা ও এরে অপারেশনের জন্য
from ultralytics import YOLO                      # YOLOv11 অবজেক্ট ডিটেকশন মডেল লোড ও প্রসেস করার জন্য

# ==============================================================================
# ধাপ ২: গ্লোবাল ভ্যারিয়েবল এবং মডেল পাথ কনফিগারেশন (Configuration & Model Loading)
# ==============================================================================
# মডেল ফাইল খোঁজার স্থান (প্রথমে custom models/best.pt, না থাকলে yolo11n.pt ডাউনলোড হবে)
CUSTOM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best.pt")
DEFAULT_MODEL_NAME = "yolo11n.pt"                # YOLOv11 Pretrained Nano Model

_model_instance = None                           # মডেল মেমরিতে ক্যাশ রাখার জন্য গ্লোবাল ভ্যারিয়েবল

def get_yolo_model() -> YOLO:
    """
    YOLOv11 মডেল মেমরিতে লোড করে রিটার্ন করে (Singleton Pattern)।
    """
    global _model_instance
    if _model_instance is None:
        if os.path.exists(CUSTOM_MODEL_PATH):
            print(f"[INFO] কাস্টম মডেল পাওয়া গেছে: {CUSTOM_MODEL_PATH}") # কাস্টম ওয়েটস লোড বার্তা
            _model_instance = YOLO(CUSTOM_MODEL_PATH)                   # কাস্টম মডেল ওয়েটস লোড
        else:
            print(f"[INFO] ডিফল্ট YOLOv11 Nano মডেল লোড হচ্ছে: {DEFAULT_MODEL_NAME}") # ডিফল্ট মডেল বার্তা
            _model_instance = YOLO(DEFAULT_MODEL_NAME)                   # YOLOv11 Pretrained model লোড
    return _model_instance

# ==============================================================================
# ধাপ ৩: ইনফারেন্স পাইপলাইন ফাংশন (Inference Pipeline Core Function)
# ==============================================================================
def run_yolo_inference(
    image_source: Union[bytes, str, Image.Image],
    conf_threshold: float = 0.25
) -> Dict[str, Any]:
    """
    ইমেজ ইনপুট নিয়ে YOLOv11 এর মাধ্যমে অবজেক্ট ডিটেক্ট করে এবং ডিনোমিনেশন/ক্লাস,
    কনফিডেন্স স্কোর ও বাউন্ডিং বক্স রিটার্ন করে।
    """
    # ৩.১: ইনপুট ইমেজ প্রসেসিং (Image Loading & Preprocessing)
    if isinstance(image_source, bytes):
        image = Image.open(io.BytesIO(image_source)).convert("RGB") # বাইট স্ট্রিম থেকে PIL ইমেজে রূপান্তর
    elif isinstance(image_source, str):
        image = Image.open(image_source).convert("RGB")             # পাথ থেকে ইমেজ লোড করা
    elif isinstance(image_source, Image.Image):
        image = image_source.convert("RGB")                         # PIL ইমেজ ডিরেক্ট গ্রহণ করা
    else:
        raise ValueError("অকার্যকর ইমেজ ফরম্যাট! Image bytes, file path অথবা PIL Image প্রদান করুন।")

    # ৩.২: মডেল ইনফারেন্স চালানো (Model Prediction Execution)
    model = get_yolo_model()                                        # মেমরি থেকে YOLOv11 মডেল সংগ্রহ
    results = model.predict(source=image, conf=conf_threshold, verbose=False) # ইমেজে ডিটেকশন প্রসেস চালু

    # ৩.৩: ডিটেকশন রেজাল্ট এক্সট্র্যাক্ট ও ফরম্যাট করা (Extracting Prediction Output)
    predictions = []
    result = results[0]                                             # প্রথম ইমেজের ইনফারেন্স রেজাল্ট

    for box in result.boxes:
        cls_id = int(box.cls[0].item())                             # ক্লাস আইডি এক্সট্র্যাক্ট করা
        class_name = model.names[cls_id]                            # আইডির বিপরীতে ক্লাসের নাম / Denomination
        confidence = round(float(box.conf[0].item()), 4)            # কনফিডেন্স স্কোর ৪ ডেসিমাল পয়েন্টে রাউন্ড
        xyxy = box.xyxy[0].tolist()                                  # বাউন্ডিং বক্স কোঅর্ডিনেট [xmin, ymin, xmax, ymax]
        bbox = [round(coord, 2) for coord in xyxy]                  # পিক্সেল কোঅর্ডিনেট ২ ডেসিমাল রাউন্ড

        predictions.append({
            "class_name": class_name,                               # ডিনোমিনেশন বা অবজেক্টের নাম
            "confidence": confidence,                               # মডেলের কনফিডেন্স স্কোর
            "bounding_box": {                                       # বাউন্ডিং বক্স অবজেক্ট
                "xmin": bbox[0],
                "ymin": bbox[1],
                "xmax": bbox[2],
                "ymax": bbox[3]
            }
        })

    # ৩.৪: ফাইনাল আউটপুট রেসপন্স রিটার্ন (Final Output Dict)
    return {
        "success": True,                                            # সফলভাবে ইনফারেন্স শেষ নির্দেশক
        "total_detections": len(predictions),                       # মোট কয়টি অবজেক্ট ডিটেক্ট হয়েছে
        "predictions": predictions                                  # ডিটেক্ট করা অবজেক্টসমূহের লিস্ট
    }
