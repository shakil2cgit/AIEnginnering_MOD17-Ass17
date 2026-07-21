# ==============================================================================
# ধাপ ১: প্রয়োজনীয় প্যাকেজ ইমপোর্ট করা (Import Dependencies)
# ==============================================================================
import os                                        # ফাইল সিস্টেম পাথ ব্যবহারের জন্য
import sys                                       # সিস্টেম এনকোডিং হ্যান্ডলিং
import json                                      # রেজাল্ট সুন্দরভাবে কনসোলে প্রিন্ট করার জন্য
from PIL import Image, ImageDraw, ImageFont      # ইমেজ তৈরি এবং ড্র করার জন্য PIL
from app.inference import run_yolo_inference     # আমরা তৈরি করা YOLO ইনফারেন্স ফাংশন

# UTF-8 এনকোডিং কনফিগারেশন (উইনডোজ কনসোল আউটপুটের জন্য)
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

# ==============================================================================
# ধাপ ২: ডেমো টেস্ট ইমেজ তৈরি/চেক করা (Prepare Demo Image)
# ==============================================================================
def create_sample_demo_image(output_path: str = "demo_sample.jpg") -> str:
    """
    যদি কোনো ছবি না থাকে তবে একটি স্যাম্পল কালারফুল ইমেজ তৈরি করে সেভ করে।
    """
    if not os.path.exists(output_path):
        # একটি ৪০০x৪০০ পিক্সেলের নতুন ইমেজ তৈরি করা
        img = Image.new("RGB", (400, 400), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        # লাল রঙের একটি শেপ ড্র করা
        draw.rectangle([50, 50, 200, 200], fill=(220, 50, 50), outline=(0, 0, 0))
        # নীল রঙের একটি বৃত্ত শেপ ড্র করা
        draw.ellipse([220, 220, 350, 350], fill=(50, 100, 220), outline=(0, 0, 0))
        img.save(output_path)                   # ইমেজ সেভ করা
        print(f"[INFO] Sample demo image created: {output_path}")
    return output_path

# ==============================================================================
# ধাপ ৩: মেইন ডেমো ইনফারেন্স এক্সিকিউশন (Run Inference & Display Output)
# ==============================================================================
def main():
    print("=" * 60)
    print("      YOLOv11 Single Image Inference Demonstration")
    print("=" * 60)

    # ৩.১: স্যাম্পল ইমেজ পাথ নির্ধারণ (Set Image Path)
    image_path = create_sample_demo_image("demo_sample.jpg")

    # ৩.২: ইনফারেন্স চালানো (Execute Inference)
    print(f"\n[Step 1] Running YOLOv11 inference on: {image_path}...")
    result = run_yolo_inference(image_path, conf_threshold=0.25) # ইনফারেন্স পাইপলাইন কল

    # ৩.৩: আউটপুট কনসোলে প্রিন্ট করা (Print JSON Output)
    print("\n[Step 2] Inference Output Response (JSON):")
    print(json.dumps(result, indent=2, ensure_ascii=False))     # ফরম্যাট করা JSON আউটপুট

    # ৩.৪: ভিজ্যুয়ালাইজেশন সেভ করা (Save Detection Visualization)
    if result["total_detections"] > 0:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for pred in result["predictions"]:
            bbox = pred["bounding_box"]
            label = f"{pred['class_name']} ({pred['confidence']})"
            draw.rectangle([bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]], outline="red", width=3)
            draw.text((bbox["xmin"], max(0, bbox["ymin"] - 10)), label, fill="red")
        
        vis_path = "demo_detection_result.jpg"
        img.save(vis_path)                                      # ভিজ্যুয়ালাইজড ইমেজ সেভ
        print(f"\n[Step 3] Visualized image with detections saved at: {vis_path}")
    else:
        print("\n[Step 3] No objects crossed confidence threshold (Pretrained Model Test Image).")

    print("\n[SUCCESS] Single Image Inference Pipeline completed successfully!")

if __name__ == "__main__":
    main()                                                      # প্রোগ্রাম রান করা
