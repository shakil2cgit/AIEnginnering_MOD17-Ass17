# ==============================================================================
# 5টি টেস্ট ইমেজ তৈরির জন্য স্ক্রিপ্ট (Script to Generate 5 Sample Test Images)
# ==============================================================================
import os                                        # ফাইল এবং ডিরেক্টরি পাথ ব্যবহারের জন্য
from PIL import Image, ImageDraw                 # ইমেজ জেনারেশন এবং অবজেক্ট আঁকার জন্য PIL

def generate_5_test_images():
    output_dir = "test_images"
    os.makedirs(output_dir, exist_ok=True)       # test_images ফোল্ডার তৈরি না থাকলে তৈরি করা

    images_data = [
        {"name": "test1.jpg", "bg": (240, 240, 240), "shape": "rect", "color": (220, 50, 50), "bbox": [50, 50, 250, 250]},
        {"name": "test2.jpg", "bg": (220, 240, 220), "shape": "circle", "color": (50, 180, 50), "bbox": [100, 100, 300, 300]},
        {"name": "test3.jpg", "bg": (240, 220, 240), "shape": "rect", "color": (50, 50, 220), "bbox": [80, 80, 280, 280]},
        {"name": "test4.jpg", "bg": (255, 245, 200), "shape": "circle", "color": (230, 140, 30), "bbox": [60, 60, 240, 240]},
        {"name": "test5.jpg", "bg": (210, 230, 250), "shape": "rect", "color": (140, 50, 200), "bbox": [120, 120, 320, 320]},
    ]

    for item in images_data:
        file_path = os.path.join(output_dir, item["name"])
        img = Image.new("RGB", (400, 400), color=item["bg"]) # ইমেজের ব্যাকগ্রাউন্ড কালার দিয়ে তৈরি
        draw = ImageDraw.Draw(img)
        
        if item["shape"] == "rect":
            draw.rectangle(item["bbox"], fill=item["color"], outline=(0, 0, 0), width=2) # রেক্টেঙ্গেল আঁকা
        else:
            draw.ellipse(item["bbox"], fill=item["color"], outline=(0, 0, 0), width=2)    # এলিপস আঁকা
            
        img.save(file_path)                      # ইমেজ ফাইলে সেভ করা
        print(f"[INFO] Test image created: {file_path}")

if __name__ == "__main__":
    generate_5_test_images()                     # স্ক্রিপ্ট রান করা
