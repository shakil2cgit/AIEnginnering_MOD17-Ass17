# ==============================================================================
# ধাপ ১: প্রয়োজনীয় লাইব্রেরি ইমপোর্ট করা (Import Dependencies)
# ==============================================================================
import os                                        # ফাইল পাথ প্রসেস করার জন্য
import sys                                       # সিস্টেম এনকোডিং হ্যান্ডলিং
import json                                      # JSON রেসপন্স প্রিন্ট করার জন্য
import requests                                  # REST API রিকোয়েস্ট পাঠানোর জন্য

# UTF-8 এনকোডিং কনফিগারেশন (উইনডোজ কনসোল আউটপুটের জন্য)
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

# ==============================================================================
# ধাপ ২: কনফিগারেশন এবং টেস্ট ইমেজ লিস্ট (Configuration & Setup)
# ==============================================================================
API_URL = "http://127.0.0.1:8000/predict"        # স্থানীয় FastAPI প্রেডিকশন এন্ডপয়েন্ট
TEST_DIR = "test_images"                         # টেস্ট ইমেজের ডিরেক্টরি

TEST_FILES = [
    "test1.jpg",
    "test2.jpg",
    "test3.jpg",
    "test4.jpg",
    "test5.jpg"
]

# ==============================================================================
# ধাপ ৩: API টেস্ট এক্সিকিউশন ফাংশন (API Testing Suite Function)
# ==============================================================================
def run_api_tests():
    print("=" * 70)
    print("       FastAPI YOLOv11 REST API Automated Test Suite")
    print("=" * 70)
    print(f"Target API Endpoint: {API_URL}\n")

    # ৩.১: ৫টি টেস্ট ইমেজের ওপর এন্ড-টু-এন্ড টেস্ট
    passed_tests = 0
    total_tests = len(TEST_FILES)

    for idx, filename in enumerate(TEST_FILES, start=1):
        file_path = os.path.join(TEST_DIR, filename)
        if not os.path.exists(file_path):
            print(f"[ERROR] Test file not found: {file_path}")
            continue

        print(f"----------------------------------------------------------------------")
        print(f"Test {idx}/{total_tests}: Uploading file -> {filename}")
        print(f"----------------------------------------------------------------------")

        try:
            with open(file_path, "rb") as img_file:
                # HTTP POST Multipart Form Upload তৈরি
                files = {"file": (filename, img_file, "image/jpeg")}
                response = requests.post(API_URL, files=files, timeout=10) # API তে POST রিকোয়েস্ট পাঠানো

            print(f"HTTP Status Code: {response.status_code}")       # স্ট্যাটাস কোড প্রিন্ট
            if response.status_code == 200:
                data = response.json()
                print("API JSON Response:")
                print(json.dumps(data, indent=2, ensure_ascii=False))  # ফরম্যাট করা JSON আউটপুট
                passed_tests += 1
            else:
                print(f"[FAIL] Request failed! Response: {response.text}")

        except Exception as e:
            print(f"[EXCEPTION] Error during API test: {str(e)}")

    # ৩.২: এরর হ্যান্ডলিং টেস্ট (Edge Cases / Negative Testing)
    print("\n" + "=" * 70)
    print("       Negative Testing (Error Handling Edge Cases)")
    print("=" * 70)

    # টেস্ট ১: টেক্সট ফাইল আপলোড (Invalid Format Test)
    print("\n[Edge Case 1] Invalid file format upload (text/plain)...")
    fake_file = {"file": ("test.txt", b"Hello World", "text/plain")}
    res_err1 = requests.post(API_URL, files=fake_file)
    print(f"Status Code: {res_err1.status_code} | Response: {res_err1.json()}") # ৪১৫ স্ট্যাটাস প্রত্যাশিত

    # টেস্ট ২: শূন্য বাইট খালি ফাইল আপলোড (Empty file test)
    print("\n[Edge Case 2] Empty zero-byte file upload...")
    empty_file = {"file": ("empty.jpg", b"", "image/jpeg")}
    res_err2 = requests.post(API_URL, files=empty_file)
    print(f"Status Code: {res_err2.status_code} | Response: {res_err2.json()}") # ৪০০ স্ট্যাটাস প্রত্যাশিত

    # ৩.৩: সামারি ফলাফল প্রিন্ট
    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {passed_tests}/{total_tests} image tests PASSED successfully!")
    print("=" * 70)

if __name__ == "__main__":
    run_api_tests()                              # টেস্ট সুইট চালানো
