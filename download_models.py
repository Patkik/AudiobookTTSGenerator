"""Download Kokoro model weights to models/ directory."""
import urllib.request
import os

MODELS_DIR = "models"
BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"

FILES = [
    ("kokoro-v1.0.onnx", "~320MB - full precision model"),
    ("voices-v1.0.bin",  "~27MB  - voice style vectors"),
]

def download(filename: str, description: str) -> None:
    dest = os.path.join(MODELS_DIR, filename)
    if os.path.exists(dest):
        print(f"  ✓ {filename} already present")
        return
    url = f"{BASE_URL}/{filename}"
    print(f"  ↓ Downloading {filename} ({description})...")
    urllib.request.urlretrieve(url, dest)
    print(f"  ✓ {filename} saved to {dest}")

if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, desc in FILES:
        download(name, desc)
    print("\nDone. Model files ready in models/")
