# download_model.py
import urllib.request
import os

model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
model_path = "models/sam_vit_h_4b8939.pth"

if not os.path.exists(model_path):
    print("⏳ Downloading SAM model (2.4GB)...")
    print("This may take 5-10 minutes...")
    urllib.request.urlretrieve(model_url, model_path)
    print("✅ Download complete!")
else:
    print("✅ Model already exists!")