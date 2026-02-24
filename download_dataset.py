# ── FILE: download_dataset.py ──
# Downloads the Fake and Real News Dataset from Kaggle using kagglehub
# and copies the CSV files into the dataset/ folder.

import os
import shutil
import kagglehub

# Download dataset (cached automatically after first download)
print("📥 Downloading dataset from Kaggle...")
dataset_path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
print(f"✅ Downloaded to: {dataset_path}")

# Copy CSV files into the local dataset/ folder
base_dir = os.path.dirname(os.path.abspath(__file__))
dest_dir = os.path.join(base_dir, "dataset")
os.makedirs(dest_dir, exist_ok=True)

for filename in ["Fake.csv", "True.csv"]:
    src = os.path.join(dataset_path, filename)
    dst = os.path.join(dest_dir, filename)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        size_mb = os.path.getsize(dst) / (1024 * 1024)
        print(f"✅ Copied {filename} → dataset/ ({size_mb:.1f} MB)")
    else:
        print(f"⚠️  {filename} not found in downloaded dataset")

print("\n🎉 Done! You can now run: python train_model.py")
