# get_data.py
import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURATION ---
HF_REPO = "tianyi-in-the-bush/penncampus_image2gps_larger_merged_cleaned"
LOCAL_FOLDER = "geo_data"  # We will save everything here
IMAGES_DIR = os.path.join(LOCAL_FOLDER, "images")
CSV_PATH = os.path.join(LOCAL_FOLDER, "data.csv")

def main():
    print(f"🚀 Downloading data from {HF_REPO}...")
    
    # 1. Download the dataset from Hugging Face
    try:
        ds = load_dataset(HF_REPO, split="train")
    except Exception as e:
        print(f"❌ Error connecting to Hugging Face: {e}")
        print("Tip: If this is a private repo, run 'huggingface-cli login' in terminal first.")
        return

    # 2. Setup local folders
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    print("📦 Saving images and creating CSV...")
    
    metadata_rows = []
    
    # 3. Loop through every item, save image, and record stats
    for idx, item in tqdm(enumerate(ds), total=len(ds)):
        # Create a unique filename for the image
        image_filename = f"img_{idx:05d}.jpg"
        full_image_path = os.path.join(IMAGES_DIR, image_filename)
        
        # Save the image to disk
        # (Assuming the dataset has an 'image' column which is a PIL object)
        if 'image' in item:
            item['image'].save(full_image_path)
        else:
            print(f"⚠️ Warning: No image found for item {idx}")
            continue

        # Collect the coordinates (Handling different column names)
        # We look for 'lat'/'latitude' and 'lon/'longitude'
        lat = item.get('lat') or item.get('latitude') or item.get('Latitude')
        lon = item.get('lon') or item.get('longitude') or item.get('Longitude')
        
        if lat is None or lon is None:
            continue

        # Add to our list for the CSV
        metadata_rows.append({
            "image_path": full_image_path,  # This points to the file we just saved
            "lat": lat,
            "lon": lon
        })

    # 4. Save the CSV file
    df = pd.DataFrame(metadata_rows)
    df.to_csv(CSV_PATH, index=False)
    
    print(f"\n✅ SUCCESS! Data is ready.")
    print(f"   Images saved in: {IMAGES_DIR}")
    print(f"   CSV saved at:    {CSV_PATH}")

if __name__ == "__main__":
    main()