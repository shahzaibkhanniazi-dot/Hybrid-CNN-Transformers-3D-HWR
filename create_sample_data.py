import os
import json
import numpy as np

# Automatically create the required subfolders
target_dir = os.path.join("data", "OnHW", "data")
os.makedirs(target_dir, exist_ok=True)

sample_words = ["hello", "world", "superior", "thesis", "transformer", "sensor", "pen", "signal"]
annotations_train = []
annotations_val = []

for i in range(20):
    word = sample_words[i % len(sample_words)]
    filename = f"sample_{i:04d}.csv"
    filepath = os.path.join(target_dir, filename)
    
    # Generate 13-channel random kinematic data (150 to 300 timesteps)
    t_steps = np.random.randint(150, 300)
    fake_imu_data = np.random.randn(t_steps, 13)
    
    # Save CSV file with semicolon delimiter
    np.savetxt(filepath, fake_imu_data, delimiter=';', fmt='%.4f')
    
    meta = {
        "filename": filename,
        "label": word,
        "id_writer": (i % 4) + 1
    }
    
    if i < 16:
        annotations_train.append(meta)
    else:
        annotations_val.append(meta)

# Save JSON annotations
with open(os.path.join("data", "OnHW", "train.json"), "w", encoding="utf-8") as f:
    json.dump({"annotations": annotations_train}, f, indent=2)

with open(os.path.join("data", "OnHW", "val.json"), "w", encoding="utf-8") as f:
    json.dump({"annotations": annotations_val}, f, indent=2)

print("[+] Done! All folders, train.json, val.json, and CSV data files have been created inside data/OnHW/")