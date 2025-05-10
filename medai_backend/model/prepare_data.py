import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import preprocess_input

# === CONFIG ===
IMG_DIR_1 = "data/HAM10000_images_part_1"
IMG_DIR_2 = "data/HAM10000_images_part_2"
CSV_PATH = "data/HAM10000_metadata.csv"
OUT_DIR = "preprocessed"
IMG_SIZE = (224, 224)
CLASSES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

print("Lade Metadaten...")
df = pd.read_csv(CSV_PATH)
df = df[df['dx'].isin(CLASSES)]

# → explizites Label-Mapping
label_mapping = {name: idx for idx, name in enumerate(CLASSES)}
df['label'] = df['dx'].map(label_mapping)

# Zeige Mapping zur Kontrolle
print("Label-Mapping:")
print(label_mapping)

# Bildpfade zuordnen
def resolve_path(image_id):
    filename = image_id + ".jpg"
    path1 = os.path.join(IMG_DIR_1, filename)
    path2 = os.path.join(IMG_DIR_2, filename)
    return path1 if os.path.exists(path1) else path2

df['path'] = df['image_id'].map(resolve_path)

# Verteilung anzeigen
print("Klassenverteilung:")
counts = df['dx'].value_counts()
print(counts)

# Balkendiagramm speichern
counts.plot(kind='bar', title="Verteilung der Klassen (dx)")
plt.xlabel("Klasse")
plt.ylabel("Anzahl")
plt.tight_layout()
plt.savefig("model/class_distribution.png")
plt.close()

# Daten splitten
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

def preprocess_and_store(subset_df, out_x_path, out_y_path, label=""):
    print(f"Verarbeite Bilder für {label}...")
    X, y = [], []
    skipped = 0

    for _, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc=label):
        try:
            img = Image.open(row['path']).convert('RGB')
            img = img.resize(IMG_SIZE)
            img = preprocess_input(np.array(img))
            X.append(img)
            y.append(row['label'])
        except FileNotFoundError:
            skipped += 1
            print(f"Datei nicht gefunden: {row['path']}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # Shuffle synchron (X und y gemeinsam)
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    np.save(out_x_path, X)
    np.save(out_y_path, y)
    print(f"{label}: {len(X)} Bilder gespeichert. {skipped} übersprungen.")

os.makedirs(OUT_DIR, exist_ok=True)
preprocess_and_store(train_df, f"{OUT_DIR}/X_train.npy", f"{OUT_DIR}/y_train.npy", "train")
preprocess_and_store(val_df, f"{OUT_DIR}/X_val.npy", f"{OUT_DIR}/y_val.npy", "val")

# Label-Check zur Sicherheit
print("\nLabelverteilung im Val-Set:")
print(np.bincount(val_df['label']))
print("Daten gespeichert in 'preprocessed/'")
