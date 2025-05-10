import subprocess
import sys

print("Schritt 1: Daten vorbereiten...")
subprocess.run([sys.executable, "model/prepare_data.py"], check=True)

print("Schritt 2: Training starten...")
subprocess.run([sys.executable, "model/training.py"], check=True)

print("Schritt 3: Ergebnisse visualisieren...")
subprocess.run([sys.executable, "model/visualize.py"], check=True)

print("Schritt 4: Modell evaluieren...")
subprocess.run([sys.executable, "model/evaluate.py"], check=True)
