import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

CLASSES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

print("Lade Validierungsdaten...")
X_val = np.load("preprocessed/X_val.npy")
y_val = np.load("preprocessed/y_val.npy")

print("Lade Modell...")
model = load_model("model/skin_model.keras")

print("Berechne Vorhersagen...")
y_pred_probs = model.predict(X_val, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion Matrix
print("Erstelle Confusion Matrix...")
cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
plt.xlabel("Vorhergesagte Klasse")
plt.ylabel("Tatsächliche Klasse")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png")
plt.show()

# Classification Report
print("Klassifikationsbericht:")
report = classification_report(y_val, y_pred, target_names=CLASSES)
print(report)

with open("model/classification_report.txt", "w") as f:
    f.write(report)
print("Report gespeichert unter: model/classification_report.txt")
