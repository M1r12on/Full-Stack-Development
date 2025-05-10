import json
import matplotlib.pyplot as plt

with open("model/history.json", "r") as f:
    history = json.load(f)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['loss'], label="Train Loss")
plt.plot(history['val_loss'], label="Val Loss")
plt.title("Loss Verlauf")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label="Train Acc")
plt.plot(history['val_accuracy'], label="Val Acc")
plt.title("Accuracy Verlauf")
plt.legend()

plt.tight_layout()
plt.savefig("model/training_plot.png")
plt.show()
