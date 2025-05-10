import tensorflow as tf
import numpy as np

def load_dataset(x_path, y_path, batch_size=32, shuffle=True):
    # Verwendet mmap_mode, um nicht alles auf einmal in den RAM zu laden
    X = np.load(x_path, mmap_mode='r')
    y = np.load(y_path, mmap_mode='r')

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
