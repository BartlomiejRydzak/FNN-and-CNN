import os
import json

os.environ['TF_USE_LEGACY_KERAS'] = '1'

def init_tf_config():
    """
    Initializes TF_CONFIG based on TASK_TYPE and TASK_INDEX environment variables.
    Required for MultiWorkerMirroredStrategy.
    
    Run on each machine with:
    - Machine 1 (chief): TASK_TYPE=chief TASK_INDEX=0 python script.py
    - Machine 2 (worker): TASK_TYPE=worker TASK_INDEX=0 python script.py
    
    Or set USE_DISTRIBUTED=0 for single-machine testing
    """
    use_distributed = os.environ.get("USE_DISTRIBUTED", "1") == "1"
    
    if not use_distributed:
        print("✓ Running in single-machine mode (no distributed training)")
        return None
    
    task_type = os.environ.get("TASK_TYPE", "chief")
    task_index = int(os.environ.get("TASK_INDEX", 0))
    
    tf_config = {
        "cluster": {
            "chief": ["192.168.100.4:12345"],
            "worker": ["192.168.100.41:12345"]
        },
        "task": {
            "type": task_type,
            "index": task_index
        }
    }
    
    os.environ["TF_CONFIG"] = json.dumps(tf_config)
    print(f"TF_CONFIG set for {task_type}:{task_index}")
    return True

is_distributed = init_tf_config()

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, classification_report, precision_score
import seaborn as sns
import random as rndm

print(f"TensorFlow version: {tf.__version__}")

if is_distributed:
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    print(f"Using MultiWorkerMirroredStrategy with {strategy.num_replicas_in_sync} replicas")
else:
    strategy = tf.distribute.get_strategy()
    print("Using default strategy (single machine)")

os.makedirs("cnn", exist_ok=True)

class cnn:
    def __init__(self):
        self.model = None
        np.random.seed(42)
        tf.random.set_seed(42)
        rndm.seed(42)

    def train(self, name="cnn_cats_dogs.h5", batch_size=32, epochs=25):
        # Adjust batch size for distributed training
        num_replicas = strategy.num_replicas_in_sync
        if batch_size < num_replicas:
            print(f"⚠ batch_size ({batch_size}) < num_replicas ({num_replicas}). Adjusting to {num_replicas}.")
            batch_size = num_replicas

        train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

        training_set = train_datagen.flow_from_directory(
                "dataset/training_set",
                target_size=(64, 64),
                batch_size=batch_size,
                class_mode='binary',
                seed=42,
                shuffle=True)

        test_datagen = ImageDataGenerator(rescale=1./255)

        test_set = test_datagen.flow_from_directory(
                "dataset/test_set",
                target_size=(64, 64),
                batch_size=batch_size,
                class_mode='binary',
                shuffle=False,
                seed=42)

        # Build model within strategy scope
        with strategy.scope():
            cnn_model = tf.keras.models.Sequential()

            cnn_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=[64, 64, 3]))
            cnn_model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

            cnn_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
            cnn_model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

            cnn_model.add(tf.keras.layers.Flatten())
            cnn_model.add(tf.keras.layers.Dense(units=128, activation="relu"))
            cnn_model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

            cnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # Train the model
        cnn_model.fit(x=training_set, validation_data=test_set, epochs=epochs)

        # Save model (only chief saves in distributed training)
        if is_distributed:
            task_type = os.environ.get("TASK_TYPE", "chief")
            if task_type == "chief":
                cnn_model.save(name)
                print(f"✓ Model saved by chief worker: {name}")
        else:
            cnn_model.save(name)

        self.model = cnn_model

    def train_with_gpu(self):
        with tf.device("/GPU:0"):
            self.train()

    def load_model(self, name="cnn_cats_dogs.h5"):
        self.model = load_model(name)

    def predict_from_path(self, path):
        test_image = image.load_img(path, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = self.model.predict(test_image)

        if result[0][0] == 1:
            print("pies")
        else:
            print("kot")

    def predict_from_list(self, images):
        results = []

        for img in images:
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            result = self.model.predict(img)
            results.append(result)

        for result in results:
            if result[0][0] == 1:
                print("pies")
            else:
                print("kot")

    def evaluate_classification(self, batch_size=32, device_name="CPU", batch_id=None):
        """
        Ewaluacja modelu na zbiorze testowym — zapisuje confusion matrix do folderu cnn/
        """
        folder = f"cnn/{device_name}/batch_{batch_size}"
        os.makedirs(folder, exist_ok=True)

        # Przygotowanie zbioru testowego
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_set = test_datagen.flow_from_directory(
            "dataset/test_set",
            target_size=(64, 64),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )

        # Predykcja na zbiorze testowym
        y_pred_prob = self.model.predict(test_set)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        y_true = test_set.classes

        # Macierz pomyłek
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix ({device_name}, batch={batch_size}):\n", cm)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=list(test_set.class_indices.keys())))

        # --- Zapis raportu klasyfikacji do pliku ---
        report_path = f"{folder}/classification_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(
                y_true, y_pred, target_names=list(test_set.class_indices.keys())
            ))

        # Zapis wykresu
        plt.figure(figsize=(6,5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(test_set.class_indices.keys()),
            yticklabels=list(test_set.class_indices.keys())
        )
        plt.xlabel('Predykcja')
        plt.ylabel('Prawdziwa klasa')
        plt.title(f'Confusion Matrix ({device_name}, batch={batch_size})')

        # Nazwa pliku zależna od urządzenia i batcha
        file_name = f"{folder}/confusion_matrix.png"
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()


def plot_speedup(parameters, values, parameter_name='', y_label='', title='', name=" "):
    plt.figure(figsize=(10, 5))
    plt.plot(parameters, values, marker='o')
    plt.xlabel(parameter_name if parameter_name else 'Parametr')
    plt.ylabel(y_label if y_label else 'Wartość')
    plt.title(title if title else f'Wykres {y_label} w zależności od {parameter_name}')
    plt.grid(True)
    plt.savefig(f'cnn/{name}_plot.png', dpi=300, bbox_inches='tight')
    plt.close()


def test_batch_performance(batch_size, device_name="CPU"):
    my_cnn = cnn()

    # Pomiar czasu trenowania
    start_time = time.time()
    my_cnn.train(batch_size=batch_size)
    end_time = time.time()
    time_taken = end_time - start_time

    # Ewaluacja dokładności na zbiorze testowym
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_set = test_datagen.flow_from_directory(
        "dataset/test_set",
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='binary'
    )

    loss, accuracy = my_cnn.model.evaluate(test_set, verbose=0)
    y_pred_prob = my_cnn.model.predict(test_set)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = test_set.classes
    precision = precision_score(y_true, y_pred)

    print(device_name)
    print(f"Batch size: {batch_size}")
    print(f"Time taken: {time_taken:.2f} s")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")

    my_cnn.evaluate_classification(batch_size=batch_size, device_name=device_name)

    return time_taken, accuracy, precision


def plot_comparison(parameters, cpu_values, gpu_values, y_label='', title='', name='comparison'):
    plt.figure(figsize=(10, 5))
    plt.plot(parameters, cpu_values, marker='o', label='CPU')
    plt.plot(parameters, gpu_values, marker='s', label='GPU')
    plt.xlabel('Batch size')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'cnn/{name}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Wersja TensorFlow:")
    print(tf.__version__)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Dostępne urządzenia:")
    for device in tf.config.list_physical_devices():
        print(device)

    batches = [48, 64, 128, 256, 512]

    # --- TEST NA CPU ---
    print("\n=== Test na CPU ===")
    batch_times_cpu = []
    batch_acc_cpu = []
    batch_prec_cpu = []

    with tf.device('/CPU:0'):
        print("Using CPU for training...")
        for b in batches:
            print(f"\nBatch size: {b}")
            t, acc, prec = test_batch_performance(b, device_name="CPU")
            batch_times_cpu.append(t)
            batch_acc_cpu.append(acc)
            batch_prec_cpu.append(prec)

    plot_speedup(
        batches, batch_times_cpu,
        parameter_name='Batch size',
        y_label='Czas wykonania (s)',
        title='Czas trenowania CNN na CPU',
        name='CNN_CPU_time'
    )
    plot_speedup(
        batches, batch_acc_cpu,
        parameter_name='Batch size',
        y_label='Dokładność (Accuracy)',
        title='Dokładność CNN na CPU',
        name='CNN_CPU_accuracy'
    )
    plot_speedup(
        batches, batch_prec_cpu,
        parameter_name='Batch size',
        y_label='Precision',
        title='Precyzja CNN na CPU',
        name='CNN_CPU_precision'
    )

    print("\n✅ Testy zakończone — wyniki zapisane jako wykresy PNG.")