import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import random as rndm

# print(tf.__version__)
os.makedirs("cnn", exist_ok=True)

class cnn:
    def __init__(self):
        self.model = None
        np.random.seed(42)
        tf.random.set_seed(42)
        rndm.seed(42)

    def train(self, name="cnn_cats_dogs.h5", batch_size=32):
        # tf.keras.backend.clear_session()
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

        cnn = tf.keras.models.Sequential()

        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=[64, 64, 3]))

        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        cnn.add(tf.keras.layers.Flatten())

        cnn.add(tf.keras.layers.Dense(units=128, activation="relu"))

        cnn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

        cnn.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

        cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

        cnn.save(name)

        self.model = load_model(name)

    def train_with_gpu(self):
        with tf.device("/GPU:0"):
            self.train()

    def load_model(self, name="cnn_cats_dogs.h5"):
        self.model = load_model(name)

    def predict_from_path(self, path):
        test_image = image.load_img(path, target_size = (64, 64))
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
            # test_image = image.load_img("dataset/pies.jpg", target_size = (64, 64))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            result = self.model.predict(img)
            # training_set.class_indices
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


def test_batch_time(batch_size):
    my_cnn = cnn()
    start_time = time.time()
    my_cnn.train(batch_size=batch_size)
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken for batch size {batch_size}: {time_taken:.4f} seconds")
    return time_taken


def test_batch_accuracy(batch_size):
    my_cnn = cnn()
    my_cnn.train(batch_size=batch_size)

    # Tworzymy zbiór testowy (taki sam jak w metodzie train)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_set = test_datagen.flow_from_directory(
        "dataset/test_set",
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='binary'
    )

    # Ewaluacja modelu
    loss, accuracy = my_cnn.model.evaluate(test_set, verbose=0)
    print(f"batch size: {batch_size}")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy

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

    print(device_name)
    print(f"Batch size: {batch_size}")
    print(f"Time taken: {time_taken:.2f} s")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    my_cnn.evaluate_classification(batch_size=batch_size, device_name=device_name)


    return time_taken, accuracy

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

# if __name__ == "__main__":
#     my_cnn = cnn()
#     my_cnn.load_model()

#     images = []
#     images.append(image.load_img("dataset/pies2.jpg", target_size = (64, 64)))
#     images.append(image.load_img("dataset/kot.jpg", target_size = (64, 64)))
#     images.append(image.load_img("dataset/pies.jpg", target_size = (64, 64)))
#     images.append(image.load_img("dataset/kot2.png", target_size = (64, 64)))

#     images.append(image.load_img("dataset/kot3.jpg", target_size = (64, 64)))
#     images.append(image.load_img("dataset/kot4.jpg", target_size = (64, 64))) #zle
#     images.append(image.load_img("dataset/kot5.jpg", target_size = (64, 64)))
#     images.append(image.load_img("dataset/pies3.jpg", target_size = (64, 64)))
#     images.append(image.load_img("dataset/pies4.jpg", target_size = (64, 64)))
    
#     my_cnn.predict_from_list(images)

#     my_cnn.predict_from_path("dataset/kot.jpg")

#     # my_cnn = cnn()
#     # my_cnn.load_model("cnn_cats_dogs.h5")
#     my_cnn.evaluate_classification(batch_size=32)


if __name__ == "__main__":
    print("wersja tensorflow:")
    print(tf.__version__)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Dostępne urządzenia:")
    for device in tf.config.list_physical_devices():
        print(device)

    batches = [8, 16, 32, 64]

    # --- TEST NA GPU ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("\n=== Test na GPU ===")
        batch_times_gpu = []
        batch_acc_gpu = []

        with tf.device('/GPU:0'):
            print("Using GPU for training...")
            for b in batches:
                print(f"\nBatch size: {b}")
                t, acc = test_batch_performance(b, device_name="GPU")
                batch_times_gpu.append(t)
                batch_acc_gpu.append(acc)

        plot_speedup(
            batches, batch_times_gpu,
            parameter_name='Batch size',
            y_label='Czas wykonania (s)',
            title='Czas trenowania CNN na GPU',
            name='CNN_GPU_time'
        )
        plot_speedup(
            batches, batch_acc_gpu,
            parameter_name='Batch size',
            y_label='Dokładność (Accuracy)',
            title='Dokładność CNN na GPU',
            name='CNN_GPU_accuracy'
        )
    else:
        print("⚠️ Brak GPU — pomijam test GPU.")

    # --- TEST NA CPU ---
    print("\n=== Test na CPU ===")
    batch_times_cpu = []
    batch_acc_cpu = []

    with tf.device('/CPU:0'):
        print("Using CPU for training...")
        for b in batches:
            print(f"\nBatch size: {b}")
            t, acc = test_batch_performance(b, device_name="CPU")
            batch_times_cpu.append(t)
            batch_acc_cpu.append(acc)

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

    plot_comparison(batches, batch_times_cpu, batch_times_gpu,
                y_label='Czas wykonania (s)',
                title='Porównanie czasu trenowania CNN: CPU vs GPU',
                name='CNN_time')
    
    plot_comparison(
        batches, batch_acc_cpu, batch_acc_gpu,
        y_label='Dokładność (Accuracy)',
        title='Porównanie dokładności CNN: CPU vs GPU',
        name='CNN_accuracy'
    )



    print("\n✅ Testy zakończone — wyniki zapisane jako wykresy PNG.")

    


# wykresy te same co dla fnn
# zamiast oschylenia standardowego dac klasyfikacje


#metryki na tysiac elemetnow testowych np 
# true positive false positive lub f1 /done
#rozmiar batchu na czas i dokladnosc /done

# Rozdzialy
#teoria
#zastosowane metody
#wyniki
#dyskusja

