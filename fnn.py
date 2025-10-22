import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import random as rndm
import random
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error
import time

class fnn:
    def __init__(self):
        #function parameters
        self.x0 = None
        self.x1 = None
        self.x2 = None
        self.x3 = None
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.rand = None
        #model parmeters
        self.x = None
        self.y = None
        self.X = None
        self.Y = None
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_all_scaled = None
        self.y_pred = None
        #model
        self.model = None
        #metrics
        self.history = None
        self.mae = None
        self.loss = None
    
    # zmiana noise na 3.0 z 1.0
    def set_parameters(self, start, end, step, x0, x1, x2, x3, alpha, beta, gamma, noise=3.0):
        np.random.seed(42)
        tf.random.set_seed(42)
        rndm.seed(42)

        self.x = np.arange(start, end, step)
        # x0 = random.randint(-5, 5)
        # x1 = random.randint(-5, 5)
        # x2 = random.randint(-5, 5)
        # x3 = random.randint(-500, 500)
        self.x0 = x0
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rand = np.random.normal(loc=0.0, scale=noise, size=len(self.x))

    def define_function(self):
        self.y = self.alpha * self.x0 * np.sin(self.x) + self.beta * self.x2 * self.x3 * self.x*self.x + self.gamma * np.abs(self.x0 - self.x2) + self.rand

        #todo
        self.X = self.x.reshape(-1, 1)
        self.Y = self.y.reshape(-1, 1)
    
    def scale_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.X_train = self.x_scaler.fit_transform(self.X_train.reshape(-1, 1))
        self.X_test = self.x_scaler.transform(self.X_test.reshape(-1, 1))
        self.y_train = self.y_scaler.fit_transform(self.y_train.reshape(-1, 1))
        self.y_test = self.y_scaler.transform(self.y_test.reshape(-1, 1))

    # zmiana batch_size na 128 z 32
    def create_model(self, epochs=100, batch_size=128):
        # pierwsza wersja
        # self.model = tf.keras.Sequential([
        #     tf.keras.layers.Dense(80, activation='relu', input_shape=(1,)),
        #     tf.keras.layers.Dense(80, activation='relu'),
        #     tf.keras.layers.Dense(20, activation='relu'),
        #     tf.keras.layers.Dense(20, activation='relu'),
        #     tf.keras.layers.Dense(1)
        # ])

        # wersja porownawcza
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(1000, activation='relu', input_shape=(1,)),
            tf.keras.layers.Dense(400, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])


        # log_dir = "./logs"
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(
        #     log_dir=log_dir,
        #     histogram_freq=1,
        #     profile_batch='10,30'
        # )
        
        # self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0, callbacks=[tensorboard_callback])
        self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)


        self.loss, self.mae = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        # print(f"Test MAE: {self.mae:.4f}")

        self.X_all_scaled = self.x_scaler.transform(self.X)
        self.y_pred = self.model.predict(self.X_all_scaled)
        # self.y_pred = self.y_scaler.inverse_transform(self.y_pred.reshape(-1, 1))
    
    def show_stats(self):
        mae = mean_absolute_error(self.y_test, self.model.predict(self.X_test))
        mse = mean_squared_error(self.y_test, self.model.predict(self.X_test))
        r2 = r2_score(self.y_test, self.model.predict(self.X_test))
        rmse = root_mean_squared_error(self.y_test, self.model.predict(self.X_test))

        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"R² score: {r2:.4f}")
        print(f"RMSE score: {rmse:.4f}")

    def plot_results(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.x, self.Y, 'b', label='Oryginalna funkcja')
        plt.plot(self.x, self.y_scaler.inverse_transform(self.y_pred), 'r', label='Aproksymacja TensorFlow')
        plt.title("Porównanie funkcji i aproksymacji (TensorFlow)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig('FNN_plot_results.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_training_test(self):
        # X_train_scaled = self.x_scaler.fit_transform(self.X_train)
        X_train_scaled = self.X_train
        plt.figure(figsize=(10, 5))
        plt.scatter(X_train_scaled, self.model.predict(X_train_scaled), color='blue')
        plt.scatter(self.X_test, self.model.predict(self.X_test), color='red')
        plt.title("Zbiór treningowy i testowy")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.savefig('FNN_plot_training_test.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_loss(self):
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        mae = self.history.history['mae']
        val_mae = self.history.history['val_mae']

        epochs_range = range(1, len(loss) + 1)

        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, loss, label='Train MSE (loss)')
        plt.plot(epochs_range, val_loss, label='Validation MSE')
        plt.xlabel('Epoka')
        plt.ylabel('MSE')
        plt.title('Błąd średniokwadratowy w czasie')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, mae, label='Train MAE')
        plt.plot(epochs_range, val_mae, label='Validation MAE')
        plt.xlabel('Epoka')
        plt.ylabel('MAE')
        plt.title('Średni błąd bezwzględny w czasie')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('FNN_plot_loss.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_test_metrics(self):
        y_pred = self.model.predict(self.X_test)
        plt.figure(figsize=(10, 5))
        plt.plot(self.y_scaler.inverse_transform(self.y_test), label='Rzeczywiste wartości')
        plt.plot(self.y_scaler.inverse_transform(y_pred), label='Przewidywane wartości')
        plt.title("Porównanie rzeczywistych i przewidywanych wartości")
        plt.xlabel("Indeks")
        plt.ylabel("Wartość")
        plt.legend()
        plt.grid(True)
        plt.savefig('FNN_plot_test_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"MAE: {mean_absolute_error(self.y_test, y_pred):.4f}")
        print(f"MSE: {mean_squared_error(self.y_test, y_pred):.4f}")
        print(f"RMSE: {root_mean_squared_error(self.y_test, y_pred):.4f}")
        print(f"R² score: {r2_score(self.y_test, y_pred):.4f}")


    def plot_pred_vs_real(self):
        y_pred = self.model.predict(self.X_test)
        plt.figure(figsize=(6, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.7, color='royalblue', edgecolor='k')
        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], 'r--', label='Idealna predykcja')

        plt.xlabel('Wartość rzeczywista (y_test)')
        plt.ylabel('Wartość przewidywana (y_pred)')
        plt.title('Porównanie: przewidywana vs rzeczywista')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig('FNN_plot_pred_vs_real.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(np.corrcoef(self.y_test.flatten(), y_pred.flatten()))

    def plot_test_errors(self):
        y_pred = self.model.predict(self.X_test)
        errors = self.y_scaler.inverse_transform(self.y_test) - self.y_scaler.inverse_transform(y_pred)
        plt.figure(figsize=(10, 5))
        plt.plot(errors, label='Błąd predykcji (y_test - y_pred)')
        plt.title('Błąd predykcji na zbiorze testowym')
        plt.xlabel('Indeks próbki')
        plt.ylabel('Błąd')
        plt.legend()
        plt.grid(True)
        plt.savefig('FNN_plot_test_errors.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_error_histogram(self):
        y_pred = self.model.predict(self.X_test)
        errors = self.y_scaler.inverse_transform(self.y_test) - self.y_scaler.inverse_transform(y_pred)
        plt.figure(figsize=(8, 5))
        plt.hist(errors, bins=30, color='skyblue', edgecolor='black')
        plt.title('Rozkład błędów predykcji (test set)')
        plt.xlabel('Błąd')
        plt.ylabel('Liczba próbek')
        plt.grid(True)
        plt.savefig('FNN_plot_error_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()

    
    def generate_all_reports(self, folder_name=None, device_name="CPU"):
        import os

        base_dir = "reports"
        os.makedirs(base_dir, exist_ok=True)

        if folder_name is None:
            folder_name = "default"

        # Dodanie urządzenia do nazwy folderu
        report_dir = os.path.join(base_dir, f"{device_name}_{folder_name}")
        os.makedirs(report_dir, exist_ok=True)

        print(f"📁 Zapis raportów do: {report_dir}")

        old_cwd = os.getcwd()
        os.chdir(report_dir)

        try:
            self.show_stats()
            self.plot_results()
            self.plot_test_metrics()
            self.plot_pred_vs_real()
            self.plot_loss()
            self.plot_training_test()
            self.plot_test_errors()
            self.plot_error_histogram()
        finally:
            os.chdir(old_cwd)

        print(f"✅ Wszystkie wykresy zapisane w: {report_dir}")





def plot_speedup(parameters, times, parameter_name='', name=" "):
    plt.figure(figsize=(10, 5))
    plt.plot(parameters, times, marker='o')
    plt.xlabel(f'{parameter_name}')
    plt.ylabel('Czas wykonania (s)')
    plt.title(f'Wykres czasu wykonywania w zależności od {parameter_name}')
    plt.grid(True)
    plt.savefig(f'FNN_{name}_speedup.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_fnn():
    my_fnn = fnn()
    my_fnn.set_parameters(-20, 20, 0.05, 5, 4, 0.1, 1, 2, 3, 4)
    my_fnn.define_function()
    my_fnn.scale_data()
    return my_fnn

def create_fnn_with_noise(noise):
    my_fnn = fnn()
    my_fnn.set_parameters(-20, 20, 0.05, 5, 4, 0.1, 1, 2, 3, 4, noise=noise)
    my_fnn.define_function()
    my_fnn.scale_data()
    return my_fnn

def create_fnn_with_params(start, end, step, x0, x1, x2, x3, alpha, beta, gamma):
    my_fnn = fnn()
    my_fnn.set_parameters(start, end, step, x0, x1, x2, x3, alpha, beta, gamma)
    my_fnn.define_function()
    my_fnn.scale_data()
    return my_fnn

def create_fnn_with_batch_size(batch_size):
    my_fnn = fnn()
    my_fnn.set_parameters(-20, 20, 0.05, 5, 4, 0.1, 1, 2, 3, 4)
    my_fnn.define_function()
    my_fnn.scale_data()
    my_fnn.create_model(batch_size=batch_size)
    return my_fnn

def test_batch_time(batch_size, device_name="CPU"):
    my_fnn = fnn()
    my_fnn.set_parameters(-40, 40, 0.05, 5, 4, 0.1, 1, 2, 3, 4)
    my_fnn.define_function()
    my_fnn.scale_data()
    start_time = time.time()
    my_fnn.create_model(batch_size=batch_size)
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken for batch size {batch_size}: {time_taken:.4f} seconds")
    
    # Zapis raportów z oznaczeniem urządzenia
    my_fnn.generate_all_reports(folder_name=f"#2batch_{batch_size}", device_name=device_name)
    
    return time_taken


def test_noise_time(noise, device_name="CPU"):
    my_fnn = fnn()
    my_fnn.set_parameters(-40, 40, 0.05, 5, 4, 0.1, 1, 2, 3, 4, noise=noise)
    my_fnn.define_function()
    my_fnn.scale_data()
    start_time = time.time()
    my_fnn.create_model()
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken for noise {noise}: {time_taken:.4f} seconds")
    
    # Raporty z device_name
    my_fnn.generate_all_reports(folder_name=f"#2noise_{noise}", device_name=device_name)
    
    return time_taken


def test_noise_accuracy(noise, device_name="CPU"):
    my_fnn = fnn()
    my_fnn.set_parameters(-40, 40, 0.05, 5, 4, 0.1, 1, 2, 3, 4, noise=noise)
    my_fnn.define_function()
    my_fnn.scale_data()

    # Trenowanie modelu
    my_fnn.create_model()

    # Predykcja i metryki
    y_pred = my_fnn.model.predict(my_fnn.X_test)
    mae = mean_absolute_error(my_fnn.y_test, y_pred)
    mse = mean_squared_error(my_fnn.y_test, y_pred)
    rmse = root_mean_squared_error(my_fnn.y_test, y_pred)
    r2 = r2_score(my_fnn.y_test, y_pred)

    print(f"Noise level: {noise}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² score: {r2:.4f}")

    # Raporty z device_name
    my_fnn.generate_all_reports(folder_name=f"#2noise_{noise}", device_name=device_name)

    return r2


def test_batch_accuracy(batch_size, device_name="CPU"):
    my_fnn = fnn()
    my_fnn.set_parameters(-40, 40, 0.05, 5, 4, 0.1, 1, 2, 3, 4)
    my_fnn.define_function()
    my_fnn.scale_data()

    # Trenowanie modelu
    my_fnn.create_model(batch_size=batch_size)

    # Obliczenie metryk
    y_pred = my_fnn.model.predict(my_fnn.X_test)
    mae = mean_absolute_error(my_fnn.y_test, y_pred)
    mse = mean_squared_error(my_fnn.y_test, y_pred)
    rmse = root_mean_squared_error(my_fnn.y_test, y_pred)
    r2 = r2_score(my_fnn.y_test, y_pred)

    print(f"batch size: {batch_size}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² score: {r2:.4f}")

    # Raporty z device_name
    my_fnn.generate_all_reports(folder_name=f"#2batch_{batch_size}", device_name=device_name)

    return r2





# if __name__ == "__main__":

#     batches = [2, 4, 8, 16, 32, 64, 128, 256, 512]
#     batch_times = []
#     noise = [1.0, 2.0, 3.0, 4.0, 5.0]
#     noise_times = []

#     # my_fnn = fnn()
#     # my_fnn.set_parameters(-20, 20, 0.05, 5, 4, 0.1, 1, 2, 3, 4)
#     # my_fnn.define_function()
#     # my_fnn.scale_data()
#     # my_fnn.create_model()

#     # my_fnn.generate_all_reports()
#     import ctypes
#     ctypes.CDLL('libcupti.so')
#     print("libcupti loaded successfully")

#     print("wersja tensorflow:")
#     print(tf.__version__)

#     print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#     print("Available devices:")
#     for device in tf.config.list_physical_devices():
#         print(device)

#     device_name = tf.test.gpu_device_name()
#     if not device_name:
#         raise SystemError('GPU device not found')
#     print('Found GPU at: {}'.format(device_name))

#     gpus = tf.config.list_physical_devices('GPU')

#     if gpus:
#         with tf.device('/GPU:0'):
#             print("Using GPU for training...")
#             for i in batches:
#     #             print(f"Batch size: {i}")
#                 batch_times.append(test_batch_time(i))
#     #             # my_fnn.show_stats()
#     #             # my_fnn.plot_results()
#     #             # # my_fnn.plot_training_test()
#     #             # # my_fnn.plot_loss()
#     #             # my_fnn.plot_pred_vs_real()
#             plot_speedup(batches, batch_times, parameter_name='batch size', name="GPU")


#     #         # my_fnn.show_stats()
#     #         # my_fnn.plot_results()
#     #         # # my_fnn.plot_training_test()
#     #         # # my_fnn.plot_loss()
#     #         # my_fnn.plot_pred_vs_real()
#     #         # my_fnn.plot_test_metrics()


#     batch_times = []
#     with tf.device('/CPU:0'):
#         print("Using CPU for training...")
#         for i in batches:
#     #         print(f"Batch size: {i}")
#             batch_times.append(test_batch_time(i))
#     #         # my_fnn.show_stats()
#     # #         # my_fnn.plot_results()
#     # #         # my_fnn.plot_training_test()
#     # #         # my_fnn.plot_loss()
#     # #         # my_fnn.plot_pred_vs_real()

#         plot_speedup(batches, batch_times, parameter_name='batch size', name="CPU")


#     for i in noise:
#     #         print(f"Noise: {i}")
#             noise_times.append(test_noise_time(i))
#     #         # my_fnn.show_stats()
#     #         # my_fnn.plot_results()
#     #         # my_fnn.plot_training_test()
#     #         # my_fnn.plot_loss()
#     #         # my_fnn.plot_pred_vs_real()

#     plot_speedup(noise, noise_times, parameter_name='noise')

#     noise_levels = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
#     noise_accuracies = []

#     for n in noise_levels:
#         acc = test_noise_accuracy(n)
#         noise_accuracies.append(acc)

#     plot_speedup(noise_levels, noise_accuracies, parameter_name='poziom szumu', name='accuracy_vs_noise')

#     batch_accuracy = []
#     for i in batches:
#         acc = test_batch_accuracy(i)
#         batch_accuracy.append(acc)

#     plot_speedup(batches, batch_accuracy, parameter_name='batch size', name="accuracy_vs_batch")

#TODO
# dwa wykresy szybkosci wykonania szybkosci wykonywania od batchsize i od wielkosci szumu /done
# wykresy metryk na testowym /done
#  na prawdziwych danych /rezygnacja
#  uruchomienie na gpu /done
# luzne notatki do pracy /done
# *fnn dziala lepiej na cpu niz na gpu ze wzgledu na czas przerzucenia danych na gpu, czego nie trzeba robic na cpu*

# TODO2


# tensorflow profiler, zobaczy na czym sa straty /done, sprawdzenie innych kompilacji binarek tensorflow /rezygnacja
# dodac wykresy dokladnie od batch size i od szumu osobno na gpu i na cpu /done
# sprawdzic czy obciaza kartke graficzna w menadzerze zadan /done


# Rozszerzac batch i liczbe danych dopoki gpu nie przekoczy cpu jesli mozliwe /done
# miara dokladnosci /????????
# rozne kompilacje tensorflow czy ma znaczenie pod procesor lub gpu lub przekompilowac z kodu zrodlowego /rezygnacja

import os
import matplotlib.pyplot as plt

def plot_comparison(x, cpu_values, gpu_values, xlabel, ylabel, title, filename):
    """Porównanie CPU vs GPU na jednym wykresie."""
    plt.figure(figsize=(10, 5))
    plt.plot(x, cpu_values, marker='o', label='CPU', linewidth=2)
    plt.plot(x, gpu_values, marker='s', label='GPU', linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    os.makedirs("reports/plots", exist_ok=True)
    plt.savefig(f"reports/plots/FNN_{filename}", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # =============================
    # KONFIGURACJA PARAMETRÓW
    # =============================
    # batches = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # noise_levels = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    # noise_levels = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]

    batches = [128]
    noise_levels = [3.0]

    print("Wersja TensorFlow:", tf.__version__)
    print("Dostępne urządzenia:")
    for d in tf.config.list_physical_devices():
        print(" -", d)

    gpus = tf.config.list_physical_devices('GPU')
    has_gpu = len(gpus) > 0
    if has_gpu:
        print(f"GPU wykryte: {tf.test.gpu_device_name()}")
    else:
        print("⚠️ Brak GPU – testy GPU zostaną pominięte.")

    # =============================
    # TESTY CPU
    # =============================
    cpu_batch_times, cpu_batch_acc = [], []
    cpu_noise_times, cpu_noise_acc = [], []

    with tf.device('/CPU:0'):
        for b in batches:
            cpu_batch_times.append(test_batch_time(b, device_name="CPU"))
            cpu_batch_acc.append(test_batch_accuracy(b, device_name="CPU"))
        for n in noise_levels:
            cpu_noise_times.append(test_noise_time(n, device_name="CPU"))
            cpu_noise_acc.append(test_noise_accuracy(n, device_name="CPU"))

    # =============================
    # TESTY GPU (jeśli dostępne)
    # =============================
    gpu_batch_times, gpu_batch_acc = [], []
    gpu_noise_times, gpu_noise_acc = [], []


    if has_gpu:
        with tf.device('/GPU:0'):
            for b in batches:
                gpu_batch_times.append(test_batch_time(b, device_name="GPU"))
                gpu_batch_acc.append(test_batch_accuracy(b, device_name="GPU"))
            for n in noise_levels:
                gpu_noise_times.append(test_noise_time(n, device_name="GPU"))
                gpu_noise_acc.append(test_noise_accuracy(n, device_name="GPU"))
    else:
        gpu_batch_times = [None] * len(batches)
        gpu_batch_acc = [None] * len(batches)
        gpu_noise_times = [None] * len(noise_levels)
        gpu_noise_acc = [None] * len(noise_levels)

    # =============================
    # WYKRESY PORÓWNAWCZE
    # =============================
    print("\nGenerowanie wykresów porównawczych...")

    # 1. Czas trenowania od batch size
    plot_comparison(
        batches, cpu_batch_times, gpu_batch_times,
        xlabel='Batch size', ylabel='Czas trenowania [s]',
        title='Porównanie czasu trenowania (CPU vs GPU) - batch size',
        filename='compare_time_batch_#2.png'
    )

    # 2. Dokładność (R²) od batch size
    plot_comparison(
        batches, cpu_batch_acc, gpu_batch_acc,
        xlabel='Batch size', ylabel='Dokładność (R²)',
        title='Porównanie dokładności (CPU vs GPU) - batch size',
        filename='compare_accuracy_batch_#2.png'
    )

    # 3. Czas trenowania od noise
    plot_comparison(
        noise_levels, cpu_noise_times, gpu_noise_times,
        xlabel='Poziom szumu', ylabel='Czas trenowania [s]',
        title='Porównanie czasu trenowania (CPU vs GPU) - noise',
        filename='compare_time_noise_#2.png'
    )

    # 4. Dokładność (R²) od noise
    plot_comparison(
        noise_levels, cpu_noise_acc, gpu_noise_acc,
        xlabel='Poziom szumu', ylabel='Dokładność (R²)',
        title='Porównanie dokładności (CPU vs GPU) - noise',
        filename='compare_accuracy_noise_#2.png'
    )

    print("\n✅ Wykresy zapisane w folderze reports/plots/")
    print("Pliki:")
    print(" - compare_time_batch.png")
    print(" - compare_accuracy_batch.png")
    print(" - compare_time_noise.png")
    print(" - compare_accuracy_noise.png")
