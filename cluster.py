import numpy as np
import matplotlib.pyplot as plt

import os
import json
import tensorflow as tf

tf_config = {
    "cluster": {
        "chief": ["192.168.100.4:12345"],
        "worker": ["192.168.100.41:12345"]
    },
    "task": {"type": "worker", "index": 0}
}


os.environ['TF_CONFIG'] = json.dumps(tf_config)



strategy = tf.distribute.MultiWorkerMirroredStrategy()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import random as rndm
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
    def set_parameters(self, start, end, step, x0, x1, x2, x3, alpha, beta, gamma, noise=1.0):
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
    def create_model(self, epochs=100, batch_size=32,  architecture="small"):
        # pierwsza wersja

        with strategy.scope():
            if architecture == "small":
                
                    self.model = tf.keras.Sequential([
                        tf.keras.layers.Dense(80, activation='relu', input_shape=(1,)),
                        tf.keras.layers.Dense(80, activation='relu'),
                        tf.keras.layers.Dense(20, activation='relu'),
                        tf.keras.layers.Dense(20, activation='relu'),
                        tf.keras.layers.Dense(1)
                    ])
            else:
                raise ValueError(f"Nieznany typ architektury: {architecture}")

            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)


        self.loss, self.mae = self.model.evaluate(self.X_test, self.y_test, verbose=0)

        self.X_all_scaled = self.x_scaler.transform(self.X)
        self.y_pred = self.model.predict(self.X_all_scaled)
    
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

def test_series_by_noise(noise_values, batch_sizes, device_name="CPU"):
    """
    Dla każdej wartości noise testuje różne batch_size
    i rysuje osobne wykresy dla każdej stałej wartości noise.
    """
    os.makedirs(f"reports/compare2/{device_name}", exist_ok=True)

    for noise in noise_values:
        times = []
        accuracies = []

        for batch_size in batch_sizes:
            print(f"\n[{device_name}] noise={noise}, batch_size={batch_size}")

            # Przygotowanie modelu
            my_fnn = fnn()
            my_fnn.set_parameters(-40, 40, 0.05, 5, 4, 0.1, 1, 2, 3, 4, noise=noise)
            my_fnn.define_function()
            my_fnn.scale_data()

            start_time = time.time()
            my_fnn.create_model(batch_size=batch_size)
            end_time = time.time()

            y_pred = my_fnn.model.predict(my_fnn.X_test)
            r2 = r2_score(my_fnn.y_test, y_pred)
            elapsed = end_time - start_time

            times.append(elapsed)
            accuracies.append(r2)

            print(f"⏱ Czas: {elapsed:.3f}s | R²: {r2:.4f}")

        # --- Rysowanie wykresu czasu ---
        plt.figure(figsize=(10, 5))
        plt.plot(batch_sizes, times, marker='o', linewidth=2, color='royalblue')
        plt.xlabel('Batch size')
        plt.ylabel('Czas trenowania [s]')
        plt.title(f'{device_name} — Czas trenowania vs batch size (noise={noise})')
        plt.grid(True)
        plt.savefig(f'reports/compare2/{device_name}/time_vs_batch_noise_{noise}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # --- Rysowanie wykresu dokładności ---
        plt.figure(figsize=(10, 5))
        plt.plot(batch_sizes, accuracies, marker='s', linewidth=2, color='seagreen')
        plt.xlabel('Batch size')
        plt.ylabel('Dokładność (R²)')
        plt.title(f'{device_name} — Dokładność vs batch size (noise={noise})')
        plt.grid(True)
        plt.savefig(f'reports/compare2/{device_name}/accuracy_vs_batch_noise_{noise}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📈 Zapisano wykresy dla noise={noise} w reports/compare2/")


if __name__ == "__main__":
    batches = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    noise_levels = [0.5]

    my_fnn = fnn()
    my_fnn.set_parameters(-40, 40, 0.05, 5, 4, 0.1, 1, 2, 3, 4)
    my_fnn.define_function()


    with tf.device('/CPU:0'):
        test_series_by_noise(noise_levels, batches, 'CPU')


