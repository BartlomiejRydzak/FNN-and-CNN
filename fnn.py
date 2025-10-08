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


    def create_model(self, epochs=100, batch_size=64):

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(80, activation='relu', input_shape=(1,)),
            tf.keras.layers.Dense(80, activation='relu'),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])


        log_dir = "./logs"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            profile_batch='10,30'
        )
        
        self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0, callbacks=[tensorboard_callback])


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
        plt.savefig('plot_results.png', dpi=300, bbox_inches='tight')
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
        plt.savefig('plot_training_test.png', dpi=300, bbox_inches='tight')
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
        plt.savefig('plot_loss.png', dpi=300, bbox_inches='tight')
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
        plt.savefig('plot_test_metrics.png', dpi=300, bbox_inches='tight')
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
        plt.savefig('plot_pred_vs_real.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(np.corrcoef(self.y_test.flatten(), y_pred.flatten()))

    def generate_all_reports(self, model_trained=False, epochs=100, batch_size=64):
        if not model_trained:
            self.create_model(epochs=epochs, batch_size=batch_size)

        import os
        os.makedirs("reports", exist_ok=True)

        y_pred = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = root_mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        with open("reports/test_metrics.txt", "w") as f:
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"MSE: {mse:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"R² score: {r2:.4f}\n")

        self.plot_results()
        self.plot_test_metrics()
        self.plot_pred_vs_real()
        self.plot_loss()
        self.plot_training_test()



def plot_speedup(parameters, times, parameter_name='', name=" "):
    plt.figure(figsize=(10, 5))
    plt.plot(parameters, times, marker='o')
    plt.xlabel(f'{parameter_name}')
    plt.ylabel('Czas wykonania (s)')
    plt.title(f'Wykres czasu wykonywania w zależności od {parameter_name}')
    plt.grid(True)
    plt.savefig(f'{name}_speedup.png', dpi=300, bbox_inches='tight')
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

def test_batch_time(batch_size):
    my_fnn = fnn()
    my_fnn.set_parameters(-40, 40, 0.05, 5, 4, 0.1, 1, 2, 3, 4)
    my_fnn.define_function()
    my_fnn.scale_data()
    start_time = time.time()
    my_fnn.create_model(batch_size=batch_size)
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken for batch size {batch_size}: {time_taken:.4f} seconds")
    my_fnn.plot_results()
    return time_taken

def test_noise_time(noise):
    my_fnn = fnn()
    my_fnn.set_parameters(-20, 20, 0.05, 5, 4, 0.1, 1, 2, 3, 4, noise=noise)
    my_fnn.define_function()
    my_fnn.scale_data()
    start_time = time.time()
    my_fnn.create_model()
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken for noise {noise}: {time_taken:.4f} seconds")
    my_fnn.plot_results()
    my_fnn.plot_test_metrics()
    my_fnn.plot_pred_vs_real()
    return time_taken

if __name__ == "__main__":

    batches = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    batch_times = []
    noise = [1.0, 2.0, 3.0, 4.0, 5.0]
    noise_times = []

    # my_fnn = fnn()
    # my_fnn.set_parameters(-20, 20, 0.05, 5, 4, 0.1, 1, 2, 3, 4)
    # my_fnn.define_function()
    # my_fnn.scale_data()
    # my_fnn.create_model()
    import ctypes
    ctypes.CDLL('libcupti.so')
    print("libcupti loaded successfully")

    print("wersja tensorflow:")
    print(tf.__version__)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    print("Available devices:")
    for device in tf.config.list_physical_devices():
        print(device)

    device_name = tf.test.gpu_device_name()
    if not device_name:
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        with tf.device('/GPU:0'):
            print("Using GPU for training...")
            for i in batches:
                print(f"Batch size: {i}")
                batch_times.append(test_batch_time(i))
                # my_fnn.show_stats()
                # my_fnn.plot_results()
                # # my_fnn.plot_training_test()
                # # my_fnn.plot_loss()
                # my_fnn.plot_pred_vs_real()
            plot_speedup(batches, batch_times, parameter_name='batch size', name="GPU")


            # my_fnn.show_stats()
            # my_fnn.plot_results()
            # # my_fnn.plot_training_test()
            # # my_fnn.plot_loss()
            # my_fnn.plot_pred_vs_real()
            # my_fnn.plot_test_metrics()


    batch_times = []
    with tf.device('/CPU:0'):
        print("Using CPU for training...")
        for i in batches:
            print(f"Batch size: {i}")
            batch_times.append(test_batch_time(i))
    #         # my_fnn.show_stats()
    #         # my_fnn.plot_results()
    #         # my_fnn.plot_training_test()
    #         # my_fnn.plot_loss()
    #         # my_fnn.plot_pred_vs_real()

        plot_speedup(batches, batch_times, parameter_name='batch size', name="CPU")


    # for i in noise:
    #         print(f"Noise: {i}")
    #         noise_times.append(test_noise_time(i))
    #         # my_fnn.show_stats()
    #         # my_fnn.plot_results()
    #         # my_fnn.plot_training_test()
    #         # my_fnn.plot_loss()
    #         # my_fnn.plot_pred_vs_real()

    # plot_speedup(noise, noise_times, parameter_name='noise')
# dwa wykresy szybkosci wykonania szybkosci wykonywania od batchsize i od wielkosci szumu
# wykresy metryk na testowym
#  na prawdziwych danych
#  uruchomienie na gpu
# luzne notatki do pracy
# *fnn dziala lepiej na cpu niz na gpu ze wzgledu na czas przerzucenia danych na gpu, czego nie trzeba robic na cpu*