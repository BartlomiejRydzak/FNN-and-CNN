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

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")

if is_distributed:
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    print(f"Using MultiWorkerMirroredStrategy with {strategy.num_replicas_in_sync} replicas")
else:
    strategy = tf.distribute.get_strategy()
    print("Using default strategy (single machine)")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import random as rndm
import random
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error
import time


class fnn:
    def __init__(self):
        self.x0 = None
        self.x1 = None
        self.x2 = None
        self.x3 = None
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.rand = None
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
        self.model = None
        self.history = None
        self.mae = None
        self.loss = None
    
    def set_parameters(self, start, end, step, x0, x1, x2, x3, alpha, beta, gamma, noise=1.0):
        np.random.seed(42)
        tf.random.set_seed(42)
        rndm.seed(42)

        self.x = np.arange(start, end, step)
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

        self.X = self.x.reshape(-1, 1)
        self.Y = self.y.reshape(-1, 1)
    
    def scale_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.X_train = self.x_scaler.fit_transform(self.X_train.reshape(-1, 1))
        self.X_test = self.x_scaler.transform(self.X_test.reshape(-1, 1))
        self.y_train = self.y_scaler.fit_transform(self.y_train.reshape(-1, 1))
        self.y_test = self.y_scaler.transform(self.y_test.reshape(-1, 1))
    
    def create_model(self, epochs=100, batch_size=32, architecture="small"):
        X_train_f32 = self.X_train.astype(np.float32)
        y_train_f32 = self.y_train.astype(np.float32)
        X_test_f32  = self.X_test.astype(np.float32)
        y_test_f32  = self.y_test.astype(np.float32)

        num_replicas = strategy.num_replicas_in_sync
        if batch_size < num_replicas:
            print(f"⚠ batch_size ({batch_size}) < num_replicas ({num_replicas}). Adjusting to {num_replicas}.")
            batch_size = num_replicas

        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train_f32, y_train_f32))
            .shuffle(buffer_size=len(X_train_f32))
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

        val_ds = (
            tf.data.Dataset.from_tensor_slices((X_test_f32, y_test_f32))
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

        with strategy.scope():
            if architecture == "small":
                # self.model = tf.keras.Sequential([
                #     tf.keras.layers.Input(shape=(1,)),
                #     tf.keras.layers.Dense(80, activation='relu'),
                #     tf.keras.layers.Dense(80, activation='relu'),
                #     tf.keras.layers.Dense(20, activation='relu'),
                #     tf.keras.layers.Dense(20, activation='relu'),
                #     tf.keras.layers.Dense(1)
                # ])
                self.model = tf.keras.Sequential([
                    # bardzo dlugi czas wykonywania
                    tf.keras.layers.Dense(1000, activation='relu', input_shape=(1,)),
                    # tf.keras.layers.Dense(5_000, activation='relu'),
                    # tf.keras.layers.Dense(1000, activation='relu'),
                    tf.keras.layers.Dense(100, activation='relu'),
                    tf.keras.layers.Dense(50, activation='relu'),
                    tf.keras.layers.Dense(1)
                ])
            elif architecture == "large":
                self.model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(1,)),
                    tf.keras.layers.Dense(1000, activation='relu'),
                    tf.keras.layers.Dense(400, activation='relu'),
                    tf.keras.layers.Dense(100, activation='relu'),
                    tf.keras.layers.Dense(20, activation='relu'),
                    tf.keras.layers.Dense(1)
                ])
            else:
                raise ValueError(f"Unknown architecture: {architecture}")

            self.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )

        self.history = self.model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            verbose=1
        )

        eval_results = self.model.evaluate(val_ds, verbose=1)
        
        if isinstance(eval_results, list):
            if hasattr(eval_results[0], 'values'):
                self.loss = float(list(eval_results[0].values())[0].numpy())
                self.mae = float(list(eval_results[1].values())[0].numpy())
            else:
                self.loss = float(eval_results[0])
                self.mae = float(eval_results[1])
        else:
            if hasattr(eval_results, 'values'):
                self.loss = float(list(eval_results.values())[0].numpy())
                self.mae = None
            else:
                self.loss = float(eval_results)
                self.mae = None

        X_all_scaled = self.x_scaler.transform(self.X).astype(np.float32)
        predict_ds = tf.data.Dataset.from_tensor_slices(X_all_scaled).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        self.y_pred = self.model.predict(predict_ds, verbose=0)

        
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
            print(f"\n{'='*60}")
            print(f"[{device_name}] noise={noise}, batch_size={batch_size}")
            print(f"{'='*60}")

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

        plt.figure(figsize=(10, 5))
        plt.plot(batch_sizes, times, marker='o', linewidth=2, color='royalblue')
        plt.xscale('log')
        plt.xlabel('Batch size')
        plt.ylabel('Czas trenowania [s]')
        plt.title(f'{device_name} — Czas trenowania vs batch size (noise={noise})')
        plt.grid(True)
        plt.savefig(f'reports/compare2/{device_name}/time_vs_batch_noise_{noise}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(batch_sizes, accuracies, marker='s', linewidth=2, color='seagreen')
        plt.xlabel('Batch size')
        plt.ylabel('Dokładność (R²)')
        plt.title(f'{device_name} — Dokładność vs batch size (noise={noise})')
        plt.grid(True)
        plt.savefig(f'reports/compare2/{device_name}/accuracy_vs_batch_noise_{noise}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f" Zapisano wykresy dla noise={noise} w reports/compare2/")



if __name__ == "__main__":
    batches = [256]
    noise_levels = [0.5]

    my_fnn = fnn()
    my_fnn.set_parameters(-40, 40, 0.05, 5, 4, 0.1, 1, 2, 3, 4)
    my_fnn.define_function()

    test_series_by_noise(noise_levels, batches, 'CPU2')