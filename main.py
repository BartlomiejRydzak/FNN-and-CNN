import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import random as rndm
import random

np.random.seed(40)
tf.random.set_seed(40)
rndm.seed(40)

x = np.arange(-20, 20, 0.05)
# x0 = random.randint(-5, 5)
# x1 = random.randint(-5, 5)
# x2 = random.randint(-5, 5)
# x3 = random.randint(-500, 500)
x0 = 5
x1 = 4
x2 = 0.1
x3 = 1
alpha, beta, gamma = 2, 3, 4
rand = np.random.normal(loc=0.0, scale=1.0, size=len(x))

y = alpha * x0 * np.sin(x) + beta * x2 * x3 * x*x + gamma * np.abs(x0 - x2) + rand

X = x.reshape(-1, 1)
Y = y.reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(Y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# print(f"x_train rozmiar {len(X_train)} {X_train[:50]}")
# print(f"x_test rozmiar {len(X_test)} {X_test[:50]}")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(80, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(80, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.1, verbose=0)

loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MAE: {mae:.4f}")

X_all_scaled = scaler.transform(X)
y_pred = model.predict(X_all_scaled)




from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

mae = mean_absolute_error(y_test, model.predict(X_test))
mse = mean_squared_error(y_test, model.predict(X_test))
r2 = r2_score(y_test, model.predict(X_test))
rmse = root_mean_squared_error(y_test, model.predict(X_test))

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R² score: {r2:.4f}")
print(f"RMSE score: {rmse:.4f}")

# # batch-size: 32
# # MAE: 0.1355
# # MSE: 0.0256
# # R² score: 0.9765
# # RMSE score: 0.1599

# # batch-size: 16
# # MAE: 0.0737
# # MSE: 0.0099
# # R² score: 0.9909
# # RMSE score: 0.0993

# # batch-size: 64
# # MAE: 0.1261
# # MSE: 0.0239
# # R² score: 0.9780
# # RMSE score: 0.1545


# # print(f"przed skalowanie {y[:50]}")
# # print(f"po skalowaniu {y_scaled[:50]}")

plt.figure(figsize=(10, 5))
plt.plot(x, y_scaled, 'b', label='Oryginalna funkcja')
plt.plot(x, y_pred, 'r', label='Aproksymacja TensorFlow')
plt.title("Porównanie funkcji i aproksymacji (TensorFlow)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

X_train_scaled = scaler.fit_transform(X_train)
plt.figure(figsize=(10, 5))
plt.scatter(X_train_scaled, model.predict(X_train_scaled), color='blue')
plt.scatter(X_test, model.predict(X_test), color='red')
plt.title("Zbiór treningowy i testowy")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
mae = history.history['mae']
val_mae = history.history['val_mae']

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
plt.show()

# import matplotlib.pyplot as plt

# y_pred = model.predict(X_test)

# plt.figure(figsize=(6, 6))
# plt.scatter(y_test, y_pred, alpha=0.7, color='royalblue', edgecolor='k')
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Idealna predykcja')

# plt.xlabel('Wartość rzeczywista (y_test)')
# plt.ylabel('Wartość przewidywana (y_pred)')
# plt.title('Porównanie: przewidywana vs rzeczywista')
# plt.legend()
# plt.grid(True)
# plt.axis('equal')
# plt.show()

# print(np.corrcoef(y_test.flatten(), y_pred.flatten()))
