import numpy as np
import matplotlib.pyplot as plt

# =============================
# USTAWIENIE SEEDÓW
# =============================
np.random.seed(42)

# =============================
# PARAMETRY (jak w set_parameters)
# =============================
start, end, step = -40, 40, 0.05

x0 = 5
x1 = 4      # nieużywane w funkcji
x2 = 0.1
x3 = 1

alpha = 2
beta = 3
gamma = 4
noise = 5.0

# =============================
# DANE
# =============================
x = np.arange(start, end, step)

rand = np.random.normal(loc=0.0, scale=noise, size=len(x))

# =============================
# FUNKCJA
# =============================
y = (
    alpha * x0 * np.sin(x)
    + beta * x2 * x3 * x * x
    + gamma * np.abs(x0 - x2)
    + rand
)

# =============================
# WYKRES
# =============================
plt.figure(figsize=(10, 5))
plt.plot(x, y, label="y(x) z szumem (noise=1.0)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Funkcja y(x) z szumem")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("./funkcja_z_szumem_noise_1.png", dpi=300, bbox_inches="tight")
plt.close()