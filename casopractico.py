import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 🔹 Datos de ejemplo
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2.1, 2.9, 3.7, 5.1, 7.2])

# 🔹 Creación del modelo de Ridge con polinomio de grado 10
alpha_values = [0, 0.001, 0.01, 0.1, 1, 10]
plt.figure(figsize=(10, 6))

for alpha in alpha_values:
    model = make_pipeline(PolynomialFeatures(10), Ridge(alpha=alpha))
    model.fit(X, y)
    
    # 🔹 Predicción en valores más finos
    X_fit = np.linspace(0, 6, 100).reshape(-1, 1)
    y_pred = model.predict(X_fit)
    
    # 🔹 Graficamos la curva ajustada
    plt.plot(X_fit, y_pred, label=f'Alpha={alpha}')

# 🔹 Graficamos los datos originales
plt.scatter(X, y, color='black', label="Datos reales")
plt.legend()
plt.title("Efecto de Ridge Regression en un Polinomio de Grado 10")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

