import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Données
X = np.array([5, 7, 9, 12, 15, 17, 20, 22]).reshape(-1, 1)
Y = np.array([52, 48, 45, 41, 38, 35, 32, 30])

# Modèle
model = LinearRegression()
model.fit(X, Y)

# Coefficients
a = model.coef_[0]
b = model.intercept_
print(f"Equation: y = {a:.2f}x + {b:.2f}")

# Prédiction pour 30°C
cons_30 = model.predict([[30]])
print(f"Prédiction pour 30°C: {cons_30[0]:.2f} kWh/jour")
