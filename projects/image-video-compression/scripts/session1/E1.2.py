from src.utils import my_sin
import numpy as np
import matplotlib.pyplot as plt

omega = 1

t = np.linspace(0, 2*np.pi, 100) 

y = my_sin(t, omega)

plt.plot(t, y)

plt.show()

