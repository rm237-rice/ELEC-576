import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 50, endpoint=False)
x = np.cos(t**2)

plt.axis([0, 10, -1, 1])
plt.plot(t,x)
plt.show()