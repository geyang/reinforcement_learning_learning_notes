# Huber Loss

link: https://en.wikipedia.org/wiki/Huber_loss

```python
import numpy as np
import matplotlib.pyplot as plt
epsilon = 0.5
x = np.linspace(-5, 5, 101)
L = np.where(np.abs(x) < epsilon , 0.5 * x**2, epsilon * (np.abs(x) - 0.5 * epsilon))
plt.plot(x, L)
plt.show()
```

basically the Huber loss is a quadratic loss with linear taper beyond the value `epsilon`.
![#Out[0]]

