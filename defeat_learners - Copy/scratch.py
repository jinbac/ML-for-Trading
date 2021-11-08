import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(343455)

x1 = np.linspace(0,10,100)
x2 = np.linspace(0,10,100)*-1
X = np.array([x1,x2])
y = np.sin(x1)
noise = np.random.random(size=(10,))*100
noise=noise.astype(int)
print noise
x2[noise] = 1
print x2
# Y = y + noise
# print Y
# print x
# print y

# plt.plot(x)
# plt.plot(y)
# plt.plot(Y)
#
#
# plt.show()