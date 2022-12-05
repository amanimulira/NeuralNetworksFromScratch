from fromScratch import *
from Network import Network
from FCLayer import FullyConnectedLyaer
from ActivationLayer import ActivationLayer
from ActivationFunctions import tanh, tanh_prime
from LossFunctions import mse, mse_prime

x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

net = Network()

net.add(FullyConnectedLyaer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FullyConnectedLyaer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))


net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=10000, learning_rate=0.1)

out = net.predict(x_train)
print(out)