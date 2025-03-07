#!/usr/bin/env python

from numpy import zeros, array, einsum, savez, load, sqrt
from numpy.random import default_rng

from my_lib import tqdm, batched

class MLP:
	def __init__(self, layer_sizes, activations, loss, seed=1108):
		self.rng = default_rng(seed)
		self.activations = activations
		self.loss = loss
		self.weights = []
		self.biases = []
		for size1, size2 in zip(layer_sizes[:-1], layer_sizes[1:]):
			self.weights.append(self.rng.normal(0, sqrt(2/(size1+size2)), (size2, size1)))
			self.biases.append(zeros(size2))

	def forward(self, x):
		outputs = [x]
		derivatives = []
		for weight, bias, activation in zip(self.weights, self.biases, self.activations):
			score = bias + einsum('ij,kj->ki', weight, outputs[-1])
			derivatives.append(activation.derivative(score))
			outputs.append(activation(score))
		return outputs, derivatives

	def backward(self, x, y, outputs, derivatives, learning_rate):
		learning_rate /= len(x)
		gradient = self.loss.derivative(y, outputs[-1])
		for i in reversed(range(len(derivatives))):
			gradient = einsum('ij,ijl->il', gradient, derivatives[i])
			self.biases[i] -= learning_rate * einsum('ij->j', gradient)
			old_weights = self.weights[i]
			self.weights[i] -= learning_rate * einsum('ij,ik->jk', gradient, outputs[i])
			gradient = einsum('ij,jk->ik', gradient, old_weights)

	def train(self, x, y, learning_rates, batch_size):
		n_samples = len(x)
		x = x.reshape(n_samples, -1)
		y = y.reshape(n_samples, -1)
		losses = []
		with tqdm(total=len(learning_rates)*n_samples) as progress:
			for rate in learning_rates:
				loss = 0
				for batch_x, batch_y in zip(batched(x, batch_size), batched(y, batch_size)):
					actual_batch_size = len(batch_x)
					batch_x = array(batch_x)
					batch_y = array(batch_y)
					outputs, derivatives = self.forward(batch_x)
					self.backward(batch_x, batch_y, outputs, derivatives, rate*actual_batch_size/n_samples)
					loss += self.loss(batch_y, outputs[-1]) / n_samples
					progress.update(actual_batch_size)
				print(loss)
				losses.append(loss)
		return losses

	def predict(self, x):
		single = False
		if x.shape[0] > 1 and x.size == self.weights[0].shape[1]:
			single = True
			x = x.reshape(1, -1)
		outputs, derivatives = self.forward(x)
		y_pred = outputs[-1]
		if single:
			y_pred = y_pred[0]
		return y_pred

	def save(self, path):
		data_dict = {}
		for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
			data_dict[f'weight_{i}'] = weight
			data_dict[f'bias_{i}'] = bias
		savez(path, **data_dict)
	
	def load(self, path):
		data_dict = load(path)
		for i in range(len(self.weights)):
			self.weights[i] = data_dict[f'weight_{i}']
			self.biases[i] = data_dict[f'bias_{i}']

if __name__ == '__main__':
	from matplotlib import pyplot as plt
	from my_lib import sigmoid, relu, mse

	X = array([[0, 0], [0, 1], [1, 0], [1, 1]])
	Y = array([[0], [1], [1], [0]])
	mlp = MLP(
		layer_sizes=[2, 3, 1],
		activations=[sigmoid, relu],
		loss=mse
	)
	losses = mlp.train(X, Y, [0.4]*1000, 4)
	print(mlp.predict(X))
	plt.plot(losses)
	plt.show()
