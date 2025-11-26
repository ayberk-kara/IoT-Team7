import numpy as np
import random

LEARNING_RATE = 0.1

def test_gate(perceptron):
    print("0, 0 =>", perceptron.predict([0, 0]))
    print("0, 1 =>", perceptron.predict([0, 1]))
    print("1, 0 =>", perceptron.predict([1, 0]))
    print("1, 1 =>", perceptron.predict([1, 1]))

class Perceptron:
  def __init__(self, input_size):
    self.input_size = input_size + 1 # +1 for bias
    self.weights = []
    for i in range(self.input_size):
      self.weights.append(random.uniform(0, 1)) # random weights at start

  def activation_function(self, z): # based on ReLU
    if z > 0.5: # 0.5 because the values range from 0 to 1
      return 1
    else:
      return 0

  def predict(self, input_data):
        input_with_bias = [1] + input_data[:]
        z = 0
        for i in range(self.input_size):
          z += input_with_bias[i] * self.weights[i]
        return self.activation_function(z)

  def train(self, input_data, error):
      input_with_bias = [1] + input_data[:] # adds 1 at the start of array
      for i in range(self.input_size):
          self.weights[i] += LEARNING_RATE * error * input_with_bias[i]

def training_loop(perceptron, x_train, y_train):
  order = list(range(len(x_train)))
  all_correct = False
  epoch = 0
  while not all_correct:
    all_correct = True
    random.shuffle(order)
    for i in order: # each sample in training set
      x, y = x_train[i], y_train[i]
      prediction = perceptron.predict(x)
      error = y - prediction
      if error != 0:
        perceptron.train(x, error)
        all_correct = False
    print(f"Epoch {epoch} - Weights:", np.round(perceptron.weights, 2))
    epoch += 1

print("Training OR gate perceptron")
or_x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
or_y_train = [0, 1, 1, 1]

or_perceptron = Perceptron(2)
training_loop(or_perceptron, or_x_train, or_y_train)
test_gate(or_perceptron)

print("\nTraining NAND gate perceptron")
nand_x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
nand_y_train = [1, 1, 1, 0]

nand_perceptron = Perceptron(2)
training_loop(nand_perceptron, nand_x_train, nand_y_train)
test_gate(nand_perceptron)

print("\nTraining AND gate perceptron")
and_x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
and_y_train = [0, 0, 0, 1]

and_perceptron = Perceptron(2)
training_loop(and_perceptron, and_x_train, and_y_train)
test_gate(and_perceptron)

def xor(input):
    nand_output = nand_perceptron.predict([input[0], input[1]])
    or_output = or_perceptron.predict([input[0], input[1]])
    xor_output = and_perceptron.predict([or_output, nand_output])
    return xor_output

print("\nXOR")
print("0, 0 =>", xor([0, 0]))
print("0, 1 =>", xor([0, 1]))
print("1, 0 =>", xor([1, 0]))
print("1, 1 =>", xor([1, 1]))