from keras.models import Sequential
from keras.layers import *
from keras.layers.advanced_activations import *
from keras.optimizers import *
from random import randint
import numpy

# This can be almost any function of the form f(x1, x2) = y
def blackbox(x1, x2):
  return x1 * x2 # Let's teach the CPU how to multiply!!!

# Generate a random batch of xi => f(xi) pairs,
# where f is the blackbox function
def generateBatch(batchSize):
  X = numpy.zeros((batchSize, 2))
  Y = numpy.zeros(batchSize)
  
  for i in range(0, batchSize):
    xi = [ randint(0, 10), randint(0, 10) ]
    yi = blackbox(xi[0], xi[1])
    
    X[i] = numpy.array(xi)
    Y[i] = yi
    
  return X, Y

def normalizeX(x):
  return [x[0] / 10.0, x[1] / 10.0]

def normalize(X, Y):
  return numpy.array([normalizeX(xi) for xi in X]), numpy.array([yi / 100.0 for yi in Y])

def denormalize(Y):
  return numpy.array([yi*100 for yi in Y])

# Create a blank Neural Net
model = Sequential()

# Add layers to Neural Net, 2 inputs to 101 neurons to 1 neuron's output.
# In the case of multiplication up to 100, every hidden layer neuron will
# learn to output 1 given a unique input, then the final
# layer will add up all the 1's to give the product of the inputs
model.add( Dense(units=101, input_dim=2, activation=Softmax(axis=-1)) )
model.add( Dense(units=1) )

# Initialize the model randomly
model.compile(loss='mean_squared_error', optimizer="adam")

# Now start the cycle of training and evaluation
batchSize = 20000
epochs = 10000

for epoch in range(0, epochs):
  print("Training Epoch: " + str(epoch))

  # Generate batch of training data
  X, Y = generateBatch(batchSize)
  Xnorm, Ynorm = normalize(X, Y)
  
  # Train model to 'fit' X batch to Y batch
  model.fit(Xnorm, Ynorm, epochs=1)
  
  # Evaluate trained model
  X, Y = generateBatch(batchSize)
  Xnorm, Ynorm = normalize(X, Y)
  predictedY = denormalize(model.predict(Xnorm))
  
  errorSum = 0.0
  for i in range(0, batchSize):
    errorSum += abs(Y[i] - predictedY[i])
  
  print("Avg error: " + str(errorSum/batchSize))
