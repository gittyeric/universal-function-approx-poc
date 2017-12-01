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
    xi = [ randint(0, 100), randint(0, 100) ]
    yi = blackbox(xi[0], xi[1])
    
    X[i] = numpy.array(xi)
    Y[i] = yi
    
  return X, Y

# Create a blank Neural Net
model = Sequential()

# Add layers to Neural Net, 2 inputs to 50 neurons to 1 neuron's output.
# The more complex the function, the more layers or units you may need
model.add( Dense(units=50, input_dim=2, activation='linear') )
model.add( Dense(units=1, activation='linear') )

# Initialize the model randomly
model.compile(loss='mean_squared_error', optimizer="adam")

# Now start the cycle of training and evaluation
batchSize = 20000
epochs = 10000

for epoch in range(0, epochs):
  print("Training Epoch: " + str(epoch))

  # Generate batch of training data
  X, Y = generateBatch(batchSize)
  
  # Train model to 'fit' X batch to Y batch
  model.fit(X, Y, epochs=1)
  
  # Evaluate trained model
  X, Y = generateBatch(batchSize)
  predictedY = model.predict(X)
  
  errorSum = 0.0
  for i in range(0, batchSize):
    errorSum += abs(Y[i] - predictedY[i])
  
  print("Avg error: " + str(errorSum/batchSize))