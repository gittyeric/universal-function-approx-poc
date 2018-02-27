Here's a minimalistic implementation of a neural net factored as a "Universal Function Approximator".  The key realization should be that with minor modification to the neural net architecture, you can approximate any "blackbox" function with enough data and compute.  This example teaches a simple 1 layer network how to multiply 2 inputs together.  You simply need Python3 (maybe 2?) installed with the following libraries:

pip3 install numpy

pip3 install tensorflow

pip3 install keras

Then run the script from command line:

python3 keras_multiply.py

The network should learn how to multiply with 99% accuracy within 100-150 epochs (5-10 minutes).  If you have issues, there's detailed installation instructions for python, pip (package manager), keras and tensorflow.  You might also have to try the above commands without the '3' in 'python3' / 'pip3'.
