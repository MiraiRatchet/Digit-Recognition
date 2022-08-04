# Fully connected neural network for digit recognition

This neural network uses MNIST dataset of handwritten digits for training and testing.
Sigmoid is used as an activation function for hidden layers, and the last layer uses softmax.

## To start using this network

You can use different options for training. Open file Options.txt to apply them. The first option is the number of hidden layers (1 or more). The second is the number of neurons on each layer. Please, write each number on a new line. If you have 2 hidden layers, and each of them has 16 neurons, write 16 two times with a linebreak.

Open your Terminal, change directory to the Network folder and use `python3 NetworkTrain.py`.
When training is done, you can run `python3 RecognizeDigits.py`. Feel free to draw different digits for recognition.