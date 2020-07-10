# MNIST_NN
Neural Network model on the MNIST image dataset. The model can be trained with a GPU or CPU.
Best model has an architecture of (784, 128, 10) with one hidden layer of 128 neurons. Test accuracy is 98.2% after training for 25 epochs on the CPU with mini-batch SGD. Adding more hidden layers didn't increase accuracy, and creates a vanishing/exploding gradient problem. These multilayer networks were trained on a Kaggle kernel with a GPU.
