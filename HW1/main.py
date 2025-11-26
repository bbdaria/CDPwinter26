from collect import *
from network import *


def main():
    # initialize layer sizes as list
    layers = [784, 128, 64, 10]

    learning_rate = 0.1
    mini_batch_size = 16
    epochs = 5

    # initialize training, validation and testing data
    training_data, validation_data, test_data = load_mnist()

    # initialize and train neural network
    nn, train_time = train_nn(sizes=layers,
                              learning_rate=learning_rate,
                              mini_batch_size=mini_batch_size,
                              epochs=epochs,
                              matmul=np.matmul,
                              training_data=training_data,
                              validation_data=validation_data,
                              )

    print('Time to train using np.matmul: ' + str(train_time) + ' seconds')

    # testing neural network
    accuracy = nn.validate(test_data) / 100.0
    print("Test Accuracy: " + str(accuracy) + "%")


if __name__ == "__main__":
    main()
