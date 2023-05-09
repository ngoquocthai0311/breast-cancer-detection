import models
import numpy as np
import helper

EPOCHS = 20


def train_and_test_proposed_model(x_train, x_test, y_train, y_test, epochs = EPOCHS, verbose = 2):
    model = models.get_proposed_model()
    # Fit model on training data
    model.fit(x_train, y_train, epochs=epochs)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=verbose)

def train_and_test_pretrained_vgg16_model(x_train, x_test, y_train, y_test, epochs = EPOCHS, verbose = 2, transfer_learning = False):
    vgg_16_model = models.get_pretrained_vgg16(transfer_learning)
    vgg_16_model.fit(np.array(helper.convert_to_rgb(x_train)), y_train, epochs=epochs)
    vgg_16_model.evaluate(np.array(helper.convert_to_rgb(x_test)), y_test, verbose=verbose)

def train_and_test_pretrained_vgg19_model(x_train, x_test, y_train, y_test, epochs = EPOCHS, verbose = 2, transfer_learning = False):
    vgg_19_model = models.get_pretrained_vgg19(transfer_learning)
    vgg_19_model.fit(np.array(helper.convert_to_rgb(x_train)), y_train, epochs=epochs)
    vgg_19_model.evaluate(np.array(helper.convert_to_rgb(x_test)), y_test, verbose=verbose)

def train_and_test_alexnet_model(x_train, x_test, y_train, y_test, epochs = EPOCHS, verbose = 2):
    alexnet_model = models.get_alexnext_model()
    # Fit model on training data
    alexnet_model.fit(x_train, y_train, epochs=epochs)

    # Evaluate neural network performance
    alexnet_model.evaluate(x_test,  y_test, verbose=verbose)