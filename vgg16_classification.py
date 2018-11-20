import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.applications import VGG16, InceptionV3
from keras.optimizers import Adam, RMSprop
import os
from Utils import get_batch, generator



def print_layer_trainable(conv_model):
    for layer in conv_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))


def plot_training_history(history):
    # Get the classification accuracy and loss-value
    # for the training-set.
    acc = history.history['categorical_accuracy']
    loss = history.history['loss']

    # Get it for the validation-set (we only use the test-set).
    val_acc = history.history['val_categorical_accuracy']
    val_loss = history.history['val_loss']

    # Plot the accuracy and loss-values for the training-set.
    plt.plot(acc, linestyle='-', color='b', label='Training Acc.')
    plt.plot(loss, 'o', color='b', label='Training Loss')

    # Plot it for the test-set.
    plt.plot(val_acc, linestyle='--', color='r', label='Test Acc.')
    plt.plot(val_loss, 'o', color='r', label='Test Loss')

    # Plot title and legend.
    plt.title('Training and Test Accuracy')
    plt.legend()

    # Ensure the plot shows correctly.
    plt.show()






if __name__ == '__main__':


    model = VGG16(include_top=True, weights='imagenet')
    input_shape = model.layers[0].output_shape[1:3]
    print(input_shape)

    generator_train = generator(dir_path='Summer 2018 Pics/train', input_size=224, batch_size=4)

    generator_test = generator(dir_path='Summer 2018 Pics/test', input_size=224, batch_size=4)


    # steps_test = generator_test.n / batch_size
    # cls_train = generator_train.classes
    steps_test = 8
    num_classes = 3


    model.summary()
    transfer_layer = model.get_layer('block5_pool')
    conv_model = Model(inputs=model.input,
                       outputs=transfer_layer.output)

    # w1 = conv_model.get_weights()

    # Start a new Keras Sequential model.
    new_model = Sequential()

    # Add the convolutional part of the VGG16 model from above.
    new_model.add(conv_model)

    # Flatten the output of the VGG16 model because it is from a
    # convolutional layer.
    new_model.add(Flatten())

    # Add a dense (aka. fully-connected) layer.
    # This is for combining features that the VGG16 model has
    # recognized in the image.
    new_model.add(Dense(512,activation='relu'))

    # Add a dropout-layer which may prevent overfitting and
    # improve generalization ability to unseen data e.g. the test-set.
    new_model.add(Dropout(0.7))

    # Add the final layer for the actual classification.
    new_model.add(Dense(num_classes, activation='softmax'))

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 1e-4
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               1000, 0.86, staircase=True)


    optimizer = Adam(lr=learning_rate)
    loss = 'categorical_crossentropy'
    metrics = ['categorical_accuracy']

    new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    epochs = 20
    steps_per_epoch = 16

    # w2 = new_model.get_weights()

    conv_model.trainable = False
    for layer in conv_model.layers:
        layer.trainable = False
    print_layer_trainable(conv_model)


    history = new_model.fit_generator(generator=generator_train,
                                      epochs=epochs,
                                      steps_per_epoch=steps_per_epoch,
                                      # class_weight='Balanced',
                                      validation_data=generator_test,
                                      validation_steps=steps_test)

    plot_training_history(history)
    result = new_model.evaluate_generator(generator_test, steps=steps_test)
    print("Test-set classification accuracy: {0:.2%}".format(result[1]))

    saver = tf.train.Saver()
    save_dir = 'checkpoints/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    new_model.save(os.path.join(save_dir, 'vgg16_classification_V2.h5'))  # creates a HDF5 file





