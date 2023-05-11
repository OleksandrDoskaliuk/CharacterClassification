from numpy import ndarray
import tensorflow as tf
import argparse

default_model_name = 'model.h5'


def main():
    model_name, testing = parse_cli_args()

    (picture_train, label_train), (picture_test, label_test) = load_mnist_data()

    if testing == 'y':
        test_existing_model(model_name, picture_test, label_test)
    else:
        create_new_model_file(model_name, picture_train, label_train)


def parse_cli_args() -> (str, str):
    parser = argparse.ArgumentParser(
        description='This script is designed to create a digit recognition model and testing it using MNIST data')
    parser.add_argument('-m', '--model', type=str, default=default_model_name,
                        help='Model name. Note: provide full path to the model if it is not located in work dir')
    parser.add_argument('-t', '--testing', type=str, choices=['y', 'n'], default='n',
                        help='Test model flag. If `y` then test provided model. If `n` the new model will be created. Default - `n`')

    args = parser.parse_args()

    return args.model, args.testing


def create_new_model_file(model_name: str, x_train, y_train):
    picture_train_normalized = normalize_pictures_array(x_train)

    model = create_model()

    compile_model(model)

    train_model(model, picture_train_normalized, y_train)

    save_model(model, model_name)


def test_existing_model(model_name: str, x_test, y_test):
    model = tf.keras.models.load_model(model_name)

    loss, accuracy = model.evaluate(x_test, y_test)

    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")


def load_mnist_data() -> ((ndarray, ndarray), (ndarray, ndarray)):
    mnist = tf.keras.datasets.mnist

    (picture_train, label_train), (picture_test, label_test) = mnist.load_data()

    return (picture_train, label_train), (picture_test, label_test)


def normalize_pictures_array(pictures: ndarray) -> ndarray:
    return tf.keras.utils.normalize(pictures, axis=1)


def create_model():
    picture_shape = (28, 28)
    units_number = 128
    output_digits_number = 10

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=picture_shape))
    model.add(tf.keras.layers.Dense(units_number, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units_number, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(output_digits_number, activation=tf.nn.softmax))

    return model


def compile_model(model):
    optimizer = 'adam'
    loss = 'sparse_categorical_crossentropy'
    metrics = ['accuracy']

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def train_model(model, x_train, y_train):
    epochs = 3

    model.fit(x_train, y_train, epochs=epochs)


def save_model(model, model_name):
    model.save(model_name)


if __name__ == '__main__':
    main()
