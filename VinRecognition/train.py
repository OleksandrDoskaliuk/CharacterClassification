import argparse

import tensorflow as tf
from pandas import DataFrame, Series
import pandas as pd
from sklearn.model_selection import train_test_split


default_model_name = 'model.h5'
default_dataset_characters_path = './dataset/characters/A_Z Handwritten Data.csv'
default_dataset_digits_path = './dataset/digits/mnist_train.csv'

IMAGE_WIDTH = IMAGE_HEIGHT = 28
IMAGE_CHANNELS = 1

UNUSED_CHARACTERS_IN_VIN_CODE = {8: 'I', 14: 'O', 16: 'Q'}


def main():
    model_name, dataset_digits_path, dataset_characters_path = parse_cli_args()

    dataset_digits, dataset_characters = load_datasets(dataset_digits_path, dataset_characters_path)

    labels, x_train_normalized, x_validation_normalized, y_train_re, y_validation_re = prepare_data(dataset_digits, dataset_characters)

    model = create_model(labels)

    train_model(model, x_train_normalized, x_validation_normalized, y_train_re, y_validation_re)

    model.save(model_name)


def parse_cli_args() -> (str, str, str):
    parser = argparse.ArgumentParser(
        description='This script is designed to create a digit recognition model and testing it using MNIST data')
    parser.add_argument('-m', '--model', type=str, default=default_model_name,
                        help='Model name. Note: provide full path to the model if it is not located in work dir')

    parser.add_argument('--dataset_digits', type=str, default=default_dataset_digits_path,
                        help='Path to the digits CSV dataset')

    parser.add_argument('--dataset_characters', type=str, default=default_dataset_characters_path,
                        help='Path to the characters CSV dataset')

    args = parser.parse_args()

    return args.model, args.dataset_digits, args.dataset_characters


def load_datasets(dataset_digits_path: str, dataset_characters_path: str) -> (DataFrame, DataFrame):
    dataset_digits = pd.read_csv(dataset_digits_path).astype('float32')
    dataset_characters = pd.read_csv(dataset_characters_path).astype('float32')

    return dataset_digits, dataset_characters


def prepare_data(dataset_digits: DataFrame, dataset_characters: DataFrame):
    dataset_characters = remove_unused_vin_chars(dataset_characters)
    increment_label_for_digits_dataset(dataset_digits, dataset_characters)
    set_same_column_names(dataset_digits, dataset_characters)

    training_dataset_digits, testing_dataset_digits = train_test_split(dataset_digits, test_size=0.2, random_state=777)
    training_dataset_chars, testing_dataset_chars = train_test_split(dataset_characters, test_size=0.2, random_state=777)

    train_data = union_and_shuffle(training_dataset_digits, training_dataset_chars)
    testing_data = union_and_shuffle(testing_dataset_digits, testing_dataset_chars)

    x_train, y_train = separate_data_and_labels(train_data)
    x_validation, y_validation = separate_data_and_labels(testing_data)

    labels = pd.concat([y_train, y_validation], ignore_index=True)

    x_train_re, x_validation_re, y_train_re, y_validation_re = reshape_to_arrays(x_train, x_validation, y_train, y_validation)

    x_train_with_channels = with_channel(x_train_re)
    x_validation_with_channels = with_channel(x_validation_re)

    x_train_normalized = normalize(x_train_with_channels)
    x_validation_normalized = normalize(x_validation_with_channels)

    return labels, x_train_normalized, x_validation_normalized, y_train_re, y_validation_re


# VIN characters may be capital letters A through Z and numbers 1 through 0; however, the letters I, O and Q are never used in order to avoid mistakes of misreading
def remove_unused_vin_chars(dataset_characters: DataFrame) -> DataFrame:
    dataset_characters_copy = dataset_characters.loc[~dataset_characters.iloc[:, 0].isin(UNUSED_CHARACTERS_IN_VIN_CODE.keys())].copy()

    dataset_characters_copy.loc[(dataset_characters_copy.iloc[:, 0] > 8) & (dataset_characters_copy.iloc[:, 0] < 14), '0'] -= 1
    dataset_characters_copy.loc[(dataset_characters_copy.iloc[:, 0] > 14) & (dataset_characters_copy.iloc[:, 0] < 16), '0'] -= 2
    dataset_characters_copy.loc[(dataset_characters_copy.iloc[:, 0] > 16), '0'] -= 3
    return dataset_characters_copy


def increment_label_for_digits_dataset(dataset_digits: DataFrame, dataset_characters: DataFrame):
    dataset_digits.iloc[:, 0] += len(dataset_characters.iloc[:, 0].unique())


def set_same_column_names(dataset_digits: DataFrame, dataset_characters: DataFrame):
    dataset_digits.columns = dataset_characters.columns


def union_and_shuffle(dataset_digits: DataFrame, dataset_characters: DataFrame) -> DataFrame:
    union_df = pd.concat([dataset_characters, dataset_digits], ignore_index=True)
    union_shuffled_df = union_df.sample(frac=1).reset_index(drop=True)

    return union_shuffled_df


def separate_data_and_labels(dataset: DataFrame) -> (DataFrame, Series):
    data = dataset.iloc[:, 1:785]
    labels = dataset.iloc[:, 0]

    return data, labels


def reshape_to_arrays(x_train, x_validation, y_train, y_validation):
    x_train_re = x_train.to_numpy().reshape(len(x_train), 28, 28)
    y_train_re = y_train.values
    x_validation_re = x_validation.to_numpy().reshape(len(x_validation), 28, 28)
    y_validation_re = y_validation.values

    return x_train_re, x_validation_re, y_train_re, y_validation_re


def with_channel(x_re):
    x_re_with_channels = x_re.reshape(
        x_re.shape[0],
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        IMAGE_CHANNELS
    )

    return x_re_with_channels


def normalize(x):
    x_normalized = x / 255.
    return x_normalized


def create_model(labels: Series):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Convolution2D(
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
        kernel_size=5,
        filters=8,
        strides=1,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling()
    ))

    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ))
    model.add(tf.keras.layers.Convolution2D(
        kernel_size=5,
        filters=16,
        strides=1,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling()
    ))

    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(
        units=128,
        activation=tf.keras.activations.relu
    ))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(len(labels.unique()), activation='softmax'))

    # For mac M1/M2 it is recommended to use legacy Adam optimizer
    adam_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    model.compile(
        optimizer=adam_optimizer,
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )

    return model


def train_model(model, x_train_normalized, x_validation_normalized, y_train_re, y_validation_re):
    model.fit(x_train_normalized, y_train_re, validation_data=(x_validation_normalized, y_validation_re), epochs=18,
              batch_size=200, verbose=2)


if __name__ == '__main__':
    main()
