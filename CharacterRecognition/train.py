import argparse
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

default_model_name = 'model.h5'
default_dataset_path = './dataset/A_Z Handwritten Data.csv'

alphabets_mapper = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                    12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
                    23: 'X', 24: 'Y', 25: 'Z'}


def main():
    model_name, dataset_path = parse_cli_args()

    data, labels = load_dataset(dataset_path)

    create_new_model_file(model_name, data, labels)


def parse_cli_args() -> (str, str):
    parser = argparse.ArgumentParser(
        description='This script is designed to create a digit recognition model and testing it using MNIST data')
    parser.add_argument('-m', '--model', type=str, default=default_model_name,
                        help='Model name. Note: provide full path to the model if it is not located in work dir')

    parser.add_argument('-d', '--dataset', type=str, default=default_dataset_path,
                        help='Dataset path')

    args = parser.parse_args()

    return args.model, args.dataset


def load_dataset(path: str):
    try:
        dataset = pd.read_csv(path).astype('float32')
        dataset.rename(columns={'0': 'label'}, inplace=True)

        data = dataset.drop('label', axis=1)

        labels = dataset['label']

        return data, labels
    except Exception as e:
        logging.exception("Exception when loading dataset!")
        raise e


def create_new_model_file(model_name, data, labels):
    x_train, y_train, x_test, y_test = prepare_data(data, labels)

    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(labels.unique()), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=18, batch_size=200, verbose=2)

    model.save(model_name)


def prepare_data(data, labels):

    x_train, x_test, y_train, y_test = train_test_split(data, labels)

    standard_scaler = MinMaxScaler()
    standard_scaler.fit(x_train)

    x_train = standard_scaler.transform(x_train)
    x_test = standard_scaler.transform(x_test)

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    main()
