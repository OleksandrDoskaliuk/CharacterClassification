from numpy import ndarray
import tensorflow as tf
import tensorflow_datasets as tfds
import argparse

default_model_name = 'model.h5'


def main():
    model_name = parse_cli_args()

    (ds_train, ds_test), ds_info = load_mnist_data()

    create_new_model_file(model_name, ds_train, ds_test, ds_info)


def parse_cli_args() -> str:
    parser = argparse.ArgumentParser(
        description='This script is designed to create a digit recognition model and testing it using MNIST data')
    parser.add_argument('-m', '--model', type=str, default=default_model_name,
                        help='Model name. Note: provide full path to the model if it is not located in work dir')

    args = parser.parse_args()

    return args.model


def create_new_model_file(model_name: str, ds_train, ds_test, ds_info):
    ds_train = prepare_training_data(ds_train, ds_info)
    ds_test = prepare_testing_data(ds_test)

    model = create_model()

    train_model(model, ds_train, ds_test)

    save_model(model, model_name)


def load_mnist_data():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    return (ds_train, ds_test), ds_info


def prepare_training_data(ds_train, ds_info):
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    return ds_train


def prepare_testing_data(ds_test):
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    return ds_test


def normalize_pictures_array(pictures: ndarray) -> ndarray:
    return tf.keras.utils.normalize(pictures, axis=1)


def create_model():
    picture_shape = (28, 28)
    units_number = 128
    output_digits_number = 10

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=picture_shape),
        tf.keras.layers.Dense(units_number, activation='relu'),
        tf.keras.layers.Dense(units_number, activation='relu'),
        tf.keras.layers.Dense(output_digits_number)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model


def train_model(model, ds_train, ds_test):
    epochs = 10

    model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_test
    )


def save_model(model, model_name):
    model.save(model_name)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


if __name__ == '__main__':
    main()
