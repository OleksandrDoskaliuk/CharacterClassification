import argparse
from typing import Any

import numpy as np
import tensorflow as tf
import logging
import os
import cv2

tf.keras.utils.disable_interactive_logging()

default_model_name = 'model.h5'
valid_img_extensions = {'.png', '.jpg', '.jpeg'}

NUMBER_OF_IMAGES_TO_DEFINE_BACKGROUND = 10

IMAGE_WIDTH = IMAGE_HEIGHT = 28

# MAPPER VIN. Letters I, O and Q are EXCLUDED
mapper = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'J', 9: 'K', 10: 'L',
          11: 'M', 12: 'N', 13: 'P', 14: 'R', 15: 'S', 16: 'T', 17: 'U', 18: 'V', 19: 'W',
          20: 'X', 21: 'Y', 22: 'Z', 23: '0', 24: '1', 25: '2', 26: '3', 27: '4', 28: '5',
          29: '6', 30: '7', 31: '8', 32: '9'}


def main():
    img_dir, model_name = gef_directory_path_and_model_name_from_cli()

    model = load_model(model_name)

    files = list_valid_files_in_folder(img_dir)

    images_with_path = load_images_with_path(img_dir, files)

    infer_pictures(model, images_with_path)


def gef_directory_path_and_model_name_from_cli():
    parser = argparse.ArgumentParser(
        description='This script is designed to find all images in the directory and infers the number on the image.')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the directory with images')

    parser.add_argument('-m', '--model', type=str, default=default_model_name,
                        help='Model name. Note: provide full path to the model if it is not located in work dir')

    args = parser.parse_args()

    return args.input, args.model


def load_model(model_name):
    try:
        return tf.keras.models.load_model(model_name)
    except Exception as e:
        logging.exception(
            "Can not find model %s! Please verify if the model file exists. Provide full path to the model if it is not located in same folder as the script",
            model_name)
        raise e


def list_valid_files_in_folder(img_dir: str) -> list[str]:
    files = os.listdir(img_dir)

    files_extension_dict = create_file_extension_dict(files)

    filtered_dict = filter_dict_by_valid_img_extensions(files_extension_dict)

    files_filtered = create_list_of_files_from_dict(filtered_dict)

    return files_filtered


def create_file_extension_dict(files: list[str]) -> dict[str: str]:
    files_extension_dict = {}

    for file_name in files:
        file_extension = os.path.splitext(file_name)[1]

        if file_extension in files_extension_dict:
            files_extension_dict[file_extension].append(file_name)
        else:
            files_extension_dict[file_extension] = [file_name]

    return files_extension_dict


def filter_dict_by_valid_img_extensions(d: dict[str: str]):
    filtered_d = {}
    for k in d.keys():
        if k in valid_img_extensions:
            filtered_d[k] = d[k]
    return filtered_d


def create_list_of_files_from_dict(d: dict[str: str]) -> list[str]:
    result_list = list()
    for value_list in d.values():
        result_list.extend(value_list)

    return result_list


def load_images_with_path(directory, file_names: list[str]) -> list[(str, Any)]:
    images_with_path = []
    for file_name in file_names:
        try:
            file_path = os.path.join(directory, file_name)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = resize_img(img)
            images_with_path.append((img, file_path))
        except Exception as e:
            logging.exception(f"Exception loading file: {file_name} in directory: {directory}")
    return images_with_path


def infer_pictures(model, images_with_path: list[(str, Any)]):
    img_to_np_array_function = get_img_to_np_array_function(images_with_path)

    for (img, img_path) in images_with_path:
        img_as_np_array = img_to_np_array_function(img)
        inferred_character = infer_character_on_image(model, img_as_np_array)

        ascii_index = str(ord(inferred_character))

        print(f'{ascii_index.zfill(3)}, {img_path}')


def infer_character_on_image(model, img_as_np_array):
    prediction = model.predict(img_as_np_array)
    mapper_key = np.argmax(prediction)
    return mapper[mapper_key]


def get_img_to_np_array_function(images: list[(str, Any)]):
    first_ten_images = [img_path[0] for img_path in images[:NUMBER_OF_IMAGES_TO_DEFINE_BACKGROUND]]

    average_background = define_average_background_color_for_images(first_ten_images)

    if average_background > 127:
        # WHITE ON BLACK
        return img_to_np_array_invert
    else:
        # BLACK ON WHITE
        return img_to_np_array


def img_to_np_array_invert(img):
    return np.invert(img_to_np_array(img))


def img_to_np_array(img):
    return np.array([img])


def define_average_background_color_for_images(images):
    return np.mean([np.mean(img) for img in images])


def resize_img(img):
    height, width = img.shape

    # Check if image is already 28x28
    if height == IMAGE_HEIGHT and width == IMAGE_HEIGHT:
        resized_img = img
    else:
        # Resize image to 28x28
        resized_img = cv2.resize(img, (28, 28))

    return resized_img


if __name__ == '__main__':
    main()
