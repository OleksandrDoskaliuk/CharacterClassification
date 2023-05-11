import argparse

import numpy as np
import tensorflow as tf
import logging
import os
import cv2

default_model_name = 'model.h5'
valid_img_extensions = {'.png', '.jpg', '.jpeg'}

alphabets_mapper = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                    12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
                    23: 'X', 24: 'Y', 25: 'Z'}


def main():
    img_dir, model_name = def_directory_path_and_model_name_from_cli()

    model = load_model(model_name)

    files = list_valid_files_in_folder(img_dir)

    infer_digits(model, img_dir, files)


def def_directory_path_and_model_name_from_cli():
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


def list_valid_files_in_folder(img_dir: str):
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


def infer_digits(model, directory: str, files: list[str]):
    for file in files:
        file_path = os.path.join(directory, file)
        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            index = np.argmax(prediction)
            print(f'{alphabets_mapper[index]} for file: {file_path}')
        except Exception as e:
            print("Error", e)


if __name__ == '__main__':
    main()
