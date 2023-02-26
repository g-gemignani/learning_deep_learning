#!/usr/bin/env python3

import argparse
from typing import Union

import numpy as np

import tensorflow as tf


def evaluate(args) -> None:
    print(type(args))
    model = tf.keras.models.load_model("sequential_resulting_model")
    print("Loaded saved model with the following summary:")
    print(model.summary())

    image = tf.keras.utils.load_img(
        args.image_path, target_size=get_size_from_model(model)
    )
    input_arr = tf.keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.

    predictions = tf.math.sigmoid(model.predict(input_arr))
    print("The probability of the input image containing a label is:")
    for idx, x in enumerate(predictions[0]):
        print(f"\t- label {idx}: {x}")


def get_size_from_model(model) -> Union[tuple, None]:
    for layer in model.get_config().get("layers"):
        if layer.get("class_name") == "Rescaling":
            shape = layer.get("config").get("batch_input_shape")
            return shape[1], shape[2]
    return ()


def prepare_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="Classify.py",
        description="Given an image in input, the program will print the probability "
        "that a dog or cat is part of the picture",
    )
    p.add_argument("image_path", help="path of the image that needs to be classified")
    return p


if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()
    evaluate(args)
