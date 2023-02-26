#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.models import Sequential
from tensorflow.python.data import Dataset
from tensorflow.python.keras.callbacks import History


def ask_resume_learning() -> bool:
    while True:
        answer = input(
            "A checkpoint has been found. Should the program restart from it? [Y/n] "
        )
        if answer.lower() in ["y", "yes", None, ""]:
            return True
        elif answer.lower() in ["n", "no"]:
            return False
        else:
            print(f"Please answer yes or no instead of {answer}")


def create_model(n_classes: int, height: int, width: int) -> Sequential:
    data_augmentation = Sequential(
        [
            layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    m = Sequential(
        [
            data_augmentation,
            layers.Rescaling(1.0 / 255, input_shape=(height, width, 3)),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(n_classes),
        ]
    )
    m.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return m


def load_data(train_ds_path: str, validate_ds_path: str, size: tuple = None) -> tuple:
    train = tf.keras.utils.image_dataset_from_directory(train_ds_path, image_size=size)
    val = tf.keras.utils.image_dataset_from_directory(validate_ds_path, image_size=size)
    return train, val


def score_model(hist: History, n_epochs: int) -> None:
    acc = hist.history["accuracy"]
    val_acc = hist.history["val_accuracy"]

    loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]

    epochs_range = range(n_epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.show()


def train_model(
    train: Dataset,
    validate: Dataset,
    learning_model: Model,
    n_epochs: int,
    save_model: bool = True,
    checkpoint_name: str = "model_checkpoint.ckpt",
) -> History:
    # keep the images in memory after they're loaded off disk during the first epoch.
    # This will ensure the dataset does not become a bottleneck while training your model.
    t_ds = train.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    v_ds = validate.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_name, save_weights_only=True, verbose=1
    )

    # train the model
    hist = learning_model.fit(
        t_ds, validation_data=v_ds, epochs=n_epochs, callbacks=[cp_callback]
    )
    if save_model:
        learning_model.save("sequential_resulting_model")

    return hist


if __name__ == "__main__":
    # Params of the model
    test_path = "test_set"
    train_path = "training_set"
    img_height = 256
    img_width = 256
    epochs = 10
    checkpoint_path = "model_checkpoint.ckpt"

    # load the train and validation datasets
    train_ds, val_ds = load_data(test_path, train_path, (img_height, img_width))
    class_names = train_ds.class_names
    print(f"Loaded images with {len(class_names)} classes: {class_names}")

    # check if checkpoints are available
    resume = False
    if checkpoint_path + ".index" in os.listdir("."):
        resume = ask_resume_learning()

    if resume:
        model = create_model(len(class_names), img_height, img_width)
        model.load_weights(checkpoint_path)

    else:
        # prepare model
        model = create_model(len(class_names), img_height, img_width)
        print("Created learning model with the following summary:")
        print(model.summary())

    # let's now train our model
    history = train_model(train_ds, val_ds, model, epochs)

    # let's score our model
    score_model(history, epochs)
