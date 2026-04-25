import os
import random
import logging

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KERAS_HOME"] = "/tmp/edge-ai-keras"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42
MODEL_PATH = "model.h5"
EPOCHS = 5


def configure_environment():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    tf.get_logger().setLevel("ERROR")
    logging.getLogger("absl").setLevel(logging.ERROR)

    try:
        tf.config.set_visible_devices([], "GPU")
    except (RuntimeError, ValueError):
        pass


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)


def build_model():
    return keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(10, activation="softmax"),
        ]
    )


def compute_macro_metrics(y_true, y_pred, num_classes=10):
    confusion_matrix = tf.math.confusion_matrix(
        y_true, y_pred, num_classes=num_classes
    ).numpy()
    confusion_matrix = confusion_matrix.astype(np.float64)

    true_positives = np.diag(confusion_matrix)
    predicted_positives = confusion_matrix.sum(axis=0)
    actual_positives = confusion_matrix.sum(axis=1)

    precision_per_class = np.divide(
        true_positives,
        predicted_positives,
        out=np.zeros_like(true_positives),
        where=predicted_positives != 0,
    )
    recall_per_class = np.divide(
        true_positives,
        actual_positives,
        out=np.zeros_like(true_positives),
        where=actual_positives != 0,
    )
    f1_per_class = np.divide(
        2 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class,
        out=np.zeros_like(true_positives),
        where=(precision_per_class + recall_per_class) != 0,
    )

    return (
        precision_per_class.mean(),
        recall_per_class.mean(),
        f1_per_class.mean(),
    )


def print_final_metrics(loss, accuracy, macro_precision, macro_recall, macro_f1):
    separator = "=" * 54

    print()
    print(separator)
    print("RESULTADO DO TREINAMENTO - METRICAS FINAIS")
    print(separator)
    print(f"Loss:             {loss:.4f}")
    print(f"Accuracy:         {accuracy:.4f}")
    print(f"Precision (macro): {macro_precision:.4f}")
    print(f"Recall (macro):    {macro_recall:.4f}")
    print(f"F1-Score (macro):  {macro_f1:.4f}")
    print(separator)


def main():
    configure_environment()
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=128,
        validation_split=0.1,
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    macro_precision, macro_recall, macro_f1 = compute_macro_metrics(
        y_test, predicted_labels
    )

    print_final_metrics(
        test_loss,
        test_accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
    )

    model.save(MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
