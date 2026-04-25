import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf


MODEL_PATH = "model.h5"
TFLITE_PATH = "model.tflite"


def format_size(size_bytes):
    return f"{size_bytes / 1024:.2f} KB"


def configure_environment():
    try:
        tf.config.set_visible_devices([], "GPU")
    except (RuntimeError, ValueError):
        pass


def main():
    configure_environment()

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Required model file '{MODEL_PATH}' was not found. Run train_model.py first."
        )

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(TFLITE_PATH, "wb") as file:
        file.write(tflite_model)

    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()

    h5_size = os.path.getsize(MODEL_PATH)
    tflite_size = os.path.getsize(TFLITE_PATH)

    print(f"Saved optimized model to {TFLITE_PATH}")
    print(f"{MODEL_PATH} size: {format_size(h5_size)}")
    print(f"{TFLITE_PATH} size: {format_size(tflite_size)}")


if __name__ == "__main__":
    main()
