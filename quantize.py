import tensorflow as tf
import numpy as np
import glob
import cv2

converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")

converter.optimizations = [tf.lite.Optimize.DEFAULT]

data_files = glob.glob("data/training/*/*.jpg")
np.random.shuffle(data_files)

IMG_SIZE = (96, 96)

def representative_dataset():
    for file_path in data_files[:200]:
        img = cv2.imread(file_path)

        # convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # resize
        img = cv2.resize(img, IMG_SIZE)

        # normalize
        img = img.astype(np.float32) / 255.0

        # add channel dimensions
        img = np.expand_dims(img, axis=0)      # (1, 96, 96)

        img = np.expand_dims(img, axis=0)      # (1, 1, 96, 96)


        yield [img]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("model_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("model_int8.tflite created")
