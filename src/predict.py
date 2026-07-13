import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from config import BEST_MODEL_PATH, IMAGE_SIZE


print("Loading trained model...")

model = tf.keras.models.load_model(BEST_MODEL_PATH)

print("Model Loaded Successfully!")


class_names = {
    0: "Benign",
    1: "Malignant"
}


def predict_image(img_path):

    print("Loading image...")

    img = image.load_img(img_path, target_size=IMAGE_SIZE)

    plt.imshow(img)
    plt.axis("off")
    plt.show()

    print("Image displayed successfully.")

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    print("Running prediction...")

    prediction = model.predict(img_array, verbose=0)

    print("Raw Prediction:", prediction)

    prediction = prediction[0][0]

    if prediction >= 0.5:
        label = "Malignant"
        confidence = prediction
    else:
        label = "Benign"
        confidence = 1 - prediction

    print("\nPrediction:", label)
    print(f"Confidence: {confidence*100:.2f}%")


if __name__ == "__main__":

    image_path = input("Enter Image Path: ")

    predict_image(image_path)