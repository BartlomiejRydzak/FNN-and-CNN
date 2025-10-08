import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing import image

# print(tf.__version__)

class cnn:
    def __init__(self):
        self.model = None

    def train(self, name="cnn_cats_dogs.h5"):
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

        training_set = train_datagen.flow_from_directory(
                "dataset/training_set",
                target_size=(64, 64),
                batch_size=32,
                class_mode='binary')

        test_datagen = ImageDataGenerator(rescale=1./255)

        test_set = test_datagen.flow_from_directory(
                "dataset/test_set",
                target_size=(64, 64),
                batch_size=32,
                class_mode='binary')

        cnn = tf.keras.models.Sequential()

        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=[64, 64, 3]))

        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        cnn.add(tf.keras.layers.Flatten())

        cnn.add(tf.keras.layers.Dense(units=128, activation="relu"))

        cnn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

        cnn.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

        cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

        cnn.save(name)

        self.model = load_model(name)

    def train_with_gpu(self):
        with tf.device("/GPU:0"):
            self.train()

    def load_model(self, name="cnn_cats_dogs.h5"):
        self.model = load_model(name)

    def predict_from_path(self, path):
        test_image = image.load_img(path, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = self.model.predict(test_image)

        if result[0][0] == 1:
            print("pies")
        else:
            print("kot")

    def predict_from_list(self, images):
        results = []

        for img in images:
            # test_image = image.load_img("dataset/pies.jpg", target_size = (64, 64))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            result = self.model.predict(img)
            # training_set.class_indices
            results.append(result)

        for result in results:
            if result[0][0] == 1:
                print("pies")
            else:
                print("kot")

if __name__ == "__main__":
    my_cnn = cnn()
    my_cnn.load_model()

    images = []
    images.append(image.load_img("dataset/pies2.jpg", target_size = (64, 64)))
    images.append(image.load_img("dataset/kot.jpg", target_size = (64, 64)))
    images.append(image.load_img("dataset/pies.jpg", target_size = (64, 64)))
    images.append(image.load_img("dataset/kot2.png", target_size = (64, 64)))

    images.append(image.load_img("dataset/kot3.jpg", target_size = (64, 64)))
    images.append(image.load_img("dataset/kot4.jpg", target_size = (64, 64))) #zle
    images.append(image.load_img("dataset/kot5.jpg", target_size = (64, 64)))
    images.append(image.load_img("dataset/pies3.jpg", target_size = (64, 64)))
    images.append(image.load_img("dataset/pies4.jpg", target_size = (64, 64)))
    
    my_cnn.predict_from_list(images)

    my_cnn.predict_from_path("dataset/kot.jpg")