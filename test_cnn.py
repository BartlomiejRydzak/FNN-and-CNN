import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model


cnn = load_model("cnn_cats_dogs.h5")

results = []

test_image = image.load_img("dataset/pies.jpg", target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
# training_set.class_indices
results.append(result)

test_image = image.load_img("dataset/kot.jpg", target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
# training_set.class_indices
results.append(result)

test_image = image.load_img("dataset/kot2.png", target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
# training_set.class_indices
results.append(result)

test_image = image.load_img("dataset/pies2.jpg", target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
# training_set.class_indices
results.append(result)

for result in results:
    if result[0][0] == 1:
        print("pies")
    else:
        print("kot")