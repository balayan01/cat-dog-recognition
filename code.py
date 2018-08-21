from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())

classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units =1, activation="sigmoid"))

classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

#image preprocessing

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('training_set', target_size=(64, 64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory('test_set', target_size=(64, 64), batch_size=32, class_mode='binary')


classifier.fit_generator(training_set, samples_per_epoch=8000, epochs=3, validation_data=test_set, nb_val_samples=1000)

training_set.class_indices

#Predicting image
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('cat.4147.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
prediction = classifier.predict(test_image)
print(prediction)
