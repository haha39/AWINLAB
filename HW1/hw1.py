import os

import random
import pandas as pd

from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

# import matplotlib.pyplot as plt
# import numpy as np

from PIL import Image


class DogClassifier:
    def __init__(self):
        self.model = self.build_model()
        # Add the class_names attribute
        self.class_names = ["Airedale", "Beagle", "Bloodhound", "Bluetick", "Chihuahua", "Collie", "Dingo",
                            "French Bulldog", "German Sheperd", "Malinois", "Newfoundland", "Pekinese",
                            "Pomeranian", "Pug", "Vizsla"]

    def build_model(self):
        '''
        Create a convolutional neural network model(CNN) :
        With one separate input layer, three convolutional layers and a maximum pooling layer
        '''

        model = Sequential()

        model.add(Input(shape=(224, 224, 3)))

        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        # Flattenning 3D feature maps into 1D vectors
        model.add(Flatten())

        # Add full connectivity layer
        model.add(Dense(512, activation='relu'))

        # Add Dropout layer to prevent overfitting
        model.add(Dropout(0.5))

        # Output layer, outputs classification results(there are 15 dog breed categories)
        model.add(Dense(15, activation='softmax'))

        # compiling the model
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Printed Model Structures
        # model.summary()
        return model

    def train(self, train_dir, valid_dir, batch_size, epochs):

        # Preprocessing and enhancement of training and validation data
        train_datagen = ImageDataGenerator(rescale=1./255)
        valid_datagen = ImageDataGenerator(rescale=1./255)

        # Generating data streams for training and validation sets
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical'
        )

        valid_generator = valid_datagen.flow_from_directory(
            valid_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical'
        )

        # Training the model
        self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=valid_generator,
            validation_steps=valid_generator.samples // batch_size,
        )

    def evaluate(self, valid_dir, batch_size):

        # Preprocessing and enhancement of validation data
        valid_datagen = ImageDataGenerator(rescale=1./255)

        # Generating data streams for validation sets
        valid_generator = valid_datagen.flow_from_directory(
            valid_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical'
        )

        # Calculating accuracy
        scores = self.model.evaluate(
            valid_generator, steps=valid_generator.samples // batch_size)
        validation_accuracy = scores[1] * 100

        print("Valid set Accuracy: %.2f%%" % validation_accuracy)

        # Output validation accuracy into Excel(no need title)
        df = pd.DataFrame({'Accuracy': [validation_accuracy]})
        df.to_excel('validation_accuracy.xlsx', index=False)

        print("Validation accuracy saved to validation_accuracy.xlsx")

    def center_crop_image(self, img):
        '''
        Pre-processing of images, with different sized images as input,
          and centered cropped images as output.
        '''

        width, height = img.size
        new_width = new_height = min(width, height)
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = (width + new_width) // 2
        bottom = (height + new_height) // 2

        return img.crop((left, top, right, bottom))

    def get_predicted_breed(self, predictions):
        '''
        To get the dog breed name based on the model predictions,
        assume that predictions is the result of the model's prediction of the image,
        determine the predicted breed based on the index of the highest probability in the probability vector
        '''

        breed_index = predictions.argmax()
        breed_name = self.class_names[breed_index]

        return breed_name

    def test(self, test_dir):

        # Getting test set file address
        test_files = os.listdir(test_dir)
        # To be fair, the order of access is randomized.
        random.shuffle(test_files)

        test_results = []

        for file_name in test_files:

            # Load image with center crop and preprocessing
            img_path = os.path.join(test_dir, file_name)
            img = Image.open(img_path)
            # Center Cropped Image
            img = self.center_crop_image(img)
            # resize
            img = img.resize((224, 224))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            img_array = img_array.reshape((1,) + img_array.shape)

            # Making predictions about the image
            predictions = self.model.predict(img_array)
            predicted_breed = self.get_predicted_breed(predictions)

            test_results.append((file_name, predicted_breed))

        # Output results into Excel(no need title)
        df = pd.DataFrame(test_results, columns=[
                          'File Name', 'Predicted Breed'])
        df.to_excel('test_data.xlsx', index=False, header=False)
        print("Test results saved to test_data.xlsx")


def main():
    classifier = DogClassifier()
    train_dir = 'archive/train'
    valid_dir = 'archive/valid'
    test_dir = 'archive/testing_set'
    batch_size = 32
    epochs = 10

    classifier.train(train_dir, valid_dir, batch_size, epochs)
    classifier.evaluate(valid_dir, batch_size)
    classifier.test(test_dir)


if __name__ == "__main__":
    main()
