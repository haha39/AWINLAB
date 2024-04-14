{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/haha39/AWINLAB/blob/main/version2/HW1/readme.txt\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Step 1 : Setting the initial value**"
      ],
      "metadata": {
        "id": "PLXWi8oMHCQa"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 306,
      "metadata": {
        "id": "uCERUSloZ10s"
      },
      "outputs": [],
      "source": [
        "import os\n",
        "\n",
        "import random\n",
        "\n",
        "import pandas as pd\n",
        "\n",
        "import matplotlib.pyplot as plt\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow import keras\n",
        "from keras.preprocessing.image import img_to_array\n",
        "from keras.applications.imagenet_utils import preprocess_input\n",
        "from keras.preprocessing.image import ImageDataGenerator\n",
        "from keras.models import Sequential\n",
        "from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization\n",
        "from keras.optimizers import RMSprop\n",
        "from keras.regularizers import l2\n",
        "from keras.callbacks import ReduceLROnPlateau\n"
      ],
      "metadata": {
        "id": "Ls7dwKkazU2l"
      },
      "execution_count": 307,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lOA2183_Twx_",
        "outputId": "bfad66e3-2d2e-4a6e-8400-6aef95af7db2"
      },
      "execution_count": 308,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Step 2 : Building the CNN model**"
      ],
      "metadata": {
        "id": "RRfD1lwbeQEU"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "* **新增了 Dropout 以及 L2regularization 技術，以避免 overfitting**"
      ],
      "metadata": {
        "id": "_OGHeZFcIMP3"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "'''\n",
        "Create a convolutional neural network model(CNN) :\n",
        "With one separate input layer, three convolutional layers and a maximum pooling layer\n",
        "'''\n",
        "\n",
        "model = Sequential()\n",
        "\n",
        "model.add(Input(shape=(224, 224, 3)))\n",
        "\n",
        "model.add(Conv2D(32, (3, 3), activation='relu'))\n",
        "model.add(MaxPooling2D((2, 2)))\n",
        "\n",
        "model.add(Conv2D(64, (3, 3), activation='relu'))\n",
        "model.add(MaxPooling2D((2, 2)))\n",
        "\n",
        "model.add(Conv2D(128, (3, 3), activation='relu'))\n",
        "model.add(MaxPooling2D((2, 2)))\n",
        "\n",
        "model.add(Conv2D(128, (3, 3), activation='relu'))\n",
        "model.add(MaxPooling2D((2, 2)))\n",
        "# Flattenning 3D feature maps into 1D vectors\n",
        "model.add(Flatten())\n",
        "\n",
        "# Add full connectivity layer\n",
        "model.add(Dense(512, activation='relu'))\n",
        "\n",
        "# Add Dropout layer to prevent overfitting\n",
        "model.add(Dropout(0.5))\n",
        "\n",
        "# Add a fully connected layer and apply L2 regularization\n",
        "model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.001)))\n",
        "\n",
        "# Output layer, outputs classification results(there are 15 dog breed categories)\n",
        "model.add(Dense(15, activation='softmax'))\n",
        "\n",
        "# compiling the model\n",
        "model.compile(optimizer='adam', loss='categorical_crossentropy',\n",
        "                      metrics=['accuracy'])\n",
        "\n",
        "# Printed Model Structures\n",
        "# model.summary()\n",
        "#return model"
      ],
      "metadata": {
        "id": "-u0aTqOyaCb7"
      },
      "execution_count": 309,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Step 3 : Training the CNN model and using valid set to test it**"
      ],
      "metadata": {
        "id": "yflu3OcDeRB4"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "* **優化 Learning Rate 的部分，以增加準確率**\n",
        "* **使用 `ImageDataGenerator()` 產生額外的影像資料作訓練，以避免 overfitting 以及增加準確率**\n"
      ],
      "metadata": {
        "id": "gE1DRyzgI62O"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "learning_rate_function = ReduceLROnPlateau(monitor='val_accuracy',\n",
        "                patience=3,\n",
        "                verbose=1,\n",
        "                factor=0.5,\n",
        "                min_lr=0.00001)"
      ],
      "metadata": {
        "id": "99aTTPVGS-dp"
      },
      "execution_count": 310,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#def train(train_dir, valid_dir, batch_size, epochs):\n",
        "\n",
        "train_dir = '/content/drive/MyDrive/archive/train'\n",
        "valid_dir = '/content/drive/MyDrive/archive/valid'\n",
        "\n",
        "batch_size = 100\n",
        "epochs = 20\n",
        "\n",
        "# Preprocessing and enhancement of training and validation data\n",
        "train_datagen = ImageDataGenerator(\n",
        "            rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)\n",
        "valid_datagen = ImageDataGenerator(rescale=1./255)\n",
        "\n",
        "# Generating data streams for training and validation sets\n",
        "train_generator = train_datagen.flow_from_directory(\n",
        "            train_dir,\n",
        "            target_size=(224, 224),\n",
        "            batch_size=batch_size,\n",
        "            class_mode='categorical'\n",
        "        )\n",
        "\n",
        "valid_generator = valid_datagen.flow_from_directory(\n",
        "            valid_dir,\n",
        "            target_size=(224, 224),\n",
        "            batch_size=batch_size,\n",
        "            class_mode='categorical'\n",
        "        )\n",
        "\n",
        "# Training the model\n",
        "train_history = model.fit(\n",
        "            train_generator,\n",
        "            steps_per_epoch=min(train_generator.samples // batch_size, 150),\n",
        "            epochs=epochs,\n",
        "            validation_data=valid_generator,\n",
        "            validation_steps=valid_generator.samples // batch_size,\n",
        "            callbacks=[learning_rate_function]\n",
        "        )\n"
      ],
      "metadata": {
        "id": "KHe-avw7bZrv",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "5f782495-a076-4593-b1a5-279c89b4b2dc"
      },
      "execution_count": 311,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found 1714 images belonging to 15 classes.\n",
            "Found 150 images belonging to 15 classes.\n",
            "Epoch 1/20\n",
            "17/17 [==============================] - 24s 1s/step - loss: 3.1110 - accuracy: 0.0948 - val_loss: 2.9721 - val_accuracy: 0.1000 - lr: 0.0010\n",
            "Epoch 2/20\n",
            "17/17 [==============================] - 22s 1s/step - loss: 2.8238 - accuracy: 0.1394 - val_loss: 2.6179 - val_accuracy: 0.1900 - lr: 0.0010\n",
            "Epoch 3/20\n",
            "17/17 [==============================] - 22s 1s/step - loss: 2.6238 - accuracy: 0.1828 - val_loss: 2.4917 - val_accuracy: 0.2300 - lr: 0.0010\n",
            "Epoch 4/20\n",
            "17/17 [==============================] - 22s 1s/step - loss: 2.4231 - accuracy: 0.2311 - val_loss: 2.2349 - val_accuracy: 0.3300 - lr: 0.0010\n",
            "Epoch 5/20\n",
            "17/17 [==============================] - 22s 1s/step - loss: 2.2902 - accuracy: 0.2745 - val_loss: 2.0987 - val_accuracy: 0.2900 - lr: 0.0010\n",
            "Epoch 6/20\n",
            "17/17 [==============================] - 22s 1s/step - loss: 2.1719 - accuracy: 0.2974 - val_loss: 2.1205 - val_accuracy: 0.3400 - lr: 0.0010\n",
            "Epoch 7/20\n",
            "17/17 [==============================] - 22s 1s/step - loss: 2.0810 - accuracy: 0.3414 - val_loss: 1.9258 - val_accuracy: 0.3600 - lr: 0.0010\n",
            "Epoch 8/20\n",
            "17/17 [==============================] - 22s 1s/step - loss: 2.0904 - accuracy: 0.3302 - val_loss: 1.9580 - val_accuracy: 0.3700 - lr: 0.0010\n",
            "Epoch 9/20\n",
            "17/17 [==============================] - 22s 1s/step - loss: 2.0061 - accuracy: 0.3649 - val_loss: 1.9174 - val_accuracy: 0.3900 - lr: 0.0010\n",
            "Epoch 10/20\n",
            "17/17 [==============================] - 23s 1s/step - loss: 1.8694 - accuracy: 0.4165 - val_loss: 1.6989 - val_accuracy: 0.4900 - lr: 0.0010\n",
            "Epoch 11/20\n",
            "17/17 [==============================] - 23s 1s/step - loss: 1.7562 - accuracy: 0.4406 - val_loss: 1.6186 - val_accuracy: 0.4600 - lr: 0.0010\n",
            "Epoch 12/20\n",
            "17/17 [==============================] - 23s 1s/step - loss: 1.6526 - accuracy: 0.4782 - val_loss: 1.7854 - val_accuracy: 0.4200 - lr: 0.0010\n",
            "Epoch 13/20\n",
            "17/17 [==============================] - 22s 1s/step - loss: 1.5866 - accuracy: 0.5006 - val_loss: 1.5443 - val_accuracy: 0.5200 - lr: 0.0010\n",
            "Epoch 14/20\n",
            "17/17 [==============================] - 22s 1s/step - loss: 1.6739 - accuracy: 0.4665 - val_loss: 1.8277 - val_accuracy: 0.3200 - lr: 0.0010\n",
            "Epoch 15/20\n",
            "17/17 [==============================] - 22s 1s/step - loss: 1.5817 - accuracy: 0.5093 - val_loss: 1.4842 - val_accuracy: 0.5300 - lr: 0.0010\n",
            "Epoch 16/20\n",
            "17/17 [==============================] - 22s 1s/step - loss: 1.4630 - accuracy: 0.5483 - val_loss: 1.4431 - val_accuracy: 0.6100 - lr: 0.0010\n",
            "Epoch 17/20\n",
            "17/17 [==============================] - 22s 1s/step - loss: 1.4201 - accuracy: 0.5626 - val_loss: 1.6488 - val_accuracy: 0.4500 - lr: 0.0010\n",
            "Epoch 18/20\n",
            "17/17 [==============================] - 22s 1s/step - loss: 1.4060 - accuracy: 0.5843 - val_loss: 1.5270 - val_accuracy: 0.5400 - lr: 0.0010\n",
            "Epoch 19/20\n",
            "17/17 [==============================] - ETA: 0s - loss: 1.3103 - accuracy: 0.5799\n",
            "Epoch 19: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.\n",
            "17/17 [==============================] - 22s 1s/step - loss: 1.3103 - accuracy: 0.5799 - val_loss: 1.4904 - val_accuracy: 0.5600 - lr: 0.0010\n",
            "Epoch 20/20\n",
            "17/17 [==============================] - 23s 1s/step - loss: 1.1810 - accuracy: 0.6357 - val_loss: 1.2313 - val_accuracy: 0.6000 - lr: 5.0000e-04\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Step 4 : Showing the Valid set Accuracy**"
      ],
      "metadata": {
        "id": "s1f9SVr6FTkU"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Calculating accuracy\n",
        "scores = model.evaluate(\n",
        "            valid_generator, steps=valid_generator.samples // batch_size)\n",
        "validation_accuracy = scores[1] * 100\n",
        "\n",
        "print(\"\\nValid set Accuracy: %.2f%%\\n\" % validation_accuracy)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cyr2NYMdFUbh",
        "outputId": "b5719fb3-8a2f-444f-ce47-41bd2a947598"
      },
      "execution_count": 312,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 0s 464ms/step - loss: 1.3943 - accuracy: 0.5800\n",
            "\n",
            "Valid set Accuracy: 58.00%\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def show_train_history(train_history, train, validation):\n",
        "  train_acc = train_history.history['accuracy']\n",
        "  val_acc = train_history.history['val_accuracy']\n",
        "\n",
        "  plt.plot(range(1, len(train_acc) + 1), train_acc, label='Training Accuracy')\n",
        "  plt.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy')\n",
        "  plt.title('Training and Validation Accuracy')\n",
        "  plt.xlabel('Epochs')\n",
        "  plt.ylabel('Accuracy')\n",
        "  plt.legend()\n",
        "  plt.show()"
      ],
      "metadata": {
        "id": "rMUMyeJhFEZN"
      },
      "execution_count": 313,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "show_train_history(train_history, 'accuracy', 'val_accuracy')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "RH3_0U13Fa5h",
        "outputId": "664c322d-1034-4b26-8e3f-b2062e4548a2"
      },
      "execution_count": 314,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjcAAAHHCAYAAABDUnkqAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAACcOElEQVR4nOzdd3gUVRfA4d+m9wKptAChQwokgIBUQRAB6b0j2FAU8UMsNAsWVFQUFOlIEaQpvSpNCCWh914SCJDed+f7Y8hCSELabjYJ532ePDuZnbn37CawJ7dqFEVREEIIIYQoIcxMHYAQQgghhCFJciOEEEKIEkWSGyGEEEKUKJLcCCGEEKJEkeRGCCGEECWKJDdCCCGEKFEkuRFCCCFEiSLJjRBCCCFKFEluhBBCCFGiSHIjxBMMHjyYihUr5uveiRMnotFoDBtQEXP58mU0Gg3z5s0r9Lo1Gg0TJ07Ufz9v3jw0Gg2XL1/O8d6KFSsyePBgg8ZTkN8VIYRhSXIjiiWNRpOrr507d5o61KfeW2+9hUaj4fz589le8+GHH6LRaDh69GghRpZ3N2/eZOLEiYSGhpo6lCydOnUKjUaDjY0NUVFRpg5HCJOR5EYUSwsXLszw1aZNmyzP16xZs0D1zJo1izNnzuTr3o8++ojExMQC1V8S9OvXD4DFixdne82SJUvw8/PD398/3/UMGDCAxMREfHx88l1GTm7evMmkSZOyTG4K8rtiKIsWLcLLywuAFStWmDQWIUzJwtQBCJEf/fv3z/D9f//9x5YtWzKdf1xCQgJ2dna5rsfS0jJf8QFYWFhgYSH/xBo2bEiVKlVYsmQJ48ePz/T8vn37uHTpEl988UWB6jE3N8fc3LxAZRREQX5XDEFRFBYvXkzfvn25dOkSv//+Oy+//LJJY8pOfHw89vb2pg5DlGDSciNKrBYtWlCnTh0OHTpEs2bNsLOz44MPPgBgzZo1vPjii5QpUwZra2t8fX355JNP0Gq1Gcp4fBxF+hiTqVOn8uuvv+Lr64u1tTX169cnJCQkw71ZjbnRaDSMHDmS1atXU6dOHaytralduzYbN27MFP/OnTsJDg7GxsYGX19ffvnll1yP49m1axc9evSgQoUKWFtbU758ed55551MLUmDBw/GwcGBGzdu0LlzZxwcHHB3d2fMmDGZ3ouoqCgGDx6Ms7MzLi4uDBo0KNddH/369eP06dMcPnw403OLFy9Go9HQp08fUlJSGD9+PEFBQTg7O2Nvb0/Tpk3ZsWNHjnVkNeZGURQ+/fRTypUrh52dHS1btuTEiROZ7r137x5jxozBz88PBwcHnJyceOGFFwgLC9Nfs3PnTurXrw/AkCFD9F2f6eONshpzEx8fz7vvvkv58uWxtramevXqTJ06FUVRMlyXl9+L7OzZs4fLly/Tu3dvevfuzb///sv169czXafT6fj+++/x8/PDxsYGd3d32rVrx8GDBzNct2jRIho0aICdnR2urq40a9aMzZs3Z4j50TFP6R4fz5T+c/nnn394/fXX8fDwoFy5cgBcuXKF119/nerVq2Nra0vp0qXp0aNHluOmoqKieOedd6hYsSLW1taUK1eOgQMHEhkZSVxcHPb29owaNSrTfdevX8fc3JwpU6bk8p0UJYH8WSlKtLt37/LCCy/Qu3dv+vfvj6enJ6D+h+vg4MDo0aNxcHBg+/btjB8/npiYGL7++uscy128eDGxsbG88soraDQavvrqK7p27crFixdz/At+9+7drFy5ktdffx1HR0d++OEHunXrxtWrVyldujQAR44coV27dnh7ezNp0iS0Wi2TJ0/G3d09V697+fLlJCQk8Nprr1G6dGkOHDjAjz/+yPXr11m+fHmGa7VaLW3btqVhw4ZMnTqVrVu38s033+Dr68trr70GqEnCSy+9xO7du3n11VepWbMmq1atYtCgQbmKp1+/fkyaNInFixdTr169DHX/8ccfNG3alAoVKhAZGclvv/1Gnz59GD58OLGxscyePZu2bdty4MABAgMDc1VfuvHjx/Ppp5/Svn172rdvz+HDh3n++edJSUnJcN3FixdZvXo1PXr0oFKlSkRERPDLL7/QvHlzTp48SZkyZahZsyaTJ09m/PjxjBgxgqZNmwLQuHHjLOtWFIVOnTqxY8cOhg0bRmBgIJs2beK9997jxo0bfPfddxmuz83vxZP8/vvv+Pr6Ur9+ferUqYOdnR1Llizhvffey3DdsGHDmDdvHi+88AIvv/wyaWlp7Nq1i//++4/g4GAAJk2axMSJE2ncuDGTJ0/GysqK/fv3s337dp5//vlcv/+Pev3113F3d2f8+PHEx8cDEBISwt69e+nduzflypXj8uXLzJgxgxYtWnDy5El9K2tcXBxNmzbl1KlTDB06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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Step 5 : Using the testing_set to test my model**"
      ],
      "metadata": {
        "id": "KO5_ymTLeR2g"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "* **改用`resize()`解決測試集影像大小與訓練集影像大小不一的問題，以提高 Test.xlsx 類別的平均程度**"
      ],
      "metadata": {
        "id": "5RzN1BbYeSn8"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def resize_image(img, target_size=(224, 224)):\n",
        "    '''\n",
        "    Resize image using bilinear interpolation method\n",
        "    '''\n",
        "    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)\n",
        "\n",
        "    return resized_img"
      ],
      "metadata": {
        "id": "6KWjLmV8bvdt"
      },
      "execution_count": 315,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import cv2\n",
        "from PIL import Image\n"
      ],
      "metadata": {
        "id": "RumCX0ngkTBJ"
      },
      "execution_count": 316,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def get_predicted_breed(predictions):\n",
        "  '''\n",
        "  To get the dog breed name based on the model predictions,\n",
        "  assume that predictions is the result of the model's prediction of the image,\n",
        "  determine the predicted breed based on the index of the highest probability in the probability vector\n",
        "  '''\n",
        "\n",
        "  class_names = [\"Airedale\", \"Beagle\", \"Bloodhound\", \"Bluetick\", \"Chihuahua\", \"Collie\", \"Dingo\",\n",
        "                            \"French Bulldog\", \"German Sheperd\", \"Malinois\", \"Newfoundland\", \"Pekinese\",\n",
        "                            \"Pomeranian\", \"Pug\", \"Vizsla\"]\n",
        "\n",
        "  breed_index = predictions.argmax()\n",
        "  breed_name = class_names[breed_index]\n",
        "\n",
        "  return breed_name"
      ],
      "metadata": {
        "id": "H4pgwTMZb1Re"
      },
      "execution_count": 317,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#def test(test_dir):\n",
        "\n",
        "test_dir = '/content/drive/MyDrive/archive/testing_set'\n",
        "\n",
        "# Getting test set file address\n",
        "test_files = os.listdir(test_dir)\n",
        "# To be fair, the order of access is randomized.\n",
        "random.shuffle(test_files)\n",
        "\n",
        "test_results = []\n",
        "\n",
        "for file_name in test_files:\n",
        "\n",
        "  # Load image with center crop and preprocessing\n",
        "  img_path = os.path.join(test_dir, file_name)\n",
        "  img = Image.open(img_path)\n",
        "  img_array = img_to_array(img)\n",
        "\n",
        "  # resize\n",
        "  img_array = resize_image(img_array, target_size=(224, 224))\n",
        "  img_array = preprocess_input(img_array)\n",
        "  img_array = img_array.reshape((1,) + img_array.shape)\n",
        "\n",
        "  # Making predictions about the image\n",
        "  predictions = model.predict(img_array)\n",
        "  predicted_breed = get_predicted_breed(predictions)\n",
        "\n",
        "  test_results.append((file_name, predicted_breed))\n",
        "\n",
        "  # Output results into Excel(no need title)\n",
        "  df = pd.DataFrame(test_results, columns=[\n",
        "                        'File Name', 'Predicted Breed'])\n",
        "  df.to_excel('test_data.xlsx', index=False, header=False)\n",
        "\n",
        "print(\"Test results saved to test_data.xlsx\")"
      ],
      "metadata": {
        "id": "WQy9Mqhdb6bf",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "2ee97043-6e8d-4209-d07b-214171ae4fbe"
      },
      "execution_count": 318,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 0s 94ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 23ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 24ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 18ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 18ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 18ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 18ms/step\n",
            "1/1 [==============================] - 0s 18ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 18ms/step\n",
            "1/1 [==============================] - 0s 17ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 18ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 18ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 18ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 19ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "1/1 [==============================] - 0s 20ms/step\n",
            "Test results saved to test_data.xlsx\n"
          ]
        }
      ]
    }
  ],
  "metadata": {
    "language_info": {
      "name": "python"
    },
    "colab": {
      "provenance": [],
      "machine_shape": "hm",
      "gpuType": "V100",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "nbformat": 4,
  "nbformat_minor": 0
}