# # TensorFlow and tf.keras
# import tensorflow as tf
# from tensorflow import keras

# # Helper libraries
# import numpy as np
# import matplotlib.pyplot as plt

import os

# print(tf.__version__)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

# 创建一个序贯模型
model = Sequential()

# 添加一个独立的输入层
model.add(Input(shape=(224, 224, 3)))

# 添加第一个卷积层和最大池化层
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# 添加第二个卷积层和最大池化层
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# 添加第三个卷积层和最大池化层
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# 将三维特征图展平为一维向量
model.add(Flatten())

# 添加全连接层
model.add(Dense(512, activation='relu'))

# 添加 Dropout 层，防止过拟合
model.add(Dropout(0.5))

# 输出层，输出分类结果
model.add(Dense(15, activation='softmax'))  # 假设有15个类别

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# 打印模型结构
model.summary()


# 指定数据集文件夹路径
dataset_folder = "./archive"  # 将 "path_to_dataset_folder" 替换为实际的文件夹路径

# 分别访问 train、valid 和 test 文件夹
train_folder = os.path.join(dataset_folder, "train")
valid_folder = os.path.join(dataset_folder, "valid")
test_folder = os.path.join(dataset_folder, "test")

# if os.path.exists(valid_folder):
#     # 遍历验证集文件夹中的子文件夹（每个子文件夹代表一种狗的品种）
#     for breed_folder in os.listdir(valid_folder):
#         breed_path = os.path.join(valid_folder, breed_folder)
#         print(f"訪問驗證集 {breed_folder} 文件夹，路徑為: {breed_path}")
