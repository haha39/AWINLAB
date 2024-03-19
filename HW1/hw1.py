import os
import random
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import matplotlib.pyplot as plt


class DogClassifier:
    def __init__(self):
        self.model = self.build_model()
        # 添加 class_names 属性
        self.class_names = ["Airedale", "Beagle", "Bloodhound", "Bluetick", "Chihuahua", "Collie", "Dingo",
                            "French Bulldog", "German Sheperd", "Malinois", "Newfoundland", "Pekinese",
                            "Pomeranian", "Pug", "Vizsla"]

    def build_model(self):
        # 创建一个卷积神经网络模型
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
        # model.summary()
        return model

    def train(self, train_dir, batch_size, epochs):
        # 使用 ImageDataGenerator 对训练和验证数据进行预处理和增强
        train_datagen = ImageDataGenerator(rescale=1./255)
        # valid_datagen = ImageDataGenerator(rescale=1./255)

        # 使用 flow_from_directory 方法生成训练和验证集的数据流
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical'
        )

        # valid_generator = valid_datagen.flow_from_directory(
        #     valid_dir,
        #     target_size=(224, 224),
        #     batch_size=batch_size,
        #     class_mode='categorical'
        # )

        # 使用 fit 方法训练模型
        self.model.fit(
            train_generator,
            steps_per_epoch=min(train_generator.samples //
                                batch_size, 200),  # 设置最大步数为200
            epochs=epochs,
            # validation_data=valid_generator,
            # validation_steps=valid_generator.samples // batch_size
        )

    def evaluate(self, valid_dir, batch_size):
        valid_data_generator = ImageDataGenerator(
            preprocessing_function=preprocess_input)

        valid_generator = valid_data_generator.flow_from_directory(
            valid_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        evaluation = self.model.evaluate(
            valid_generator,
            steps=valid_generator.samples // batch_size,
            verbose=1
        )
        print("Validation Accuracy:", evaluation[1])

        # 获取模型在验证集上的预测结果
        predictions = self.model.predict(
            valid_generator, steps=valid_generator.samples // batch_size, verbose=1)

        # 绘制准确率和损失值曲线
        self.plot_metrics(self.model.history.history)

    def plot_metrics(self, history):
        # 绘制准确率曲线
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.show()

        # 绘制损失值曲线
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

    def test(self, test_dir):
        # 测试模型的方法
        # 获取测试集文件名列表
        test_files = os.listdir(test_dir)
        random.shuffle(test_files)

        # 初始化测试结果列表
        test_results = []

        for file_name in test_files:
            # 加载图像并进行预处理
            img_path = os.path.join(test_dir, file_name)
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            img_array = img_array.reshape((1,) + img_array.shape)

            # 对图像进行预测
            predictions = self.model.predict(img_array)
            predicted_breed = self.get_predicted_breed(predictions)

            # 将测试结果添加到列表中
            test_results.append((file_name, predicted_breed))

        # 将测试结果写入 Excel 文件
        df = pd.DataFrame(test_results, columns=[
                          'File Name', 'Predicted Breed'])
        df.to_excel('test_data.xlsx', index=False, header=False)  # 不写入标题行
        print("Test results saved to test_data.xlsx")

    def get_predicted_breed(self, predictions):
        # 根据模型预测结果获取狗狗品种名称
        # 这里假设 predictions 是模型对图像的预测结果，是一个概率向量
        # 根据概率向量中最大概率的索引来确定预测的品种
        breed_index = predictions.argmax()
        breed_name = self.class_names[breed_index]
        return breed_name


def main():
    classifier = DogClassifier()
    train_dir = 'archive/train'  # 训练集路径
    valid_dir = 'archive/valid'  # 验证集路径
    test_dir = 'archive/testing_set'  # 测试集路径
    batch_size = 32
    epochs = 10

    classifier.train(train_dir, batch_size, epochs)  # 训练模型
    classifier.evaluate(valid_dir, batch_size)  # 评估模型
    classifier.test(test_dir)  # 测试模型


if __name__ == "__main__":
    main()
