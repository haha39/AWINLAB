import os

from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input


class DogClassifier:
    def __init__(self):
        self.model = self.build_model()

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

    def train(self, train_dir, valid_dir, batch_size, epochs):
        # 使用 ImageDataGenerator 对训练和验证数据进行预处理和增强
        train_datagen = ImageDataGenerator(rescale=1./255)
        valid_datagen = ImageDataGenerator(rescale=1./255)

        # 使用 flow_from_directory 方法生成训练和验证集的数据流
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

        # 使用 fit_generator 方法训练模型
        self.model.fit(
            train_generator,
            steps_per_epoch=min(train_generator.samples //
                                batch_size, 200),  # 设置最大步数为200
            epochs=epochs,
            validation_data=valid_generator,
            validation_steps=valid_generator.samples // batch_size
        )

    def evaluate(self, valid_dir, batch_size):
        valid_datagen = ImageDataGenerator(rescale=1./255)

        valid_generator = valid_datagen.flow_from_directory(
            valid_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical'
        )

        scores = self.model.evaluate(
            valid_generator, steps=valid_generator.samples // batch_size)
        print("Valid set Accuracy: %.2f%%" % (scores[1] * 100))


def main():
    classifier = DogClassifier()
    train_dir = 'archive/train'  # 训练集路径
    valid_dir = 'archive/valid'  # 验证集路径
    batch_size = 32
    epochs = 10

    classifier.train(train_dir, valid_dir, batch_size, epochs)  # 训练模型
    classifier.evaluate(valid_dir, batch_size)  # 评估模型


if __name__ == "__main__":
    main()
