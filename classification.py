import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Путь к данным для обучения и валидации
base_dir = 'wallpaper-pictures'


'''Преобразует изображения для использования в модели. 
В данном случае изображения масштабируются (rescale) с диапазона [0, 255] до [0, 1].'''
# Создание генераторов данных для обучения и валидации
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2  # Используем часть данных для валидации
)

'''Загружает изображения из папок и автоматически создает классы на основе имен папок. 
Использует параметр validation_split, чтобы разделить данные на тренировочные и валидационные.'''
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Создание модели
'''Используем предобученную модель VGG16 без верхнего уровня (include_top=False).
Верхний уровень (fully connected слои) будет создан заново.'''
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
'''Flatten: Преобразует выходные данные сверточных слоев в одномерный массив.'''
x = Flatten()(base_model.output)
'''Dense: Полносвязные слои с активацией ReLU и softmax.
Первый полносвязный слой имеет 1024 нейрона, второй слой имеет столько нейронов, сколько у нас классов (train_generator.num_classes).'''
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Компиляция модели
'''compile: Настраивает модель для обучения. 
Используем оптимизатор Adam, функцию потерь categorical_crossentropy и метрику accuracy.'''
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
'''fit: Обучает модель на тренировочных данных и проверяет точность на валидационных данных. 
Устанавливаем количество эпох (epochs=50).'''
model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator
)


# Функция для классификации и перемещения изображений
'''classify_images: Классифицирует новые изображения и перемещает их в соответствующие папки.
os.path.exists, os.makedirs: Проверяет существование папки и создает её, если не существует.
os.walk: Проходит по всем файлам в указанной директории.
load_img, img_to_array: Загружает и преобразует изображение в массив.
np.expand_dims: Добавляет измерение для соответствия формату модели.
model.predict: Предсказывает класс изображения.
np.argmax: Определяет индекс класса с наибольшей вероятностью.
shutil.move: Перемещает изображение в соответствующую папку.'''
def classify_images(model, img_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(img_dir):
        for file in files:
            img_path = os.path.join(root, file)
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.

            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            class_label = list(train_generator.class_indices.keys())[predicted_class]

            class_dir = os.path.join(output_dir, class_label)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            shutil.move(img_path, os.path.join(class_dir, file))


# Применение классификации к новым данным
'''path_to_new_images: Путь к папке с новыми изображениями для классификации.
path_to_output_dir: Путь к папке для сохранения классифицированных изображений.
classify_images: Вызывает функцию для классификации новых изображений и их перемещения в папки.'''
new_images_dir = 'path_to_new_images'
output_dir = 'path_to_output_dir'

classify_images(model, new_images_dir, output_dir)
