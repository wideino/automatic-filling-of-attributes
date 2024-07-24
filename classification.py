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

# Создание генераторов данных для обучения и валидации
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2  # Используем часть данных для валидации
)

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
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Компиляция модели
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator
)


# Функция для классификации и перемещения изображений
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
new_images_dir = 'path_to_new_images'
output_dir = 'path_to_output_dir'

# if not os.path.exists(new_images_dir):
#     os.mkdir(new_images_dir)
#
# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)

classify_images(model, new_images_dir, output_dir)
