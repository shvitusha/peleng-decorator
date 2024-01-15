import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow
from keras.applications import InceptionV3
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator

# Параметры
image_size = (224, 224)
batch_size = 32
num_epochs = 10

data_dir = "D:\\Peleng\\data"


# Создайте ImageDataGenerator для обучения
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Загрузите изображения с использованием flow_from_directory
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',  # Используйте 'sparse' для регрессии
    subset='training'
)

# Валидационный генератор (если указан параметр validation_split)
val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation'
)

# Создание и загрузка предварительно обученной модели InceptionV3
base_model = InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

# Замораживаем веса предварительно обученной части
base_model.trainable = False

# Добавляем свои слои поверх InceptionV3
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='linear')  # Линейная активация для регрессии
])

# Компиляция модели
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='mean_squared_error', metrics=['mae'])

# Обучение модели
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator
)

# Предсказание возраста на тестовом наборе данных
predictions = model.predict(val_generator)

# Преобразование меток возраста из каталогов в список
actual_ages = []
for subdir in os.listdir(data_dir):
    subdir_path = os.path.join(data_dir, subdir)
    if os.path.isdir(subdir_path) and subdir.startswith("age_group_"):
        # Извлекаем возраст из названия подкаталога
        age_label = int(subdir.split('_')[2])  # Предполагаем, что название подкаталога имеет вид "age_group_X"
        actual_ages.extend([age_label] * len(os.listdir(subdir_path)))

# Вывод фактического и предсказанного возраста для первых нескольких изображений
num_samples = min(5, len(predictions))  # Ограничиваем вывод для первых 5 изображений
for i in range(num_samples):
    actual_age = actual_ages[i]
    predicted_age = predictions[i][0]
    print(f"Actual Age: {actual_age}, Predicted Age: {predicted_age}")

# Оценка модели
test_loss, test_mae = model.evaluate(val_generator)

print(f"Mean Absolute Error on Validation Set: {test_mae}")

