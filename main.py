import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from keras.applications import InceptionV3
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator

image_size = (224, 224)
batch_size = 32
num_epochs = 10
data_dir = "D:\\Peleng\\data"

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation'
)

base_model = InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='linear')
])

model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='mean_squared_error', metrics=['mae'])

history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator
)

predictions = model.predict(val_generator)

actual_ages = []
for subdir in os.listdir(data_dir):
    subdir_path = os.path.join(data_dir, subdir)
    if os.path.isdir(subdir_path) and subdir.startswith("age_group_"):
        age_label = int(subdir.split('_')[2])
        actual_ages.extend([age_label] * len(os.listdir(subdir_path)))

num_samples = min(5, len(predictions))
for i in range(num_samples):
    actual_age = actual_ages[i]
    predicted_age = predictions[i][0]
    print(f"Реальный возраст: {actual_age}, предсказанный возраст: {predicted_age}")

test_loss, test_mae = model.evaluate(val_generator)

print(f"Средняя абсолютная ошибка на валидационном множестве: {test_mae}")

