import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from keras.models import Sequential
import numpy as np
from tensorflow.keras.preprocessing import image

import keras
import random

directory = 'dogs-vs-cats/train'

# Képek címe
images = os.listdir(directory)

# Random képek
num_random_images = 16
random_images = random.sample(images, num_random_images)

# rács mérete
rows = 4
cols = 4

#Kép méret
target_size = (150, 150)

# rács inicializálása
fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

# Képek megjelenítése 4x4 rácsban
for i in range(rows):
    for j in range(cols):
        img_path = os.path.join(directory, random_images[i * cols + j])
        img = cv2.imread(img_path)
        resized_image = cv2.resize(img, target_size)
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        axes[i, j].imshow(image_rgb)
        axes[i, j].axis('off')

plt.tight_layout()
plt.show()

print(len(images))

dogs = []
cats = []

for i in images:
    category = i.split('.')[0]
    if category == 'dog':
        dogs.append(i)
    else:
        cats.append(i)
print("kutyák" , len(dogs))
print("macskák" , len(cats))

# Fájlnevek alapján kategóriák hozzárendelése
classes = []
for i in images:
    category = i.split('.')[0]
    if category == 'dog':
        classes.append(1)
    else:
        classes.append(0)

# DataFrame létrehozása
df = pd.DataFrame({
    'id': images,
    'class': classes
})

# DataFrame megjelenítése
print(df.head())      

df['class'] = df['class'].astype(str)

train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

train_data = train_df[:10000]
valid_data = valid_df[:2000]
datagen = ImageDataGenerator(rescale=1./255)

# Tanító adatgenerátor létrehozása
train_generator = datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=directory,
    x_col="id",
    y_col="class",
    target_size=(32, 32),
    batch_size= 32,
    class_mode='binary'
)

# Validációs adatgenerátor létrehozása
validation_generator = datagen.flow_from_dataframe(
    dataframe=valid_data,
    directory=directory,
    x_col="id",
    y_col="class",
    target_size=(32, 32),
    batch_size=32,
    class_mode='binary'
)

model = Sequential()

# Konvolúciós réteg és max pooling réteg 
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))

# Teljesen összekapcsolt rétegek hozzáadása
# Bináris klasszifikációs probléma ezért 1 kimeneti neuron és aktivációs függvény sigmoid.
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# CNN konfiguráció.
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.summary()

# tanítás
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)
model.save('test.keras')


#model = keras.models.load_model('test.keras')


folder_path = 'dogs-vs-cats/test1/'

test_images = os.listdir(folder_path)
len(test_images)

# Tanítási görbe plottolás.
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

def plot(train_accuracy,val_accuracy):
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    
    # Tanulási és validációs accuracy ábrázolása
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

plot(train_accuracy,val_accuracy)

# Modell sikerességének tesztelése a teszt adathalmazon.
def predictic_image(image_path):

    # Kép beolvasása
    plot = cv2.imread(img_path)
    
    plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
    
    # Kép megjelenítése
    plt.imshow(plot)
    plt.axis('off')
    plt.show()

    # Megfelelő formátumra konvertálás
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalizálás
    
    # Predikció
    predictions = model.predict(img_array)
    if predictions[0] < 0.5:
        print("A képen egy macska van")
    else:
        print("A képen egy kutya van")

img_path = 'dogs-vs-cats/test1/22.jpg' 
predictic_image(img_path)

