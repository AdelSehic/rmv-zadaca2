#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Postavljanje random seed-a za reproducibilnost
tf.random.set_seed(42)
np.random.seed(42)

print("TensorFlow verzija:", tf.__version__)
print("Keras verzija:", keras.__version__)

# 1. UČITAVANJE I PRIPREMA PODATAKA
print("\n=== UČITAVANJE CIFAR-10 DATASETA ===")
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Nazivi klasa
class_names = ['Avion', 'Automobil', 'Ptica', 'Mačka', 'Jelen', 
               'Pas', 'Žaba', 'Konj', 'Brod', 'Kamion']

print(f"Trening set: {x_train.shape} slike, {y_train.shape} labeli")
print(f"Test set: {x_test.shape} slike, {y_test.shape} labeli")
print(f"Broj klasa: {len(class_names)}")

# Normalizacija piksela na [0,1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Konvertovanje labela u kategorijske (one-hot encoding)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(f"Normalizovane slike - min: {x_train.min():.2f}, max: {x_train.max():.2f}")

# 2. KREIRANJE KONVOLUCIONE NEURONSKE MREŽE
print("\n=== KREIRANJE CNN MODELA ===")

model = keras.Sequential([
    # Prvi konvolucioni blok
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Drugi konvolucioni blok
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Treći konvolucioni blok
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Flatovanje i potpuno povezani slojevi
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Kompajliranje modela
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Prikaz strukture modela
print("\n=== STRUKTURA MODELA ===")
model.summary()

# 3. DEFINISANJE CALLBACK-a
print("\n=== DEFINISANJE CALLBACK-a ===")

# Early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Learning rate scheduler
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.0001
)

# Model checkpoint
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_cifar10_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# 4. TRENIRANJE MODELA
print("\n=== TRENIRANJE MODELA ===")

# Data augmentation za poboljšanje generalizacije
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

datagen.fit(x_train)

# Treniranje
epochs = 50
batch_size = 32

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) // batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# 5. EVALUACIJA MODELA
print("\n=== EVALUACIJA MODELA ===")

# Finalna evaluacija
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Finalna test tačnost: {test_accuracy:.4f}")
print(f"Finalni test loss: {test_loss:.4f}")

# Predikcije
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Classification report
print("\n=== DETALJNE METRIKE ===")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# 6. VIZUALIZACIJA REZULTATA
print("\n=== KREIRANJE VIZUALIZACIJA ===")

# Kreiranje subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Tačnost kroz epohe
axes[0, 0].plot(history.history['accuracy'], label='Trening tačnost')
axes[0, 0].plot(history.history['val_accuracy'], label='Validaciona tačnost')
axes[0, 0].set_title('Tačnost modela kroz epohe')
axes[0, 0].set_xlabel('Epoha')
axes[0, 0].set_ylabel('Tačnost')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. Loss kroz epohe
axes[0, 1].plot(history.history['loss'], label='Trening loss')
axes[0, 1].plot(history.history['val_loss'], label='Validacioni loss')
axes[0, 1].set_title('Loss modela kroz epohe')
axes[0, 1].set_xlabel('Epoha')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# 3. Confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix')
axes[1, 0].set_xlabel('Predviđena klasa')
axes[1, 0].set_ylabel('Stvarna klasa')

# 4. Tačnost po klasama
class_accuracy = cm.diagonal() / cm.sum(axis=1)
axes[1, 1].bar(class_names, class_accuracy)
axes[1, 1].set_title('Tačnost po klasama')
axes[1, 1].set_xlabel('Klasa')
axes[1, 1].set_ylabel('Tačnost')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('cifar10_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. PRIKAZ NEKOLIKO PRIMERA PREDIKCIJA
print("\n=== PRIMERI PREDIKCIJA ===")

# Uzimanje random uzoraka
np.random.seed(42)
sample_indices = np.random.choice(len(x_test), 12, replace=False)

plt.figure(figsize=(15, 8))
for i, idx in enumerate(sample_indices):
    plt.subplot(3, 4, i + 1)
    plt.imshow(x_test[idx])
    
    true_label = class_names[y_true_classes[idx]]
    pred_label = class_names[y_pred_classes[idx]]
    confidence = np.max(y_pred[idx])
    
    color = 'green' if y_true_classes[idx] == y_pred_classes[idx] else 'red'
    plt.title(f'Tačno: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}', 
              color=color, fontsize=9)
    plt.axis('off')

plt.tight_layout()
plt.savefig('cifar10_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== ANALIZA ZAVRŠENA ===")
print(f"Model je postigao {test_accuracy:.2%} tačnost na test skupu")
print("Slike sa rezultatima su sačuvane kao 'cifar10_results.png' i 'cifar10_predictions.png'")

# Čuvanje finalnog modela
model.save('final_cifar10_model.h5')
print("Finalni model je sačuvan kao 'final_cifar10_model.h5'")
