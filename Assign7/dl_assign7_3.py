#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('pip', 'install tensorflow numpy matplotlib pandas')


# In[3]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0

print("TensorFlow version:", tf.__version__)
print("Available GPU:", tf.config.list_physical_devices("GPU"))


# In[4]:


dataset_path = tf.keras.utils.get_file(
    "flower_photos",
    origin="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
    untar=True
)

print("Downloaded dataset path:", dataset_path)
print("Folders:", os.listdir(dataset_path))


# In[5]:


if "flower_photos" in os.listdir(dataset_path):
    nested_path = os.path.join(dataset_path, "flower_photos")

    if os.path.isdir(nested_path):
        dataset_path = nested_path

print("Final dataset path:", dataset_path)
print("Contents:", os.listdir(dataset_path))


# In[6]:


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Class names:", class_names)
print("Number of classes:", num_classes)

assert num_classes == 5, "Dataset should contain five flower classes!"


# In[7]:


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# In[8]:


vgg_base = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

vgg_base.trainable = False

vgg_model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./255),
    vgg_base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

vgg_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("VGG16 output shape:", vgg_model.output_shape)

vgg_history = vgg_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=4
)


# In[9]:


vgg_loss, vgg_accuracy = vgg_model.evaluate(val_ds)

print("VGG16 Validation Loss:", vgg_loss)
print("VGG16 Validation Accuracy:", vgg_accuracy)


# In[10]:


resnet_base = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

resnet_base.trainable = False

resnet_model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./255),
    resnet_base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

resnet_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("ResNet50 output shape:", resnet_model.output_shape)

resnet_history = resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=4
)


# In[11]:


resnet_loss, resnet_accuracy = resnet_model.evaluate(val_ds)

print("ResNet50 Validation Loss:", resnet_loss)
print("ResNet50 Validation Accuracy:", resnet_accuracy)


# In[12]:


efficientnet_base = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

efficientnet_base.trainable = False

efficientnet_model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    efficientnet_base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

efficientnet_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("EfficientNetB0 output shape:", efficientnet_model.output_shape)

efficientnet_history = efficientnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=4
)


# In[13]:


efficientnet_loss, efficientnet_accuracy = efficientnet_model.evaluate(val_ds)

print("EfficientNetB0 Validation Loss:", efficientnet_loss)
print("EfficientNetB0 Validation Accuracy:", efficientnet_accuracy)


# In[14]:


import pandas as pd

results = pd.DataFrame({
    "Model": ["VGG16", "ResNet50", "EfficientNetB0"],
    "Validation Loss": [
        vgg_loss,
        resnet_loss,
        efficientnet_loss
    ],
    "Validation Accuracy": [
        vgg_accuracy,
        resnet_accuracy,
        efficientnet_accuracy
    ]
})

results["Validation Accuracy (%)"] = (
    results["Validation Accuracy"] * 100
).round(2)

print(results.to_string(index=False))


# In[15]:


model_names = ["VGG16", "ResNet50", "EfficientNetB0"]

accuracies = [
    vgg_accuracy,
    resnet_accuracy,
    efficientnet_accuracy
]

plt.figure(figsize=(8, 5))
plt.bar(model_names, accuracies)

plt.xlabel("Pre-trained Models")
plt.ylabel("Validation Accuracy")
plt.title("Transfer Learning Model Comparison")
plt.ylim(0, 1)
plt.show()


# In[16]:


plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(vgg_history.history["val_accuracy"]) + 1),
    vgg_history.history["val_accuracy"],
    marker="o",
    label="VGG16"
)

plt.plot(
    range(1, len(resnet_history.history["val_accuracy"]) + 1),
    resnet_history.history["val_accuracy"],
    marker="o",
    label="ResNet50"
)

plt.plot(
    range(1, len(efficientnet_history.history["val_accuracy"]) + 1),
    efficientnet_history.history["val_accuracy"],
    marker="o",
    label="EfficientNetB0"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy Across Epochs")
plt.legend()
plt.xticks([1, 2])
plt.show()


# In[ ]:




