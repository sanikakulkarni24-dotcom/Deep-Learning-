#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install torch torchvision transformers matplotlib pandas')


# In[2]:


import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
import matplotlib.pyplot as plt
import pandas as pd
import time
import copy


# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Use a subset to keep the practical manageable
train_subset = torch.utils.data.Subset(
    train_dataset, range(5000)
)

test_subset = torch.utils.data.Subset(
    test_dataset, range(2000)
)

train_loader = DataLoader(
    train_subset, batch_size=32, shuffle=True
)

test_loader = DataLoader(
    test_subset, batch_size=32, shuffle=False
)

print("Training images:", len(train_subset))
print("Testing images:", len(test_subset))


# In[4]:


# Undo normalization for display
def show_image(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img = img.cpu() * std + mean
    return img.permute(1, 2, 0).clamp(0, 1)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    img, label = test_dataset[i]
    ax.imshow(show_image(img))
    ax.set_title(classes[label])
    ax.axis("off")

plt.suptitle("Sample Images from CIFAR-10")
plt.tight_layout()
plt.show()


# In[5]:


cnn = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Replace the final layer for 10 CIFAR-10 classes
cnn.fc = nn.Linear(cnn.fc.in_features, 10)
cnn = cnn.to(device)

print("Pre-trained ResNet18 CNN loaded.")


# In[6]:


vit = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=10,
    ignore_mismatched_sizes=True
)

vit = vit.to(device)

print("Pre-trained Vision Transformer loaded.")


# In[7]:


# Freeze CNN feature extractor
for param in cnn.parameters():
    param.requires_grad = False

for param in cnn.fc.parameters():
    param.requires_grad = True

# Freeze ViT feature extractor
for param in vit.parameters():
    param.requires_grad = False

for param in vit.classifier.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()

cnn_optimizer = torch.optim.Adam(
    cnn.fc.parameters(), lr=0.001
)

vit_optimizer = torch.optim.Adam(
    vit.classifier.parameters(), lr=0.001
)

def train_model(model, optimizer, is_vit, epochs=2):
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if is_vit:
                outputs = model(pixel_values=images).logits
            else:
                outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {total_loss / len(train_loader):.4f}"
        )

    return model

print("Training CNN classification head...")
cnn = train_model(cnn, cnn_optimizer, False)

print("\nTraining ViT classification head...")
vit = train_model(vit, vit_optimizer, True)


# In[8]:


def evaluate_model(model, is_vit):
    model.eval()
    correct = 0
    total = 0
    start = time.time()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_vit:
                outputs = model(pixel_values=images).logits
            else:
                outputs = model(images)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    elapsed = time.time() - start
    accuracy = 100 * correct / total

    return accuracy, elapsed

cnn_accuracy, cnn_time = evaluate_model(cnn, False)

print("CNN Accuracy:", round(cnn_accuracy, 2), "%")
print("CNN Evaluation Time:", round(cnn_time, 2), "seconds")


# In[9]:


vit_accuracy, vit_time = evaluate_model(vit, True)

print("ViT Accuracy:", round(vit_accuracy, 2), "%")
print("ViT Evaluation Time:", round(vit_time, 2), "seconds")


# In[10]:


model_names = ["CNN (ResNet18)", "Vision Transformer (ViT)"]
accuracies = [cnn_accuracy, vit_accuracy]

plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, accuracies)

plt.title("CNN vs ViT - Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

for bar, value in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.show()


# In[11]:


times = [cnn_time, vit_time]

plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, times)

plt.title("CNN vs ViT - Evaluation Time")
plt.ylabel("Time (seconds)")

for bar, value in zip(bars, times):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}s",
        ha="center"
    )

plt.tight_layout()
plt.show()


# In[12]:


cnn.eval()
vit.eval()

images, labels = next(iter(test_loader))
images = images.to(device)

with torch.no_grad():
    cnn_predictions = cnn(images).argmax(dim=1)
    vit_predictions = vit(pixel_values=images).logits.argmax(dim=1)

fig, axes = plt.subplots(2, 5, figsize=(15, 7))

for i in range(10):
    img = show_image(images[i])
    true_label = classes[labels[i].item()]
    cnn_label = classes[cnn_predictions[i].item()]
    vit_label = classes[vit_predictions[i].item()]

    ax = axes.flat[i]
    ax.imshow(img)
    ax.set_title(
        f"Actual: {true_label}\nCNN: {cnn_label}\nViT: {vit_label}",
        fontsize=8
    )
    ax.axis("off")

plt.suptitle("Image Classification Predictions: CNN vs ViT")
plt.tight_layout()
plt.show()


# In[ ]:




