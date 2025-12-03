import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np
import random

# Пути
train_dir = 'D:/NeuralNetwork/data/train'
test_dir = 'D:/NeuralNetwork/data/test'
model_path = './resnet_model.pth'

# Преобразования данных
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),  # Конвертируем в RGB
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Загрузка данных
trainset = datasets.ImageFolder(root=train_dir, transform=transform)
testset = datasets.ImageFolder(root=test_dir, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

classes = sorted(('banana', 'apple', 'pear', 'grapes', 'orange', 'kiwi', 'watermelon',
           'pomegranate', 'pineapple', 'mango', 'cucumber', 'carrot',
           'capsicum', 'onion', 'potato', 'lemon', 'tomato', 'raddish',
           'beetroot', 'cabbage', 'lettuce', 'spinach', 'soy beans',
           'cauliflower', 'bell pepper', 'chilli pepper', 'turnip', 'corn',
           'sweetcorn', 'sweetpotato', 'paprika', 'jalepeno', 'ginger',
           'garlic', 'peas', 'eggplant'))

# Загрузка ResNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# Замена последнего слоя для вашего количества классов
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, len(classes))
resnet = resnet.to(device)

# Оптимизатор и функция потерь
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

# Обучение
num_epochs = 10
train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    resnet.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 1):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Вывод текущей эпохи, итерации и ошибки
        if i % 10 == 0 or i == len(trainloader):
            print(f"Эпоха [{epoch + 1}/{num_epochs}], Итерация [{i}/{len(trainloader)}], Потери: {loss.item():.4f}")

    train_losses.append(running_loss / len(trainloader))

    # Тестирование
    resnet.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = resnet(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f"Эпоха [{epoch + 1}/{num_epochs}] завершена. Средняя ошибка: {running_loss / len(trainloader):.4f}, Точность: {accuracy:.2f}%")

# Сохранение модели
torch.save(resnet.state_dict(), model_path)
print(f"ResNet-модель сохранена в {model_path}")

# Предсказания для случайных изображений
def imshow(img, title):
    img = img / 2 + 0.5  # Разнормализация
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')

resnet.eval()
dataiter = iter(testloader)
images, labels = next(dataiter)

# Выбор 10 случайных изображений
indices = random.sample(range(len(images)), 10)
for idx in indices:
    image = images[idx]
    label = labels[idx].item()
    with torch.no_grad():
        output = resnet(image.unsqueeze(0).to(device))
        _, predicted = torch.max(output, 1)
    predicted_label = predicted.item()

    # Отображение изображений
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    imshow(image, title=f"Исходный: {classes[label]}")
    plt.subplot(1, 2, 2)
    imshow(image, title=f"Предсказание: {classes[predicted_label]}")
    plt.show()

# Вычисление точности по каждому классу
class_correct = [0] * len(classes)
class_total = [0] * len(classes)

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnet(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(len(classes)):
    print(f"Точность для {classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%")

# График потерь и точности
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Потери')
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Точность')
plt.xlabel('Эпоха')
plt.ylabel('Значение')
plt.legend()
plt.title('График потерь и точности')
plt.show()
