import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
import os

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

# Список классов
classes = sorted('banana', 'apple', 'pear', 'grapes', 'orange', 'kiwi', 'watermelon',
           'pomegranate', 'pineapple', 'mango', 'cucumber', 'carrot',
           'capsicum', 'onion', 'potato', 'lemon', 'tomato', 'raddish',
           'beetroot', 'cabbage', 'lettuce', 'spinach', 'soy beans',
           'cauliflower', 'bell pepper', 'chilli pepper', 'turnip', 'corn',
           'sweetcorn', 'sweetpotato', 'paprika', 'jalepeno', 'ginger',
           'garlic', 'peas', 'eggplant')

# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка ResNet и настройка модели
resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, len(classes))
resnet = resnet.to(device)

# Загрузка сохраненной модели
if os.path.exists(model_path):
    resnet.load_state_dict(torch.load(model_path))
    print(f"Модель загружена из {model_path}")

# Оптимизатор и функция потерь
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

# Обучение
start_epoch = 13  # Начинаем с 13 эпохи
num_epochs = 15   # Дообучение до 15 эпох
train_losses = []

for epoch in range(start_epoch, num_epochs):
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
    print(f"Эпоха [{epoch + 1}/{num_epochs}] завершена. Средняя ошибка: {running_loss / len(trainloader):.4f}")

# Сохранение модели
torch.save(resnet.state_dict(), model_path)
print(f"Модель дообучена и сохранена в {model_path}")

# График потерь
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Потери')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.title('График потерь')
plt.legend()
plt.show()
