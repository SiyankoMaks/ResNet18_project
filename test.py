import torch
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Устройство (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Определение архитектуры модели (ResNet18 с 36 выходами)
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 36)  # 36 классов
model = model.to(device)

# Загрузка сохранённых весов
model.load_state_dict(torch.load('modelF.pth', map_location=device))
model.eval()

# Список классов
classes = sorted(('banana', 'apple', 'pear', 'grapes', 'orange', 'kiwi', 'watermelon',
           'pomegranate', 'pineapple', 'mango', 'cucumber', 'carrot',
           'capsicum', 'onion', 'potato', 'lemon', 'tomato', 'raddish',
           'beetroot', 'cabbage', 'lettuce', 'spinach', 'soy beans',
           'cauliflower', 'bell pepper', 'chilli pepper', 'turnip', 'corn',
           'sweetcorn', 'sweetpotato', 'paprika', 'jalepeno', 'ginger',
           'garlic', 'peas', 'eggplant'))

# Преобразования изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Размер, используемый при обучении
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dir = 'D:/NeuralNetwork/data/test'

# Загрузка тестового набора
test_images = []
test_labels = []
for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(test_dir, class_name)
    if not os.path.exists(class_dir):
        continue
    for file_name in os.listdir(class_dir):
        if file_name.endswith(('png', 'jpg', 'jpeg')):
            test_images.append(os.path.join(class_dir, file_name))
            test_labels.append(class_idx)

# Инициализация метрик
class_correct = [0] * len(classes)
class_total = [0] * len(classes)
total_loss = 0.0
criterion = torch.nn.CrossEntropyLoss()

# Оценка модели
model.eval()
with torch.no_grad():
    for img_path, true_label in zip(test_images, test_labels):
        img = Image.open(img_path).convert('RGB')
        input_img = transform(img).unsqueeze(0).to(device)  # Добавляем batch dimension
        true_label_tensor = torch.tensor([true_label]).to(device)
        
        # Предсказание
        outputs = model(input_img)
        loss = criterion(outputs, true_label_tensor)
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        predicted_label = predicted.item()
        
        # Обновление метрик
        class_total[true_label] += 1
        if predicted_label == true_label:
            class_correct[true_label] += 1

# Вывод средней ошибки
avg_loss = total_loss / len(test_images)
print(f"Средняя ошибка: {avg_loss:.4f}")

# Вывод общей точности
total_correct = sum(class_correct)
total_samples = sum(class_total)
overall_accuracy = 100.0 * total_correct / total_samples
print(f"Общая точность: {overall_accuracy:.2f}%")

# Вывод точности по каждому классу
for i in range(len(classes)):
    if class_total[i] > 0:
        class_accuracy = 100.0 * class_correct[i] / class_total[i]
        print(f"Точность для класса '{classes[i]}': {class_accuracy:.2f}%")
    else:
        print(f"Класс '{classes[i]}' не представлен в тестовом наборе.")

# Визуализация изображений с предсказаниями
selected_images = np.random.choice(test_images, 10, replace=False)
fig, axes = plt.subplots(2, 10, figsize=(20, 5))
for i, img_path in enumerate(selected_images):
    original_class = os.path.basename(os.path.dirname(img_path))
    
    # Загрузка изображения
    img = Image.open(img_path).convert('RGB')
    input_img = transform(img).unsqueeze(0).to(device)
    
    # Предсказание
    with torch.no_grad():
        outputs = model(input_img)
        _, predicted = torch.max(outputs, 1)
        predicted_class = classes[predicted.item()]
    
    # Отображение
    axes[0, i].imshow(img)
    axes[0, i].set_title(f"True: {original_class}")
    axes[0, i].axis('off')

    axes[1, i].imshow(img)
    axes[1, i].set_title(f"Pred: {predicted_class}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()
