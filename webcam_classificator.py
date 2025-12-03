import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms, models
import time
import os

class WebcamClassifier:
    def __init__(self, model_path='./resnet_model.pth', device=None):
        """
        Инициализация классификатора для работы с веб-камерой
        
        Args:
            model_path: путь к сохраненной модели
            device: устройство для вычислений (cuda/cpu)
        """
        self.classes = sorted(('banana', 'apple', 'pear', 'grapes', 'orange', 'kiwi', 'watermelon',
                              'pomegranate', 'pineapple', 'mango', 'cucumber', 'carrot',
                              'capsicum', 'onion', 'potato', 'lemon', 'tomato', 'raddish',
                              'beetroot', 'cabbage', 'lettuce', 'spinach', 'soy beans',
                              'cauliflower', 'bell pepper', 'chilli pepper', 'turnip', 'corn',
                              'sweetcorn', 'sweetpotato', 'paprika', 'jalepeno', 'ginger',
                              'garlic', 'peas', 'eggplant'))
        
        # Определение устройства
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Инициализация модели
        self.model = self.load_model(model_path)
        
        # Трансформации для изображения
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Статус работы классификатора
        self.running = False
        
        # Для выделения объекта
        self.last_prediction = None
        self.last_confidence = 0.0
        
    def load_model(self, model_path):
        """Загрузка предварительно обученной модели"""
        # Создание модели ResNet18
        model = models.resnet18()
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, len(self.classes))
        
        # Загрузка весов
        try:
            if self.device.type == 'cpu':
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            else:
                model.load_state_dict(torch.load(model_path))
            
            model = model.to(self.device)
            model.eval()
            print(f"Model successfully loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure the model has been trained and saved!")
            return None
    
    def preprocess_frame(self, frame):
        """Предобработка кадра для подачи в модель"""
        # Конвертация из BGR (OpenCV) в RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Конвертация в PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Применение трансформаций
        input_tensor = self.transform(pil_image)
        
        # Добавление batch dimension
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        return input_batch
    
    def predict(self, frame, roi=None):
        """Предсказание класса для кадра или ROI"""
        if self.model is None:
            return None, 0.0
        
        try:
            # Если задана область интереса (ROI), используем ее
            if roi is not None:
                x, y, w, h = roi
                # Проверка на корректность ROI
                if w > 10 and h > 10 and x >= 0 and y >= 0 and x + w <= frame.shape[1] and y + h <= frame.shape[0]:
                    roi_frame = frame[y:y+h, x:x+w]
                else:
                    roi_frame = frame
            else:
                roi_frame = frame
            
            # Предобработка кадра
            input_batch = self.preprocess_frame(roi_frame)
            
            # Предсказание
            with torch.no_grad():
                output = self.model(input_batch)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
                
                predicted_class = self.classes[predicted_idx.item()]
                confidence_score = confidence.item()
                
                # Сохраняем последнее предсказание для отображения при паузе
                self.last_prediction = predicted_class
                self.last_confidence = confidence_score
                
            return predicted_class, confidence_score
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0
    
    def detect_object(self, frame):
        """
        Простой детектор объектов на основе порогового значения
        Возвращает ROI (область интереса) для объекта
        """
        # Конвертация в HSV для лучшего выделения
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Создание маски для выделения объектов (настройте под свои нужды)
        # Эта маска ищет яркие/цветные объекты на темном фоне
        lower_bound = np.array([0, 30, 30])
        upper_bound = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Улучшение маски
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Поиск контуров
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Выбираем самый большой контур
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Увеличиваем ROI на 20% для лучшего охвата объекта
            padding = int(min(w, h) * 0.2)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            
            return (x, y, w, h)
        
        return None
    
    def run_webcam(self, camera_id=0, show_fps=True, confidence_threshold=0.5):
        """
        Запуск классификации в реальном времени с веб-камеры
        
        Args:
            camera_id: ID камеры (0 для встроенной, 1 для внешней)
            show_fps: показывать ли FPS
            confidence_threshold: порог уверенности для отображения
        """
        if self.model is None:
            print("Model not loaded! Cannot start classifier.")
            return
        
        # Открытие веб-камеры
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Cannot open camera with ID {camera_id}")
            return
        
        # Создание окна
        cv2.namedWindow('Fruit/Vegetable Classifier', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Fruit/Vegetable Classifier', 800, 600)
        
        print("\n" + "="*50)
        print("FRUIT/VEGETABLE CLASSIFIER")
        print("="*50)
        print("Instructions:")
        print("- Press 'Q' to quit")
        print("- Press 'S' to save current frame")
        print("- Press 'P' to pause/resume")
        print("- Press 'R' to reset object detection")
        print("- Press 'D' to toggle object detection")
        print("- Press 'I' to show info")
        print("="*50 + "\n")
        
        self.running = True
        paused = False
        frame_buffer = None
        show_detection = True  # Флаг для показа детекции объектов
        object_roi = None
        
        # Переменные для расчета FPS
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0
        
        # Создаем папку для сохранения скриншотов
        screenshot_dir = "classifier_screenshots"
        os.makedirs(screenshot_dir, exist_ok=True)
        
        # Инструкция для отображения в окне
        instructions = [
            "INSTRUCTIONS:",
            "Q - Quit",
            "S - Save screenshot",
            "P - Pause/Resume",
            "D - Toggle object detection",
            "R - Reset detection"
        ]
        
        while self.running:
            if not paused:
                # Захват кадра
                ret, frame = cap.read()
                
                if not ret:
                    print("Cannot capture frame!")
                    break
                
                # Сохраняем кадр в буфер (для паузы)
                frame_buffer = frame.copy()
                
                # Поиск объекта на кадре (если включено)
                if show_detection:
                    object_roi = self.detect_object(frame)
                else:
                    object_roi = None
                
                # Предсказание
                predicted_class, confidence = self.predict(frame, object_roi)
            else:
                # Используем буферизованный кадр при паузе
                if frame_buffer is not None:
                    frame = frame_buffer.copy()
                    predicted_class = self.last_prediction
                    confidence = self.last_confidence
                else:
                    continue
            
            # Копия кадра для отображения
            display_frame = frame.copy()
            
            # Расчет FPS (только при не-паузе)
            if not paused:
                fps_frame_count += 1
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_frame_count
                    fps_frame_count = 0
                    fps_start_time = time.time()
            
            # Рисуем ROI для объекта, если он обнаружен
            if object_roi and show_detection:
                x, y, w, h = object_roi
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Добавляем текст "Object detected"
                cv2.putText(display_frame, "Object detected", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Отображение результатов классификации
            if predicted_class and confidence > confidence_threshold:
                # Цвет рамки в зависимости от уверенности
                if confidence > 0.8:
                    border_color = (0, 255, 0)  # Зеленый
                    text_color = (0, 255, 0)
                elif confidence > 0.6:
                    border_color = (0, 255, 255)  # Желтый
                    text_color = (0, 255, 255)
                else:
                    border_color = (0, 165, 255)  # Оранжевый
                    text_color = (0, 165, 255)
                
                # Если есть ROI, рисуем рамку вокруг объекта
                if object_roi and show_detection:
                    x, y, w, h = object_roi
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), border_color, 3)
                
                # Фон для текста
                cv2.rectangle(display_frame, (10, 10), (350, 100), (0, 0, 0), -1)
                
                # Текст предсказания
                label_text = f"Class: {predicted_class}"
                confidence_text = f"Confidence: {confidence:.1%}"
                
                cv2.putText(display_frame, label_text, (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
                cv2.putText(display_frame, confidence_text, (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            else:
                # Отображение "Object not recognized"
                cv2.rectangle(display_frame, (10, 10), (300, 50), (0, 0, 0), -1)
                cv2.putText(display_frame, "Object not recognized", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Отображение FPS
            if show_fps:
                cv2.putText(display_frame, f"FPS: {fps}", (display_frame.shape[1] - 100, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Статус паузы и детекции
            status_y = display_frame.shape[0] - 60
            cv2.putText(display_frame, f"Status: {'PAUSED' if paused else 'ACTIVE'}", 
                       (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Detection: {'ON' if show_detection else 'OFF'}", 
                       (20, status_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 255, 0) if show_detection else (0, 0, 255), 2)
            
            # Отображение инструкций в правом верхнем углу
            instruction_y = 120
            for i, instruction in enumerate(instructions):
                cv2.putText(display_frame, instruction, 
                           (display_frame.shape[1] - 220, instruction_y + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Отображение кадра
            cv2.imshow('Fruit/Vegetable Classifier', display_frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q') or cv2.getWindowProperty('Fruit/Vegetable Classifier', cv2.WND_PROP_VISIBLE) < 1:
                self.running = False
                break
            elif key == ord('p') or key == ord('P'):  # Пауза/возобновление
                paused = not paused
                print(f"Status: {'PAUSED' if paused else 'RESUMED'}")
            elif key == ord('s') or key == ord('S'):  # Сохранение кадра (работает и при паузе)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(screenshot_dir, f"screenshot_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved as: {filename}")
            elif key == ord('d') or key == ord('D'):  # Вкл/выкл детекцию объектов
                show_detection = not show_detection
                object_roi = None
                print(f"Object detection: {'ENABLED' if show_detection else 'DISABLED'}")
            elif key == ord('r') or key == ord('R'):  # Сброс детекции
                object_roi = None
                print("Object detection reset")
            elif key == ord('i') or key == ord('I'):  # Информация
                print("\n" + "="*50)
                print(f"Device: {self.device}")
                print(f"Number of classes: {len(self.classes)}")
                print(f"Model: ResNet18")
                print(f"Confidence threshold: {confidence_threshold}")
                print("="*50 + "\n")
        
        # Освобождение ресурсов
        cap.release()
        cv2.destroyAllWindows()
        print("\nClassifier stopped.")

def main():
    """Основная функция для запуска классификатора"""
    import argparse
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Классификация фруктов и овощей с веб-камеры')
    parser.add_argument('--model', type=str, default='./resnet_model.pth',
                       help='Путь к файлу модели (по умолчанию: ./resnet_model.pth)')
    parser.add_argument('--camera', type=int, default=0,
                       help='ID камеры (по умолчанию: 0)')
    parser.add_argument('--cpu', action='store_true',
                       help='Принудительно использовать CPU')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Порог уверенности (по умолчанию: 0.5)')
    
    args = parser.parse_args()
    
    # Определение устройства
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Создание и запуск классификатора
    classifier = WebcamClassifier(
        model_path=args.model,
        device=device
    )
    
    # Запуск с обработкой исключений
    try:
        classifier.run_webcam(
            camera_id=args.camera,
            confidence_threshold=args.threshold
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()