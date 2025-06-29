# 📘 Назва проєкту

"Використання штучного інтелекту для керування безпілотним автомобілем"

---

## 👤 Автор

- **ПІБ**: Кащук Максим
- **Група**: ФеП-43
- **Керівник**: Ляшкевич Василь, кандидат технічних наук, доцент
- **Дата виконання**: [05.06.2025]

---

## 🎯 Мета та завдання дипломної роботи

**Мета:**
> Провести дослідження та тренування сегментаційної моделі глибинного навчання для подальшого використання в автомобільній сфері

**Завдання:**
- Проаналізувати наявні підходи для сегментації дорожнього покриття
- Провести обробку даних датасету для подальшого навчання
- Імплементувати архітектуру DeepLabV3 / DeepLabV3+
- Пропсисати конфігурацію та етапи тренування
- Імплементувати скрипти візуалізації та показу роботи моделі

---

## 📌 Загальна інформація

- **Тип проєкту**: Модель глибинного навчання
- **Мова програмування**: Python
- **Фреймворки / Бібліотеки**: TensorFlow, OpenCV, Matplotlib, Numpy, Pandas

---

## 🧠 Опис функціоналу

- 🔐 Завантаження датасету та встановлення середовища
- 🗒️ Обробка датасету, підготовка даних для тренування та формування полігональних масок для сегментації
- 💾 Допоміжний функціонал для роботи з даними для тренування
- 🌐 Реалізація архітектури глибинних мереж DeepLabV3 та DeepLabV3+ для задачі сегментації
- 📱 Навчання сегментаційної моделі глибинного навчання
- 📱 Показ виконання моделі (inference) на даних взятих з відкритих джерел

---

## 🧱 Опис основних скриптів / файлів

| Клас / Файл     | Призначення |
|----------------|-------------|
| `core/environment-setup/requirements.txt`      | Перелік залежностей, які необхідно втсановити |
| `core/environment-setup/setup.py`      | Налаштування середовища та вивантаження датасету |
| `core/data-processing/data-filter-npz.ipynb`      | Підготовка тренувальних даних у форматі .npz |
| `core/data-processing/data-filter-tfrecord.ipynb`      | Підготовка тренувальних даних у форматі .tfrecord |
| `core/data-processing/data-filter-roi-tfrecord.ipynb`      | Підготовка тренувальних даних з масштабуванням до ROI (region of interest) у форматі .tfrecord |
| `core/data-processing/example.ipynb`      | Візуалізація кінцевої сегментаційної instance-маски |
| `core/lane-worker/canny_filter_test.ipynb`      | Візуалізація зображення з датасету з фільтром Canny |
| `core/lane-worker/image_line_example.ipynb`      | Візуалізація ліній з анотацій датасету |
| `core/lane-worker/mask_coords_example.ipynb`      | Візуалізація формування полігональної маски |
| `core/model-learning/CuLane/DeepLabV3/utils/general_utils.py`      | Допоміжний функціонал (Метрики, обробка датасету та pre-processing) для DeepLabV3 |
| `core/model-learning/CuLane/DeepLabV3/deeplabv3.py`      | Реалізація моделі глибинного навчання DeepLabV3 |
| `core/model-learning/CuLane/DeepLabV3/main.py`      | Тренування моделі глибинного навчання DeepLabV3 для сегментації дорожнього покриття |
| `core/model-learning/CuLane/DeepLabV3_Plus/utils/general_utils.py`      | Допоміжний функціонал (Метрики, обробка датасету та pre-processing) для DeepLabV3+ |
| `core/model-learning/CuLane/DeepLabV3_Plus/deeplabv3_plus.py`      | Реалізація моделі глибинного навчання DeepLabV3+ |
| `core/model-learning/CuLane/DeepLabV3_Plus/main.py`      | Тренування моделі глибинного навчання DeepLabV3+ для сегментації дорожнього покриття |
| `models/deeplabv3.keras`      | Файл моделі глибинного навчання для сегментації дорожнього покриття |
| `models/inference.ipynb`      | Показ роботи моделі глибинного навчання на фото з відкритого джерела в форматі Jupyter notebook |
| `models/inference.py`      | Показ роботи моделі глибинного навчання на фото з відкритого джерела в форматі Python script |

---

## ▶️ Як запустити проєкт

### 1. Встановлення інструментів

- Встановити інтерпретатор мови програмування Python 3.12
- Виконати наступну команду для встановлення залежностей: pip install -r core/environment-setup/requirements.txt
- Вивантажити модель за наступним посиланням: (https://drive.google.com/file/d/1KwiNWcno-uAgg0B7Pp5FKjrSnJURK4Rr/view?usp=sharing) та розмістити її за шляхом models/deeplabv3.keras
- Запустити Python скрипт за допогою команди python3 models/inference.py або ж відкрити models/inference.py як Jupyter notebook, вибрати Python інтерпретатор в `Select Kernel` та запустити виконання

### 2. Клонування репозиторію

```bash
git clone https://github.com/maxkashchuk/automotive-segmentation.git
cd automotive-segmentation
```

### 3. Встановлення залежностей

```bash
pip install -r core/environment-setup/requirements.txt
```

## 📷 Приклади / скриншоти

- **Приклад instance-маски:**  
  ![Приклад instance-маски](/screenshots/instance_maks_example.png)

- **Передбачення моделі:**  
  ![Передбачення моделі](/screenshots/inference_result.png)

---

## 🧪 Проблеми і рішення

| Проблема              | Рішення                            |
|----------------------|------------------------------------|
| Не всі дані відповідають критеріям для обробки та навчання | Фільтрація даних згідно додаткових критеріїв |
| Використання mixed precision при тренуванні | Інтеграція TensorFlow 1.X API в архітектуру моделі |
| Модель вивчає пусті пікселі (padding)         | Виокремлення padding як окремий клас в instance-масці та ігнорування даного класу в loss |
| Метрики тренування DeepLabV3+ демонструють повільний ріст та повільне сходження         | Масштабування по ROI (region of interest) |
| DeepLabV3+ фокусується на другорядних деталях та показує погані результати по сегментації дорожнього покриття     | Використання DeepLabV3 для покращення генералізації та досягнення успішного результату     |

---

## 🧾 Використані джерела / література

- Spatial As Deep: Spatial CNN for Traffic Scene Understanding - Дослідження від розробників датасету CuLane
- Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation - Дослідження по DeepLabV3+
- Rethinking Atrous Convolution for Semantic Image Segmentation - Дослідження по DeepLabV3
- TensorFlow API Documentation - документація TensorFlow
- OpenCV API Documentation - документація OpenCV
- Matplotlib API Documentation - документація Matplotlib
- NumPy API Documentation - документація NumPy
- Lattice-ai. DeepLabV3-Plus-TensorFlow - приклад реалізації DeepLabV3+ на TensorFlow 2

---