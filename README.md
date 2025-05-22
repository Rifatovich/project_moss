# Moss Segmentation Project

Локальный комплект для обучения модели сегментации мха и веб‑инференса через Streamlit.

```
.
├─ moss_app.py                 # Streamlit‑интерфейс
├─ model_train.ipynb           # обучение U‑Net
├─ models/                     # сюда сохраняются .pth и .pt
├─ requirements.txt            # полный стек (train + UI)
├─ requirements_app.txt        # только для UI
├─ .streamlit/                 # конфигурация сервера (лимит upload 10 GB)
└─ project/                    # CamVid‑подобные данные + label_colors.txt
```

---

## Быстрый старт (только веб‑приложение)

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements_app.txt
streamlit run moss_app.py           # откроется http://localhost:8501
```

> ⚠️  В `models/` должен лежать `best_model_new.pt`

---

## Обучение модели

```bash
pip install -r requirements.txt
jupyter lab
```

Загрузите свой датасет
Откройте `model_train.ipynb`, выполните все ячейки.  
После обучения появятся:
* `models/best_model_new.pth`
* `models/best_model_new.pt`

---

## Запуск приложения

```bash
streamlit run moss_app.py           # откроется http://localhost:8501
```

### Конфигурация Streamlit

В .streamlit/ с `config.toml` поднят лимит загрузки до 10 GB:

```toml
[server]
maxUploadSize = 10000   # МБ
```

При необходимости измените значение и перезапустите приложение.

---

## Функциональность UI

- Загрузка изображений (одиночно, списком или ZIP‑архивом)
- Сегментация классов “green_moss / soil / background”
- Подсчёт площади мха и почвы
- Группировка по образцам и неделям (имя файла вида `<sample>-<week> …`)
- Графики динамики роста
- Скачивание сводной таблицы (.xlsx)
---

## Формат имени файла

```
111-02 RIMG1234.png
│  │
│  └── неделя  (02)
└───── образец (111)
```

---

## Структура датасета (CamVid‑подобная)

```
project/
├── Train/               # RGB-изображения
├── Trainannot/          # маски‑PNG с цветами из label_colors.txt
├── Validation/          # валидационные изображения
├── Validationannot/     # их маски
├── label_colors.txt     # R G B label — ключ для раскраски масок
├── Train.txt            # (опц.) список путей файлов Train, если нужен
└── Validation.txt       # (опц.) список путей файлов Validation
```
