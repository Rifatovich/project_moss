import cv2
import numpy as np
import torch
from io import BytesIO
from PIL import Image as PILImage
import streamlit as st
import tempfile
import os
from PIL import Image
from io import BytesIO as _BytesIO
import albumentations as A
import zipfile
import re
import pandas as pd
from urllib.parse import quote
from PIL import Image as PILImage

# ——— Константы ———
CLASS_LIST = ["фон", "мох", "почва"]
RESIZE_DIM = 512
MACHINE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.classes.__path__ = []
MODEL_NET = torch.jit.load('models/best_model_new.pt', map_location=MACHINE)

# ——— Преобразования ———
def build_augmentation():
    return A.Compose([
        A.LongestMaxSize(max_size=RESIZE_DIM),
        A.PadIfNeeded(min_height=RESIZE_DIM, min_width=RESIZE_DIM),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# ——— Инференс ———
def run_inference(image):
    h0, w0, _ = image.shape
    transforms = build_augmentation()
    augmented = transforms(image=image)
    processed = augmented['image']

    input_tensor = torch.from_numpy(processed).to(MACHINE).unsqueeze(0).permute(0, 3, 1, 2).float()
    MODEL_NET.eval()
    with torch.no_grad():
        prediction = MODEL_NET(input_tensor)
        softmaxed = torch.softmax(prediction, dim=1)

    softmaxed = softmaxed.squeeze().cpu().detach().numpy()
    predicted_class = np.argmax(softmaxed, axis=0)

    if h0 > w0:
        delta = int(((h0 - w0) / 2) / h0 * RESIZE_DIM)
        predicted_class = predicted_class[:, delta + 1 : RESIZE_DIM - delta - 1]
    elif w0 > h0:
        delta = int(((w0 - h0) / 2) / w0 * RESIZE_DIM)
        predicted_class = predicted_class[delta + 1 : RESIZE_DIM - delta - 1, :]

    resized_mask = cv2.resize(predicted_class, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return resized_mask

# ——— Загрузка и парсинг ———
def parse_filename(file_name):
    match = re.match(r"(\d+)-(\d+)", file_name)
    if match:
        return match.group(1), int(match.group(2))
    return file_name, -1

def unzip_and_collect_images(zip_file):
    temp_dir = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir.name)

    collected = []
    for root, _, files in os.walk(temp_dir.name):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, file)
                try:
                    image = np.array(Image.open(path).convert("RGB"))
                    collected.append((file, image))
                except:
                    st.warning(f"Файл {file} не удалось открыть")
    return collected

def load_uploaded_images(label_text):
    uploads = st.file_uploader(label_text, accept_multiple_files=True)
    result = []
    for file in uploads or []:
        name = file.name.lower()
        if name.endswith(('.jpg', '.jpeg', '.png')):
            try:
                image = np.array(Image.open(file).convert("RGB"))
                result.append((file.name, image))
            except:
                st.warning(f"Файл {file.name} не удалось открыть")
        elif name.endswith('.zip'):
            result.extend(unzip_and_collect_images(file))
        else:
            st.warning(f"Файл {file.name} не поддерживается")
    return result

# ——— Маска поверх изображения ———
def render_mask_overlay(mask: np.ndarray, base_img: np.ndarray, visibility: dict):
    color_scheme = {
        "фон": np.array([0, 0, 0]),
        "мох": np.array([42, 125, 209]),      
        "почва": np.array([210, 180, 140]),   
    }

    overlay = base_img.copy()
    transparent = np.zeros_like(base_img)

    for idx, label in enumerate(CLASS_LIST):
        if visibility.get(label, False):
            if label == "фон":
                overlay[mask == idx] = color_scheme[label]  # фон — полностью
            else:
                transparent[mask == idx] = color_scheme[label]  # полупрозрачные

    blended = cv2.addWeighted(overlay, 1, transparent, 0.7, 0)
    return blended

# ——— Интерфейс Streamlit ———
def main_app():
    st.set_page_config(
        page_title="Сегментация мха",
        page_icon='🌿',
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""<style>h1 a.anchor-link, h2 a.anchor-link, h3 a.anchor-link {
    display: none !important;}</style>""", unsafe_allow_html=True)
    st.title('Анализ изображения: сегментация мха')

    images = load_uploaded_images('Загрузите одно или несколько изображений (или .zip)')
    if not images:
        return

    results_data = []
    for name, img in images:
        mask = run_inference(img)
        moss = np.sum(mask == 1)
        soil = np.sum(mask == 2)
        total = moss + soil
        percent = moss / total * 100 if total > 0 else 0
        sample_id, week = parse_filename(name)
        results_data.append({
            "Файл": name,
            "Образец": sample_id,
            "Неделя": week,
            "Покрытие мхом (%)": round(percent, 2)
        })

    df = pd.DataFrame(results_data).sort_values(by=["Образец", "Неделя"])
    st.markdown("### 📊 Сводная таблица покрытия мхом по образцам")
    st.dataframe(df)

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Анализ мха")

    st.sidebar.download_button(
        label="📥 Скачать таблицу (.xlsx)",
        data=buf.getvalue(),
        file_name="сводная_таблица_мха.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    selected = st.selectbox("Выберите образец для анализа динамики:", df["Образец"].unique())
    filtered = df[df["Образец"] == selected].sort_values("Неделя")
    st.line_chart(filtered.set_index("Неделя")["Покрытие мхом (%)"])

    st.sidebar.markdown("### 🖌️ Показать маски по умолчанию:")
    default_mask = {
        "мох": st.sidebar.checkbox("Мох", value=True),
        "почва": st.sidebar.checkbox("Почва", value=False),
        "фон": st.sidebar.checkbox("Фон", value=False)
    }

    st.sidebar.markdown("### 🧭 Быстрый переход:")
    for name, _ in images:
        encoded = quote(name)
        st.sidebar.markdown(f"<a href='#{encoded}'>{name}</a>", unsafe_allow_html=True)

    for name, image in images:
        encoded = quote(name)
        st.markdown(f"<a name='{encoded}'></a>", unsafe_allow_html=True)
        st.markdown(f"### 🖼 {name}")
        mask = run_inference(image)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Оригинальное изображение")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("Маска сегментации")

        st.markdown(f"**Показать маски для {name}:**")
        col_mask = st.columns(len(CLASS_LIST))
        toggles = {}
        for i, label in enumerate(CLASS_LIST):
            with col_mask[i]:
                toggles[label] = st.checkbox(label.capitalize(), value=default_mask.get(label, False), key=f"{name}_{label}")

        if all(toggles[k] == default_mask[k] for k in CLASS_LIST):
            toggles = default_mask

        with col2:
            st.image(render_mask_overlay(mask, image, toggles), use_container_width=True)

        moss_area = np.sum(mask == 1)
        soil_area = np.sum(mask == 2)
        total_area = moss_area + soil_area
        percent = moss_area / total_area * 100 if total_area > 0 else 0
        
        # Сохранение маски в цветах для обучения
        from PIL import Image as PILImage
        from io import BytesIO as _BytesIO
        color_map = {
            0: [0, 0, 0],
            1: [42, 125, 209],
            2: [170, 240, 209],
        }
        mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for idx, color in color_map.items():
            mask_rgb[mask == idx] = color

        buf = BytesIO()
        PILImage.fromarray(mask_rgb).save(buf, format="PNG")
        st.download_button(
            label="📥 Скачать маску",
            data=buf.getvalue(),
            file_name=f"{name}_mask.png",
            mime="image/png"
        )

        if "all_masks_zip" not in st.session_state:
            st.session_state["all_masks_zip"] = {}
        st.session_state["all_masks_zip"][name] = mask_rgb

        st.markdown(f"**Покрытие мхом:** {percent:.2f}%")


    # Кнопка скачивания всех масок
    import tempfile, zipfile, os
    if "all_masks_zip" in st.session_state:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "moss_masks.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for name, mask_rgb in st.session_state["all_masks_zip"].items():
                    out_path = os.path.join(tmpdir, f"{name}_mask.png")
                    PILImage.fromarray(mask_rgb).save(out_path)
                    zipf.write(out_path, arcname=os.path.basename(out_path))
            with open(zip_path, "rb") as f:
                st.sidebar.download_button(
                    label="📦 Скачать все маски (ZIP)",
                    data=f.read(),
                    file_name="moss_masks.zip",
                    mime="application/zip"
                )


if __name__ == '__main__':
    main_app()