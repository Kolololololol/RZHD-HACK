import streamlit as st
import pandas as pd
import cv2
from PIL import Image
import numpy as np
import base64
import os
import shutil
import json
import cv2
import numpy as np
from structures import TrainPath, Camera
import time
from db import clear_events_table, get_data_query_result


CONFIG_CAMERA_POLYGONS_PATH = r".\configs\camera_polygons_config"
CAMERA_NAMES_CONFIG = r".\configs\camera_source_config.json"

CAMERA_LIST = []


def config_system():
    """
        Конфигурация всех камер в системе
    """

    # получение списка камер
    with open(CAMERA_NAMES_CONFIG, "r", encoding="utf-8") as file:
        camera_names_data: dict = json.load(file)

        for camera_name, camera_info in camera_names_data.items():
            camera_source_path = camera_info["source"]
            camera_polygons_path = os.path.join(
                CONFIG_CAMERA_POLYGONS_PATH, camera_info["polygons_file_name"])

            cur_camera = Camera(camera_source_path, camera_name)

            with open(camera_polygons_path, "r", encoding="utf-8") as polygon_file:
                polygons_coord = json.load(polygon_file)

                cur_camera.insert_train_pathes(polygons_coord)

            CAMERA_LIST.append(cur_camera)


def system_start_with_data_editor(stframe, title, data_cont, info):
    """
    Главная функция - запуск системы контроля и отображение редактора данных
    """
    # Подключение к камерам
    for camera in CAMERA_LIST:
        camera._connect_to_cam()

    stop_button = st.button("Остановить систему", key="stop_button")
    count = 0
    while not stop_button:
        imgs = []
        all_detected_objects = {
            "ID": [],
            "Class Name": [],
            "Path Name": [],
            "Start Time": [],
            "Image": []
        }

        # Получаем пути и найденные на них объекты
        for camera in CAMERA_LIST:
            obj_list = camera.detect_objects()
            sorted_path_list = camera.sort_detected_obj_to_pathes(obj_list)

            for path_name, detected_objects in sorted_path_list.items():
                for detected_obj in detected_objects:
                    all_detected_objects["ID"].append(detected_obj.obj_id)
                    all_detected_objects["Class Name"].append(detected_obj.cls_name)
                    all_detected_objects["Path Name"].append(path_name)
                    all_detected_objects["Start Time"].append(detected_obj.start_time)
                    all_detected_objects["Image"].append(camera.cur_annotated_frame)
                train_path = camera.train_pathes[path_name]
                train_path.check_detected_obj(sorted_path_list[path_name])
            imgs.append(camera.cur_annotated_frame)

        # Обработка изображений без CUDA 
        resized_imgs = [cv2.resize(img, (600, 600)) for img in imgs]
        final_img = np.hstack(resized_imgs)
        
        # Обновление изображения
        final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        stframe.image(final_img, use_column_width=False)



        # Используем редактор данных для отображения обнаруженных объектов 
        count += 1
        if count == 20:
            title.subheader("Данные обнаруженных объектов")
            data_cont = display_data_editor(all_detected_objects, data_cont)
            count = 0

        info.subheader("Для наглядной статистики перейдите во вкладку - 'Статистика с камер'")
        # Проверка нажатия кнопки остановки
        if stop_button:
            stop_button = st.button("Остановить систему", key=f"stop_button_update_{time.time()}")

#Функция для преобразования изображения в base64 строки для отображения в DataFrame
def convert_image_to_base64(img):
    if isinstance(img, np.ndarray):
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    return None

def display_data_editor(data_dict, data_cont):
    """
    Функция принимает данные из словаря и выводит их в st.data_editor.
    :param data_dict: Словарь с данными для отображения.
    :param data_cont: Контейнер для отображения данных.
    """
    # Преобразуем изображения в base64 строки для отображения в DataFrame
    if 'Image' in data_dict:
        images = data_dict.pop('Image')
        data_dict['Image'] = [convert_image_to_base64(img) for img in images]
    
    # Преобразуем словарь в DataFrame для удобства отображения
    df = pd.DataFrame(data_dict)
    
    # Ограничиваем размер DataFrame для избежания переполнения
    if len(df) > 50:
        df = df.tail(50)
    
    # Используем st.data_editor для отображения данных
    data_cont.data_editor(df, column_config={
        'ID': st.column_config.NumberColumn("Идентификатор"),
        'Class Name': st.column_config.TextColumn("Название класса"),
        'Path Name': st.column_config.TextColumn("Название пути"),
        'Start Time': st.column_config.DatetimeColumn("Время начала"),
        'Image': st.column_config.ImageColumn("Изображение")
    })
    return data_cont

def upload_video():
    """
    Функция для загрузки видео файла, очистки папки uploaded_video и сохранения загруженного файла.
    Возвращает путь к загруженному файлу.
    """
    uploaded_file = st.file_uploader("Загрузите видео в удобном для вас формате", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # Очищаем папку uploaded_video, если она существует
        if os.path.exists("uploaded_video"):
            shutil.rmtree("uploaded_video")
        
        # Создаем папку uploaded_video
        os.makedirs("uploaded_video")
        
        # Сохраняем загруженный файл в папку uploaded_video
        video_path = os.path.join("uploaded_video", uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"Видео успешно загружено")
        return video_path

#Функции для выводов график статистики
def get_count_char(data : pd.DataFrame) -> None:
    data_dict = {
    "Номер камеры" : [],
    "Наименование пути" : [],
    "Количество поездов" : []
    }

    for item in data.iterrows():
        data_dict["Номер камеры"].append(item[1][0])
        data_dict["Наименование пути"].append(item[1][1])
        data_dict["Количество поездов"].append(item[1][2])

    return data_dict


def get_count_char2(data : pd.DataFrame) -> None:
    data_dict = {
    "Номер камеры" : [],
    "Количество поездов" : []
    }

    for item in data.iterrows():
        data_dict["Номер камеры"].append(item[1][0])
        data_dict["Количество поездов"].append(item[1][1])

    return data_dict

#Настройка streamlit страницы
st.set_page_config(
    page_title="ПМС-205",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.header('Автоматизированная путевая машинная станция🚉', divider="rainbow")
st.subheader("*Выберите необходимую для вас функцию*👇")
select_action = st.selectbox("Выберите необходимую для вас функцию",("-- Выберите опцию --","Видео аналитика с камер", "Статистика с камер"), label_visibility="hidden", index=0)
if select_action =="Видео аналитика с камер":
    start_button = st.button("Запуск камеры", icon="🚀")
    if start_button:
        st.subheader("***Вывод результата:***")
        clear_events_table()
        stframe = st.empty()
        title = st.empty()
        data_cont = st.empty()
        info = st.empty()
        config_system()
        system_start_with_data_editor(stframe, title, data_cont, info)

if select_action == "Статистика с камер":
    st.subheader("***Здесь собрана вся статистика в виде графиков:***")
    chart1 = st.empty()
    chart2 = st.empty()
    
    res1 = get_data_query_result("""SELECT camera_name, track_name, COUNT(*) AS track_count FROM events GROUP BY camera_name, track_name ORDER BY track_count DESC;""")
    df1 = pd.DataFrame(get_count_char(res1))
    chart1.bar_chart(data=df1, x="Наименование пути", y="Количество поездов", color="Номер камеры")

    res2 = get_data_query_result("""SELECT 
    camera_name,
    COUNT(DISTINCT object_name) AS track_count
    FROM 
    events
    GROUP BY 
    camera_name
    ORDER BY 
    camera_name;""")
    df2 = pd.DataFrame(get_count_char2(res2))
    chart2.bar_chart(data=df2, x="Номер камеры", y="Количество поездов", color="Номер камеры")
    
