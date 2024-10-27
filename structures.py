import os

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from shapely.geometry import Polygon, box

import collections
from typing import Optional

import datetime

from db import add_event


class Camera():
    """
        Объект камеры, с изображений которой
        будет производиться распознавание
    """

    def __init__(self, video_path: str, camera_name_: str):
        # путь до камеры (RTSP / *.mp4)
        self.camera_name = camera_name_
        self._video_path: str = video_path
        self.cv_cam: Optional[cv2.VideoCapture] = None        # объект камеры
        self.train_pathes = {}                              # массив путей

        self.path_to_log_file = os.path.join('logs', self.camera_name + ".txt")

        with open(file=self.path_to_log_file, mode="w", encoding="utf-8") as file:
            file.write("")

        # объект изображения, с отрисованными рамками
        self.cur_annotated_frame = None

        self.model = YOLO("models/best.pt")

        self._connect_to_cam()

    def _connect_to_cam(self) -> bool:
        """
            Функция подключения к камере / источнику видео
        """
        self.cv_cam = cv2.VideoCapture(self._video_path)
        ret, _ = self.cv_cam.read()

        if not ret:
            self.cv_cam = cv2.VideoCapture(self._video_path)
            ret, _ = self.cv_cam.read()
            if not ret:
                raise "[ERROR] Не удалось прочитать изображение с камеры!"

    def insert_train_pathes(self, train_polygons_config_data: dict) -> bool:
        """
            Конфигурация полигонов для конкретной камеры
        """
        for (path_name, coords) in train_polygons_config_data.items():
            self.train_pathes[path_name] = TrainPath(
                path_name, self.camera_name, coords, self.path_to_log_file)

    def detect_objects(self):
        """
            Cоздание объектов на основе детекции
            нейронной сети
        """

        ret, frame = self.cv_cam.read()
        if not ret:
            self.cv_cam = cv2.VideoCapture(self._video_path)
            ret, frame = self.cv_cam.read()
            if not ret:
                raise "[ERROR] Не удалось прочитать изображение с камеры!"

        track_res = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml")[0]

        self.cur_annotated_frame = track_res.plot()

        obj_list = []       # массив всех найденных объектов

        for obj_info in track_res.boxes:
            id = int(obj_info.id)           # получаем id объекта
            # уверенность распознавания
            conf = round(float(obj_info.conf), 2)

            # FIXME Преобразовать класс в текст
            cls_idx = self.model.names[int(obj_info.cls)]

            coords = np.array(obj_info.xyxy[0], dtype=np.uint32)

            obj_list.append(DetectedObject(id, cls_idx, coords, datetime.datetime.now(),
                                           None))

        return obj_list

    def sort_detected_obj_to_pathes(self, detected_obj_list: list) -> dict:
        """
            Соотношение координат объектов к координатам
            путей на изображении
        """

        sorted_obj_dict = {}

        for detected_obj in detected_obj_list:

            max_path_conf = -1
            max_conf_path_name = ""

            for train_path_name, train_path_obj in self.train_pathes.items():
                conf = train_path_obj.get_overlay_percentage(
                    detected_obj.polygon)
                if max_path_conf < conf:
                    max_path_conf = conf
                    max_conf_path_name = train_path_name

            if max_path_conf > 50:              # FIXME: сделать ПЕРЕМЕННОЙ
                if sorted_obj_dict.get(max_conf_path_name, None):
                    sorted_obj_dict[max_conf_path_name].append(detected_obj)
                else:
                    sorted_obj_dict[max_conf_path_name] = [detected_obj]

        return sorted_obj_dict


class TrainPath():
    """
        Объект железнодорожного пути
    """

    def __init__(self, name_, camera_name_, n_sides_coords, log_file_path_: str):
        """
            n_sides_coords - Координаты вершин N-сторонней фигуры
            пример: [(x1, y1), (x2, y2), (x3, y3), ..., (xn, yn)]
        """
        self.name: str = name_
        self.camera_name = camera_name_
        self.polygon: Optional[Polygon] = Polygon(
            n_sides_coords)        # фигура железнодорожного пути
        self.detected_objects = {}    # мапа текущих объектов

        self.log_file_path = log_file_path_

    def get_overlay_percentage(self, rectangle: Polygon) -> float:
        """
            Description: 

            rectange - полигон распознанного объекта
        """

        # Нахождение пересечения
        intersection = self.polygon.intersection(rectangle)

        # Площадь наложения
        overlay_area = intersection.area

        # Площадь прямоугольника (можно также использовать площадь n_sided_polygon)
        rectangle_area = rectangle.area

        # Процентное соотношение
        overlay_percentage = (overlay_area / rectangle_area) * 100

        return overlay_percentage

    def check_detected_obj(self, new_detected_obj):
        """
            Проверка текущих объектов на пути и добавление новых
        """

        for detect_obj_data in self.detected_objects.values():

            if not detect_obj_data.save_flag:
                detect_obj_data.save_count += 1

            detect_obj_data.save_flag = False

        for new_detect_obj_data in new_detected_obj:
            new_detect_obj_data_name = "_".join([str(new_detect_obj_data.cls_name),
                                                str(new_detect_obj_data.obj_id)])

            if not self.detected_objects.get(new_detect_obj_data_name, None):
                self.detected_objects[new_detect_obj_data_name] = new_detect_obj_data

                with open(file=self.log_file_path, mode="a", encoding="utf-8") as file:
                    obj_name = new_detect_obj_data.cls_name + "_" + \
                        str(new_detect_obj_data.obj_id)
                    file.write(
                        f"[{new_detect_obj_data.start_time.strftime(f"%D.%M.%Y %H:%M:%S")}] Объект [{obj_name}] появился на пути [{self.name}]\n")

                add_event(self.camera_name, self.name, obj_name,
                          "OBJ_APPEAR", new_detect_obj_data.start_time, 0.5)

            else:
                self.detected_objects[new_detect_obj_data_name].save_flag = True
                self.detected_objects[new_detect_obj_data_name].save_count = 0

        deleted_keys = []
        for key in self.detected_objects.keys():

            if self.detected_objects[key].save_count > 50:

                self.detected_objects[key].end_time = datetime.datetime.now()

                with open(file=self.log_file_path, mode="a", encoding="utf-8") as file:
                    obj_name = self.detected_objects[key].cls_name + "_" + \
                        str(self.detected_objects[key].obj_id)
                    file.write(
                        f"[{self.detected_objects[key].end_time.strftime(f"%D.%M.%Y %H:%M:%S")}] Объект [{obj_name}] исчез с пути [{self.name}]\n")

                obj_name = self.detected_objects[key].cls_name + "_" + \
                    str(self.detected_objects[key].obj_id)

                add_event(self.camera_name, self.name, obj_name,
                          "OBJ_LEAVE", self.detected_objects[key].end_time, 0.5)

                
                deleted_keys.append(key)

        for key in deleted_keys:
            del self.detected_objects[key]


class DetectedObject():
    """
        Объект, распознанный моделью
        нейронной сети
    """

    def __init__(self, obj_id_: int, cls_name_: str,
                 coords: list, start_time_: ..., time_end_=None):

        self.obj_id = obj_id_           # ID-объекта
        self.cls_name = cls_name_       # Наименование класса

        self.polygon = box(*coords)     # площадь объекта на изображении

        self.start_time = start_time_   # время появления объекта
        self.end_time = time_end_       # время исчезновения объекта

        self.save_flag = True
        self.save_count = 0
