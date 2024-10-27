import os
import cv2
import json
from copy import deepcopy

"""
    В данном модуле приведен пример разметки полигонов
    железнодорожного пути ДЛЯ КОНКРЕТНОЙ КАМЕРЫ
"""

TEST_VIDEO_PATH = r".\rzhd2_dataset\20241018113004031_aa60160f16224537b055643642a236b3_AB1788721.mp4"
TEST_FRAME_PATH = r".\img\test.png"

CAMERA_CONFIG_FOLDER_NAME = r".\configs\camera_polygons_config"
DEFAULT_CAMERA_CONFIG_FILE_NAME = "camera_polygons_coords"

COLORS = [
    (156, 23, 189),
    (34, 56, 78),
    (10, 20, 30),
    (123, 234, 45),
    (45, 198, 123),
    (90, 67, 180),
    (255, 144, 0),
    (200, 123, 78),
    (76, 150, 200),
    (67, 89, 210)
]


def save_first_frame(video_path: str):
    """
        video_path - путь до видео/RTSP-камеры с которой
        (которого) будет считано изображение 
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if not ret:
        raise Exception("[ERROR] Не удалось прочитать кадр!")

    cv2.imwrite(TEST_FRAME_PATH, frame)


def config_camera(img_path: str) -> list:
    """
        img_path - путь до файла, на котором будет происходить разметка
        полигонов
    """
    all_train_pathes = {}

    cur_path_points = []  # [(x1, y1), (x2, y2), ..., (xN, yN)]
    image = cv2.imread(img_path)

    cur_path_idx = 0

    def select_points(event, x, y, flags, param):

        # Функция для обработки кликов мыши
        if event == cv2.EVENT_LBUTTONDOWN:  # Левый клик мыши для выбора точки
            cur_path_points.append((x, y))
            # Рисуем маленький круг на выбранной точке
            cv2.circle(image, (x, y), 5, COLORS[cur_path_idx], -1)
        if len(cur_path_points) > 1:
            # Соединение с предыдущей точкой
            cv2.line(image, cur_path_points[-2],
                     cur_path_points[-1], COLORS[cur_path_idx], 2)

    # Загрузка изображения и установка коллбека для мыши
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", select_points)

    print("Кликните на изображении для выбора точек. Нажмите 'q' для завершения.")
    print("Нажмите 'n' для выбора разметки следующего полигона.")
    while True:
        # Обновляем изображение с новыми точками
        cv2.imshow("Image", image)
        wait_key_val = cv2.waitKey(1)
        if wait_key_val & 0xFF == ord('q'):  # Нажмите 'q' для выхода
            break
        elif wait_key_val & 0xFF == ord('n'):
            path_name = input("Введение наименование данного пути: ")
            all_train_pathes[path_name] = deepcopy(cur_path_points)
            cur_path_points = []
            if cur_path_idx == len(COLORS) - 1:
                cur_path_idx = 0
            else:
                cur_path_idx += 1

    cv2.destroyAllWindows()

    return all_train_pathes


if __name__ == "__main__":

    # получили изображение для примера разметки
    # TODO раскомментировать, если необходимо получить изображение
    save_first_frame(TEST_VIDEO_PATH)

    all_polygons_coords = config_camera(TEST_FRAME_PATH)

    # сохраняем в файл
    # создаем путь для сохранения результата
    try_idx = 0
    while True:
        if not try_idx:
            save_path = os.path.join(
                CAMERA_CONFIG_FOLDER_NAME, DEFAULT_CAMERA_CONFIG_FILE_NAME + ".json")
        else:
            save_path = os.path.join(
                CAMERA_CONFIG_FOLDER_NAME, DEFAULT_CAMERA_CONFIG_FILE_NAME + str(try_idx) + ".json")

        if not os.path.exists(save_path):
            with open(save_path, "w", encoding="utf-8") as file:
                json.dump(all_polygons_coords, file, ensure_ascii=False)

            print(f"Конфигурация полигонов для камеры ({
                  TEST_VIDEO_PATH}) сохранена по пути: {os.path.abspath(save_path)}")

            break

        try_idx += 1
