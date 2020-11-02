"""
Камеры можно поискать здесь:
https://www.insecam.org/en/bytag/Parking/
"""

import cv2
import numpy as np
import time


def draw_parking_area(img):
    """
    Размечение парковочных зон
    :param img: исходное изображение
    :return: изображение с размеченными зонами парковки
    """
    for area in parking_area:
        img = draw_area_rectangles(area, img, (255, 0, 0))
    return img


def draw_road_area(img):
    """
    Размечение зон дороги, расположенных близко к парковке (которые требуется не брать в расчёт)
    :param img: исходное изображение
    :return: изображение с размеченными зонами дороги
    """
    for area in small_road_area:
        img = draw_area_rectangles(area, img, (0, 0, 255))
    return img


def draw_area_rectangles(area, img, color):
    """
    Рисование прямоугольников для размечения зон (для наглядности)
    :param area: зона (координаты x,y левой верхней и правой нижней точек)
    :param img: исходное изображение
    :param color: цвет прямоугольника
    :return: изображение с нарисованными заданным цветом прямоугольниками, размечающими требуемую зону
    """
    x1, y1, x2, y2 = area
    start = (x1, y1)
    end = (x2, y2)
    color = color
    width = 2
    img = cv2.rectangle(img, start, end, color, width)
    return img


def check_car_in_parking_area(box, class_index):
    """
    Проверка на то, находится ли транспорт в парковочной зоне
    :param box: обведённая область вокруг объекта (координаты объекта)
    :param class_index: индекс определённого с помощью Yolo класса объекта
    :return: True, если транспорт в зоне парковки
    """
    name = classes[class_index]
    if name not in transport_classes:
        return False

    x, y, obj_width, obj_height = box
    car_x_center = x + obj_width // 2
    car_y_center = y + obj_height // 2

    for area in parking_area:
        x1, y1, x2, y2 = area
        if x1 <= car_x_center <= x2 and y1 <= car_y_center <= y2:
            for road_area in small_road_area:
                x1, y1, x2, y2 = road_area
                if x1 <= car_x_center <= x2 and y1 <= car_y_center <= y2:
                    return False
                return True
    return False


def draw_object(img, index, box, is_selected_object_in_parking_area):
    """
    Рисование объекта с подписями
    :param img: исходное изображение
    :param index: индекс определённого с помощью Yolo класса объекта
    :param box: обведённая область вокруг объекта (координаты объекта)
    :param is_selected_object_in_parking_area: при рисовании объекта учитывается то, находится ли он в зоне парковки
                                               для проведения цветовой пометки
    :return: изображение с отмеченными объектами
    """
    if classes[index] not in transport_classes:
        return img

    if is_selected_object_in_parking_area:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    x, y, w, h = box
    start = (x, y)
    end = (x + w, y + h)
    width = 2
    img = cv2.rectangle(img, start, end, color, width)

    start = (x, y - 10)
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 2
    text = classes[index]
    img = cv2.putText(img, text, start, font, font_size, color, width, cv2.LINE_AA)

    return img


def draw_cars_count(img, count):
    """
    Вывод информации о возможном количестве транспортных средств на парковке и примерном количестве свободных мест
    Для подсчётов используется скользящее среднее (при обработке нескольких кадров)
    :param img: исходное изображение
    :param count: текущее количество
    :return: изображение c информацией о возможном количестве транспортных средств на парковке
             и примерном количестве свободных мест
    """
    now = time.time()
    count_history.append([count, now])
    duration = 30  # в секундах

    counts = [pair[0] for pair in count_history if pair[1] >= now - duration]
    mid = sum(counts) / len(counts)
    mid_max_counts = [count for count in counts if count >= mid]
    mid_max = sum(mid_max_counts) / len(mid_max_counts)
    mid_max = int(np.ceil(mid_max))

    # текст выводится с обводкой (чтобы было видно при разном освещении картинки)
    white_color = (255, 255, 255)
    black_outline_color = (0, 0, 0)

    start = (45, 125)
    font_size = 1.2
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 2
    text = str(mid_max) + " Transport Means"
    img = cv2.putText(img, text, start, font, font_size, black_outline_color, width*3, cv2.LINE_AA)
    img = cv2.putText(img, text, start, font, font_size, white_color, width, cv2.LINE_AA)

    start = (45, 185)
    text = str(parking_places - mid_max) + " Free Parking Places"
    img = cv2.putText(img, text, start, font, font_size, black_outline_color, width*3, cv2.LINE_AA)
    img = cv2.putText(img, text, start, font, font_size, white_color, width, cv2.LINE_AA)
    return img


def apply_yolo_object_detection(img):
    """
    Распознавание и определение координат объектов на изображении
    с последующей проверкой на нахождение в парковочной зоне
    :param img: исходное изображение
    :return: изображение с размеченными объектами и подписями к ним, а также информации о количестве
             транспортных средств на парковке и примерном количестве свободных парковочных мест
    """
    height, width, depth = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (608, 608), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)

    class_indexes = []
    class_scores = []
    boxes = []

    # запуск поиска и распознавания объектов на изображении
    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]
            if class_score > 0:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)

                x = center_x - obj_width // 2
                y = center_y - obj_height // 2

                box = [x, y, obj_width, obj_height]
                boxes.append(box)
                class_indexes.append(class_index)
                class_scores.append(float(class_score))

    # проведение выборки
    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)

    # debug-рисование парковочной зоны
    # img = draw_parking_area(img)

    # запуск проверки на наличие транспортного средства в парковочной зоне
    selected_count = 0

    for box_index in chosen_boxes:
        box_index = box_index[0]
        box = boxes[box_index]
        class_index = class_indexes[box_index]

        selected = check_car_in_parking_area(box, class_index)
        if selected:
            selected_count += selected

        # debug-рисование объектов, входящих в искомые классы
        # img = draw_object(img, class_index, box, selected)

    img = draw_cars_count(img, selected_count)

    return img


if __name__ == '__main__':

    # загрузка весов Yolo из файлов
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # или "yolov3.weights" и "yolov3-tiny.cfg"
    layer_names = net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()
    out_layers = [layer_names[index[0] - 1] for index in out_layers_indexes]

    # загрузка классов Yolo из файла
    with open("yolo-classes.txt") as f:
        classes = f.read().split("\n")

    # определение классов Yolo, обозначающих транспорт
    transport_classes = ["car", "truck", "motorbike", "bus", "bicycle"]

    # количество парковочных мест
    parking_places = 50

    # зоны парковки (для конкретной видео-камеры)
    parking_area = [
        [20, 730, 450, 920],
        [480, 825, 1575, 980],
    ]

    # зоны дороги (для конкретной видео-камеры), расположенные близко к парковке (эти зоны требуется не брать в расчёт)
    small_road_area = [
        [300, 730, 450, 760],
        [1520, 825, 1575, 850],
    ]

    # история количества замеченных транспортных средств в течение времени
    # (используется для подсчёта скользящего среднего)
    count_history = []

    while True:
        try:
            # захват картинки с видео
            video_camera_capture = cv2.VideoCapture("http://68.188.109.50/cgi-bin/camera")

            while video_camera_capture.isOpened():
                ret, frame = video_camera_capture.read()
                if not ret:
                    break

                # применение методов распознавания объектов на изображении от Yolo
                frame = apply_yolo_object_detection(frame)

                # перевод в цветовое пространство OpenCV
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # вывод обработанного изображения на экран
                frame = cv2.resize(frame, (1920 // 2, 1080 // 2))
                cv2.imshow("Parking Lot", frame)
                if cv2.waitKey(0):
                    break

            video_camera_capture.release()
            cv2.destroyAllWindows()

        except KeyboardInterrupt:
            pass
