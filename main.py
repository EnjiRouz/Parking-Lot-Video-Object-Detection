"""
Камеры можно поискать здесь:
https://www.insecam.org/en/bytag/Parking/
"""

import cv2
import numpy as np
import time
import json
import pytz
import datetime
import smtplib
import ssl


'''
    This program version requires a gmail connection - Go to line 266-267 and add your Gmail and your password
    if there is a connection problem with your Gmail go to Gmail settings and activate 
    'Less secure apps & your Google Account' 
'''


def draw_parking_area(img):
    """
    Размечение парковочных зон
    Σήμανση χώρων στάθμευσης
    :param img: исходное изображение -  αρχική εικόνα
    :return: изображение с размеченными зонами парковки - εικόνα με επισημασμένες θέσεις στάθμευσης
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
    now_time = time.time()
    count_history.append([count, now_time])
    duration = 30  # в секундах

    counts = [pair[0] for pair in count_history if pair[1] >= now_time - duration]
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

    return [img, parking_places-mid_max] # returns list with img and available parking places that is already counted


def apply_yolo_object_detection(img):
    """
    Распознавание и определение координат объектов на изображении
    с последующей проверкой на нахождение в парковочной зоне -
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

    # debug-рисование парковочной зоны - Area to check for free parking places
    img = draw_parking_area(img)

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
        img = draw_object(img, class_index, box, selected)

    img = draw_cars_count(img, selected_count)[0]

    '''as function draw_cars_count() returns a list with img and number of free places the same function returns 
        exactly the same, so we can have the number of free places in main() without changing all the function,
        the only thing is to use the pointers [0] for img and [1] for the free places   
    '''
    number_of_cars = draw_cars_count(img, selected_count)[1]

    return [img, number_of_cars]


# function that loads data from a json file - as a simple way to save and load data instead of DB
def load_data(name_of_json_file, data_dict):
    """
    :param name_of_json_file: name of json file, from where we want to load our data
    :param data_dict: the dict that we want to fill - it defs in main()
    :return: a filled dict - the second param
    """
    try:
        with open(name_of_json_file, "r") as file:
            data_dict = json.load(file)
    except FileNotFoundError:
        print("The JSON file doesn't exists!")
        return None
    except:
        print("Failed to open or read the data from JSON file!")
        return None
    else:
        print(f"The data has been successfully loaded from file : {data_dict}")
        return data_dict


# function that saves the data in json file
def write_data_to_json_file(name_of_json_file, data_dict):
    """
    :param name_of_json_file: Name of JSON file where we want to save our data
    :param data_dict: The dict which we want to save it's data
    :return: a new JSON file or an updated JSON file if use se same name - like in that case
    """
    with open(name_of_json_file, "w") as file:
        json.dump(data_dict, file)


# function that sends an e-mail in person who subscribed to the system
def send_notification(subscriber, number_of_free_places):

    # define the system's email - This mail will send mail to subscribers
    sender_email = "your_email_here_@gmail.com"
    sender_password = "****your_code_here*****"

    # Create mail
    mail = f"Subject: Your Minnesota Parking notification\n\n" \
           f"Hello,\n\nThere are {number_of_free_places} FREE PARKING PLACES available!!!\nDrive safe!!\nThe " \
           f"Parking Lot System"

    port = 465
    # Create a secure SSL context
    context = ssl.create_default_context()

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, subscriber, mail)
    except:
        print("Something gone wrong and the mail doesn't send..!")
    else:
        print("A Notification has been successfully sent!!")
    '''
        if there is a connection problem with your Gmail go to setting and activate 
        'Less secure apps & your Google Account'
    '''


# function that shows us all the subscribers - only their emails
def show_subscribers(data_dict):
    cnt = 1
    for subscriber in data_dict:
        print(f"{cnt}. {subscriber}")
        cnt += 1


# function that search for a person if he/she is a subscriber
def is_subscriber(email, data_dict):
    if email in data_dict:
        return True
    return None


# function take user's input and checks if the subscriber's email is valid - @ and gmail.com
def input_and_check_the_validity_of_mail():
    while True:
        users_input = input("Give your Gmail in form 'example@gmail.com' or 'q' to quit: ").strip().lower()
        if users_input == "q":
            return None
        else:
            test_input = users_input.split("@")
            if len(test_input) == 2 and test_input[1] == "gmail.com":
                return users_input
            else:
                print("Invalid input, please try again!")


# function for time insert
def insert_time_and_check_validity():
    users_time = ""

    valid_hour_inputs = [str(i) for i in range(0, 24)]
    users_input = input("Please give Hour (0 up to 23): ")
    while users_input not in valid_hour_inputs:
        users_input = input("Invalid input, try again: ")
        if users_input in valid_hour_inputs:
            break
    if len(users_input) == 1:
        users_input = "0" + users_input

    users_time += users_input
    users_time += ":"

    valid_minutes_input = [str(i) for i in range(0,60)]
    users_input = input("Please give Minutes (0 up to 59): ")
    while users_input not in valid_minutes_input:
        users_input = input("Invalid input, try again: ")
        if users_input in valid_minutes_input:
            break
    if len(users_input) == 1:
        users_input = "0" + users_input
    users_time += users_input

    return users_time


# function that prints the subscriber's hours
def show_my_sub_hours(data_dict):
    user = input_and_check_the_validity_of_mail()
    if is_subscriber(user, data_dict):
        print(f"Your Notification times for {user} is: ")
        cnt = 1
        for time_sub in data_dict[user]:
            print(f"{cnt}. {time_sub}")
            cnt += 1
    elif user is None:
        return
    else:
        print(f"User {user} Not Found!")


# function that deletes a notification time
def delete_or_add_a_notification_time(data_dict, name_of_json_file):
    user = input_and_check_the_validity_of_mail()
    if is_subscriber(user, data_dict):
        print(f"Your Notification times for {user} are: ")
        cnt = 1
        for time_sub in data_dict[user]:
            print(f"{cnt}. {time_sub}")
            cnt += 1
        print("Do you want to ADD or DELETE a notification time?")
        users_input = input("For ADD press 'a', for DELETE press 'd' and 'enter' to exit: ").strip()
        if users_input == "a":
            users_input = insert_time_and_check_validity()
            data_dict[user].append(users_input)
            write_data_to_json_file(name_of_json_file, data_dict)
            print(f"The notification time {users_input} has been added!")
        elif users_input == "d":
            if len(data_dict[user]) == 0:
                print('There are no notifications times in that list!')
                return None
            elif len(data_dict[user]) == 1:
                print(f"The last notification time from the list {data_dict[user][0]}, has been deleted!")
                print("There are no more data")
                del data_dict[user][0]
                write_data_to_json_file(name_of_json_file, data_dict)
            else:
                users_input = insert_time_and_check_validity()
                if users_input not in data_dict[user]:
                    print(f"{users_input} is not in notification times list, try again! ")
                    return None
                else:
                    data_dict[user].remove(users_input)
                    write_data_to_json_file(name_of_json_file, data_dict)
                    print(f"The notification time {users_input} has been deleted!")
    else:
        print(f"User {user} is not a Subscriber, register and come back again")
        return None


# function that deletes or adds a new user
def add_or_delete_a_subscriber(data_dict, name_of_json_file):
    print("Here you can ADD or DELETE a subscriber")
    print("Type your Gmail and if it doesn't exists you can save it, else you can delete it!")
    user = input_and_check_the_validity_of_mail()
    if is_subscriber(user, data_dict):
        print(f"{user} already exists, do you want to delete?")
        users_input = input("Press 'd' if you want to DELETE or 'enter' to quit: ")
        if users_input == "d":
            data_dict.pop(user)
            write_data_to_json_file(name_of_json_file, data_dict)
            print(f"{user} has been removed from subscribers list!")
        else:
            return None
    else:
        print(f"{user} doesn't exists, do you want to Register?")
        users_input = input("Press 'a' if you want to ADD or 'enter' to quit: ")
        if users_input == "a":
            users_input = insert_time_and_check_validity()
            data_dict[user] = [users_input]
            write_data_to_json_file(name_of_json_file, data_dict)
            print(f"{user} has been Added to subscribers list")
            print("You can add more notification times from main menu")
        else:
            return None


# the menu function
def my_menu(data_dict, name_of_json_file):
    def choices():
        print("*"*15 + "Menu" + "*"*15)
        print("1. Register/Delete Subscriber")
        print("2. Add/Delete Notification Time")
        print("3. Am I a subscriber?")
        print("4. Show the Subscribers' list")
        print("5. Show my Notifications Times")
        print("6. Continue to View Parking\n   - Start Notification checks")
        print("7. Exit")
        print("*" * 15 + "*****" + "*"*15)

    while True:
        valid_input = [str(i) for i in range(1, 8)]
        choices()
        users_input = input("Type your choice: ").strip()
        while users_input not in valid_input:
            users_input = input("Try again with a number '1' up to '7': ").strip()
        if users_input == "1":
            add_or_delete_a_subscriber(data_dict, name_of_json_file)
        elif users_input == "2":
            delete_or_add_a_notification_time(data_dict, name_of_json_file)
        elif users_input == "3":
            users_input = input_and_check_the_validity_of_mail()
            if is_subscriber(users_input, data_dict):
                print(f"Yes, {users_input} is Subscriber!")
            else:
                print(f"No, {users_input} is not Subscriber, but you can easily Register!!")
        elif users_input == "4":
            show_subscribers(data_dict)
        elif users_input == "5":
            show_my_sub_hours(data_dict)
        elif users_input == "6":
            return
        elif users_input == "7":
            exit()


if __name__ == '__main__':

    '''1st addition in main for data and menu'''
    # an empty dict for my data {'example.gmail.com': ['18:45, 13:15]}
    my_data = {}
    # the empty dict filled with data from the JSON file
    my_data = load_data("data_file.json", my_data)
    # starter menu - no need to load yolo for eg. because menu handles only the data - so we put it at the top of main
    my_menu(my_data, "data_file.json")

    '''
        for sending emails I need a dict in form {"18:45": [example1@gmail.com, example2@gmail.com], 
        "20:15": [example1@gmail.com]} because in that case time is a unique key for dict and may 
        contains more than one subscriber that must noticed - so we have to  form in that type
    '''
    Data_for_notification = {}
    for key, value in my_data.items():
        for val in value:
            if val in Data_for_notification:
                Data_for_notification[val].append(key)
            else:
                Data_for_notification[val] = [key]
    ''' 
        After the exit of menu runs cv2 to show the free parking places while checking subscribers' notification times
        if there is needed to send them an email. This check takes place one in 25 seconds. A time will be checked 3-4
        pers minute. To avoid sending 3-4 the same mail we need a variable, where includes the last time that a 
        notification has sent. So this will be a criterion of sending or not an email.
          
    '''
    last_notification = "00:00"

    # загрузка весов Yolo из файлов
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")  # или "yolov3.weights" и "yolov3-tiny.cfg"
    layer_names = net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()
    out_layers = [layer_names[index[0] - 1] for index in out_layers_indexes]

    # загрузка классов Yolo из файла - load yolo classes from file
    with open("yolo-classes.txt") as f:
        classes = f.read().split("\n")

    # определение классов Yolo, обозначающих транспорт - Yolo class that suggests transport
    transport_classes = ["car", "truck", "motorbike", "bus", "bicycle"]

    # количество парковочных мест
    parking_places = 52

    # зоны парковки (для конкретной видео-камеры) - parking places - left and right part
    parking_area = [
        [20, 730, 450, 920],
        [480, 825, 1575, 980],
    ]

    # зоны дороги (для конкретной видео-камеры), расположенные близко к парковке (эти зоны требуется не брать в расчёт)
    # Places like road near the park that there is no need to take data of them

    small_road_area = [
        [300, 730, 450, 760],
        [1520, 825, 1575, 850],
    ]

    # история количества замеченных транспортных средств в течение времени
    # (используется для подсчёта скользящего среднего)
    # history of car number that have been detected / time is to calculate the transport mean
    count_history = []

    while True:
        try:
            # захват картинки с видео - capture image from video
            video_camera_capture = cv2.VideoCapture("http://68.188.109.50/cgi-bin/camera")

            while video_camera_capture.isOpened():
                ret, frame = video_camera_capture.read()

                if not ret:
                    break

                # применение методов распознавания объектов на изображении от Yolo
                # apply methods for object detection by yolo
                frame = apply_yolo_object_detection(frame)[0]

                # перевод в цветовое пространство OpenCV
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # вывод обработанного изображения на экран - output of edited img in screen
                frame = cv2.resize(frame, (1920 // 2, 1080 // 2))

                cv2.imshow("Parking Lot", frame)

                '''2nd addition in main to send the notifications in subscribers'''
                while True:
                    now = datetime.datetime.now(tz=pytz.UTC)
                    minnesota_time = now.astimezone(pytz.timezone("US/Central"))
                    str_Minnesota_time = minnesota_time.strftime("%H:%M")
                    for element in Data_for_notification:
                        if element == str_Minnesota_time and element != last_notification:
                            for email_address in Data_for_notification[element]:
                                send_notification(email_address, apply_yolo_object_detection(frame)[1])
                            last_notification = element

                    # 25 secs = 25000 ms - refreshes every 25 sec - use like method sleep()
                    if cv2.waitKey(25000):
                        break

            video_camera_capture.release()
            cv2.destroyAllWindows()

        except KeyboardInterrupt:
            pass
