#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    rodrigues_and_translation_to_view_mat3x4
)


def triangulate_points(points2d_1, points2d_2, camera_view_1, camera_view_2, proj_mat):
    """
    points2d_1 и points2d_2 - два массива двумерных точек (с двух кадров), между которыми есть двумерное соответствие
    (то есть фактически они получены на обоих кадрах из одних и тех же трехмерных точек) - это numpy массивы одинаковой
    высоты и ширины 2 (по две координаты в каждой строке)

    camera_view_1, camera_view_2 -  view-матрицы камеры в обоих кадрах - numpy массивы 3 на 4
    proj_mat - матрица проекции камеры (фактически - матрица внутренних параметров камеры) - numpy массив 3 на 3

    return: функци я выдаёт массив трехмерных точек - проекцией котрых и являются points2d_1 и points2d_2 - это
    numpy массив той же высоты, что points2d_1 и ширины 3
    """
    camera_view_1 = np.vstack((camera_view_1, [0, 0, 0, 1]))  # добавляем к матрице строку 0,0,0,1 - делая матрицу
    # view (видовую) вида 4x4 (именно так выглядит эта матрица для однородных координат) - было в 4 лекции
    camera_view_2 = np.vstack((camera_view_2, [0, 0, 0, 1]))  # то же самое
    proj_mat = np.hstack((np.vstack((proj_mat, [0, 0, 1])), [[0], [0], [1], [0]]))  # матрицу проекций тоже делаем
    # 4 на 4 - тоже в 4 лекции была
    # (просто обычно последнюю строчку в таких матрицах откидывают, так как они всегда одинаковые, но наша
    # функция работает с матрицами 4 на 4)

    # а дальше функция, взятая с практики 6 - она уже ищет трехмерные точки как описано в лекции 6, но для
    # матриц 4 на 4
    Pi_1 = proj_mat @ camera_view_1  # единственное отличие от функции с практики - тут не нужно брать обратную, так
    Pi_2 = proj_mat @ camera_view_2  # как нам на входе сразу подают view-матрицу (а не матрицу позиции камеры, которая
    # является обратной к view)

    ans = []
    for point_1, point_2 in zip(points2d_1, points2d_2):
        a = np.vstack(((Pi_1[3, :] * point_1[0] - Pi_1[0, :]),  # матрица как в лекции 6 (слайд 15) для решения системы,
                       (Pi_1[3, :] * point_1[1] - Pi_1[1, :]),  # но в отличие от лекции, тут матрицы 4 на 4, поэтому
                       (Pi_2[3, :] * point_2[0] - Pi_2[0, :]),  # последний столбик этой матрицы - это неоднородность в
                       (Pi_2[3, :] * point_2[1] - Pi_2[1, :])))  # ситеме уравнений (вектор b в np.linalg.lstsq)
        x = np.linalg.lstsq(a=a[:, :-1], b=-a[:, -1], rcond=None)  # решаем ситсему
        ans.append(x[0])  # добавляем в массив трехмерную точку x[0] - она проецируется в точки point_1 и point_2
    return np.array(ans)


def compute_reprojection_errors(points3d, points2d, proj_mat, camera_view):
    """
    points2d - имеющиеся 2d точки на изображении - это двумерный numpy массив ширины 2
    points3d - трехмерные точки, которые, мы предполагаем, будут проецироваться в 2d точки - это двумерный numpy
    массив той же высоты, что points2d, но ширины 3
    proj_mat - матрица проекции камеры (фактически - матрица внутренних параметров камеры) - numpy массив 3 на 3
    camera_view - view-матрица камеры в том же кадре, откуда 2d точки - numpy массив 3 на 4
    """
    view_proj = np.hstack((np.vstack((proj_mat, [0, 0, 1])), [[0], [0], [1], [0]])) @ \
                np.vstack((camera_view, [0, 0, 0, 1]))  # приводим матрицы
    # к виду 4 на 4 и перемножаем - получаем матрицу, которая сразу и точки относительно камеры в пространстве
    # поворачивает (с помощью view) и проецирует - а далее код функции из практики 6:
    points3d_in_4d = np.hstack((points3d, np.ones(len(points3d)).reshape(-1, 1)))  # добавляем столбец единиц,
    # чтобы сделать точки 4 мерными (однородные координаты из 4 лекции)
    get_points = view_proj @ points3d_in_4d.T  # проецирем точки
    get2d = (get_points[[0, 1], :] / get_points[3, :]).T  # выкидываем лишнюю строку и нормируем на последнюю
    # координтау (на 4 координату -там 1 по умолчанию дожна стоять в однородных координатах)
    return np.sqrt((points2d[:, 0] - get2d[:, 0]) ** 2 + (points2d[:, 1] - get2d[:, 1]) ** 2)


def choose_best_next_frame(left_lim_1, right_lim_1, left_lim_2, right_lim_2, corner_storage, corners_id_for_3d_points):
    """
    Эта функция выбирает кадр, для которого следующим искать положение камеры в нем.

    Выбирать мы будем кадр с наибольшим количеством 2d-3d соответствий - то есть уголков в этом кадре, для которых
    уже найдены 3d точки (которые в эти уголки проецируются (как мы помним, и для уголков, и для 3d точек у нас есть
    id, которые однозначно их характеризуют - те можем понять, какие уголки каким 3d точкам соответствуют)

    Выбирать будем только среди кадров, соседних с теми, для которых уже нашли позицию камеры (ведь так больше
    шансов найти кадр с наибольши количество 2d-3d соответствий) - как мы уже говорили,
    у нас две области номеров таких кадров: [left_lim_1 ... right_lim_1] и [left_lim_2 ... right_lim_2] -
    - так что, соседние с ними кадры и перебираем
    """
    interesting_frames = []  # номера кадров, которые можно сейчас рассмотреть - это соседние кадры с кадрами, для
    # которых уже известны положения камеры (так больше шансов найти 2d-2в соответствия)
    if left_lim_1 > 0:
        interesting_frames.append(left_lim_1 - 1)
    if right_lim_2 < len(corner_storage) - 1:
        interesting_frames.append(right_lim_2 + 1)
    if right_lim_1 < left_lim_2 - 2:
        interesting_frames.append(right_lim_1 + 1)
        interesting_frames.append(left_lim_2 - 1)
    if right_lim_1 == left_lim_2 - 2:
        interesting_frames.append(right_lim_1 + 1)

    max_common_id = 0  # максимальное количество 2d-3d соответствий (то есть уголков (2d точек) на интересующем
    # нас кадре, для которых уже найдены 3d точки
    best_frame = -1  # номер кадра с этим лучшим 2d-3d соответствием
    for intr_frame in interesting_frames:
        num_3d_2d = len(np.intersect1d(corner_storage[intr_frame].ids, corners_id_for_3d_points))
        if num_3d_2d > max_common_id:
            max_common_id = num_3d_2d
            best_frame = intr_frame
    assert (max_common_id > 3)  # проверяем, что вообще нашелся кадр с достаточным количество 2d-3d соответствий

    new_left_lim_1, new_right_lim_1, new_left_lim_2, new_right_lim_2 =\
        left_lim_1, right_lim_1, left_lim_2, right_lim_2
    if best_frame == left_lim_1 - 1:  # двигаем нужную границу
        new_left_lim_1 = best_frame
    elif best_frame == right_lim_1 + 1:
        new_right_lim_1 = best_frame
    elif best_frame == left_lim_2 - 1 and right_lim_1 < left_lim_2 - 2:
        new_left_lim_2 = best_frame
    elif best_frame == right_lim_2 + 1:
        new_right_lim_2 = best_frame
    return best_frame, new_left_lim_1, new_right_lim_1, new_left_lim_2, new_right_lim_2


def best_frame_for_triangl(new_frame, corner_storage, frame_with_found_cam):
    """
    Эта функция для кадра с номером new_frame возвращает номер кадра, для которого лучше сделать триангуляцию
    новых 3d точек - конечно, мы просто берем соседний кадр с наибольшим пересечением уголков с кадром new_frame
    (и для которого уже известна камера)
    """
    if new_frame == 0:  # если это первый кадр
        return 1
    elif new_frame == len(corner_storage) - 1:  # если это последний кадр
        return len(corner_storage) - 2
    else:
        if (new_frame - 1) in frame_with_found_cam and (new_frame + 1) not in frame_with_found_cam:
            return new_frame - 1
        elif (new_frame - 1) not in frame_with_found_cam and (new_frame + 1) in frame_with_found_cam:
            return new_frame + 1
        else:
            _corners1 = corner_storage[new_frame - 1].ids
            _corners2 = corner_storage[new_frame + 1].ids
            _new_corners = corner_storage[new_frame].ids

            if len(np.intersect1d(_new_corners, _corners1)) >= len(np.intersect1d(_new_corners, _corners2)):
                return new_frame - 1
            else:
                return new_frame + 1


REPROJECTION_ERROR = 3


def track_and_calc_colors(camera_parameters: CameraParameters,  # параметры камеры
                          corner_storage: CornerStorage,  # откуда уголки брать (можно обращаться по индексу -
                          frame_sequence_path: str,                         # - номеру кадра (нумерация с 0))
                          known_view_1: Optional[Tuple[int, Pose]] = None,  # известные положения камеры в двух кадрах
                          #                             ^    ^
                          #                             |    позиция камеры (которые можно перевести во view-матрицу)
                          #                             номер кадра, где задана позиция
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:  # возвращаем список позиций камеры (индексирован номерами кадров) и 3d точки

    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(  # матрица внутренних параметров камеры (где фокусное расст и центр камеры)
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    # TODO: implement
    """
    Заведем два массива - первый (corners_id_for_3d_points) - содержит в себе id (то есть номера) тех уголков, для
    которых найдены 3d точки (то есть те 3d точки, которые проецируется в эти уголки (ведь уголки с одним и тем же id - 
    это фактически изображение одних и тех же 3d точек (и они могут быть на разных кадрах))
    
    Второй массив (found_3d_points) - это соответствующие 3d точки
    """
    corners_id_for_3d_points = np.array([])
    found_3d_points = np.empty((0, 3))  # массив будет ширины 3 (3 координаты трехмерной точки)

    """
    А также еще два массива:
    frame_with_found_cam - номера кадров, где уже нашли положение камеры
    view_mats - эти самые найденные положения камеры    
    """
    frame_with_found_cam = []
    view_mats = []

    """
    Для начала - ТРИАНГУЛЯЦИЯ, то есть ищем трехмерные точки, которые соответствуют 2d-точкам на двух
    кадрах с уже известными позиция камеры
    """
    print("Начала исходной триангуляции...")
    known_frame_1 = known_view_1[0]  # номер первого кадра, для которого известно положения камеры
    known_frame_2 = known_view_2[0]  # номер второго кадра
    known_view3x4_1 = pose_to_view_mat3x4(known_view_1[1])  # известные view-матрицы позиций камеры в обоих кадрах
    known_view3x4_2 = pose_to_view_mat3x4(known_view_2[1])           # (то есть это матрицы внешних мараметров камеры)

    frame_with_found_cam.append(known_frame_1)  # добавляем то, что известно в массивы
    frame_with_found_cam.append(known_frame_2)
    view_mats.append(known_view3x4_1)
    view_mats.append(known_view3x4_2)

    known_1_corners = corner_storage[known_frame_1]  # уголки в кадрах, для которыз позиции камеры известны
    known_2_corners = corner_storage[known_frame_2]            # (имеют тип FrameCorners)
    ids_common_known_corners = np.intersect1d(known_1_corners.ids, known_2_corners.ids)  # индексы уголков, которые
    # встречаются в обоих кадрах, для которых известны позиции камеры - просто перечекае массивы с индексами уголков
    # обоих кадров -  то есть это фактически двумерные соответствия между этими кадрами - эти уголки (с этими индексами)
    # встречаются на обоих кадрах (тк их id одно и то же)
    assert (len(ids_common_known_corners) > 3)  # проверяем, что у нас достаточно много двумерных соответствий
    mask_common_ids_1_in_2 = np.in1d(known_1_corners.ids, known_2_corners.ids)  # но нам удобнее пользоваться не самими
    # уголками в пересечении - а numpy-маской, которая имеет вид numpy массива той же длины, что и known_1_corners.ids
    # и содержит True а том месте, где элемент (те индекс уголка) из known_1_corners.ids встречается и в
    # known_2_corners.ids, и False иначе...
    # короче, тут просто смотри, какие уголки из known_1_corners.ids есть в known_2_corners.ids
    mask_common_ids_2_in_1 = np.in1d(known_2_corners.ids, known_1_corners.ids)  # тут наоборот - смотрим, какие уголки
    # из known_2_corners.ids встречаются в known_1_corners.ids (то есть результат - снова numpy-маска, но на этот раз
    # это маска для known_2_corners.ids - те имеет ту же длину и содержит True на том месте, на котором элемент
    # из known_2_corners.ids содержится в known_1_corners.ids и False на том месте, чей элемент не содержится

    points2d_1 = known_1_corners.points[mask_common_ids_1_in_2]  # получаем двумерные точки на обоих кадрах, между
    points2d_2 = known_2_corners.points[mask_common_ids_2_in_1]  # которыми есть двумерное соответствие (те им одни и
    # и те же уголки соответствуют - фактически это значит, что эти двумерные точки - проекции одних и тех же
    # трехмерных точек) - в следущей строке находим эти трехмерные точки:
    points3d_for_1and2 = triangulate_points(points2d_1, points2d_2, known_view3x4_1, known_view3x4_2, intrinsic_mat)
    """заметим!!!:
    во FrameCorners уголки идут отсортированно - те в порядке увеличения их id, поэтому ids_common_known_corners,
    points2d_1 и points_2d_2 идут в одном и том же порядке (то есть для каждого i двумерная точка points2d_1[i]
    обязательно соответствует двумерной точке points2d_2[i] - то есть им обои соответсвует один и тот же уголок
    с индексом именно ids_common_known_corners[i]"""

    # считаем ошибки при проецировании найденных 3d точек обратно на оба кадра
    error_on_1_frame = compute_reprojection_errors(points3d_for_1and2, points2d_1, intrinsic_mat, known_view3x4_1)
    error_on_2_frame = compute_reprojection_errors(points3d_for_1and2, points2d_2, intrinsic_mat, known_view3x4_2)

    points3d_for_1and2 = points3d_for_1and2[(error_on_1_frame < REPROJECTION_ERROR)
                                            & (error_on_2_frame < REPROJECTION_ERROR)]  # берём только те
    # точки, которые достаточно хорошо триангулированы - то есть достаточно правильно проецируются на оба кадра
    # (правильно - то есть с точностью до REPROJECTION_ERROR пикселей)
    ids_common_known_corners = ids_common_known_corners[(error_on_1_frame < REPROJECTION_ERROR)
                                            & (error_on_2_frame < REPROJECTION_ERROR)]  # аналогично и индексы для этих
    # уголков берем подходящие (индексы уголков на обоих кадрах - это и есть индексы points3d_for_1and2 - так как
    # эти трехмерные точки соответствуют двумерным, которые есть на обоих кадрах

    assert(len(points3d_for_1and2) > 3)  # проверяем, что достаточно много точек

    corners_id_for_3d_points = np.append(corners_id_for_3d_points, ids_common_known_corners)  # добавляем id уголков
    found_3d_points = np.append(found_3d_points, points3d_for_1and2, axis=0)  # и соответсвующие 3d точки

    print("           ... исходная триангуляция завершена")
    """------------------закончили триангуляцию------------------------------------------------------------------"""


    if known_frame_1 > known_frame_2:  # делаем так, чтобы второй изестный кадр по номеру был больше первого
        known_frame_1, known_frame_2 = known_frame_2, known_frame_1
        known_view3x4_1, known_view3x4_2 = known_view3x4_2, known_view3x4_1

    """
    На данный момент всё множество кадров (с номерами от 0 до len(corner_storage) - 1 включительно), для
    которых ещё не знаем позицию камеры, поделилось на 
    три части (границы (номера кадров) - включительно):
    [0 ... known_frame_1 - 1], [known_frame_1 + 1 ... known_frame_2 - 1], [known_frame_2 + 1 ... len(corner_storage) - 1]
    (конечно, какие-то части могут быть вырожденными, если, например, известные кадры - это первые два кадра)
    
    и далее нам нужно быдет выбирать кадр из одной из трех частей, чтобы далее решать pnp с точками этого кадра - 
    но выбирать мы будем лучший подходящий кадр - то есть тот, для которого наибольшее количество соответстующих
    2d и 3d точек
    
    у нас две области номеров кадров, для которых уже знаем положения камеры - изначально каждая из этих областей
    содержит по одному кадру: known_frame_1 и known_frame_2 соответственно; эти области и делят все наши кадры
    на три части, что уже сказали в начале этого комментария
    
    далее будем поддерживать границы этих областей:
    """
    left_lim_1 = known_frame_1  # границы первой области
    right_lim_1 = known_frame_1
    left_lim_2 = known_frame_2  # границы второй области
    right_lim_2 = known_frame_2

    num_iter = 0  # подсчитываем число итераций главного цикла
    while True:
        if left_lim_1 == 0 and right_lim_2 == len(corner_storage) - 1 \
                and right_lim_1 == left_lim_2 - 1:  # наши две области кадров с известными положениями
            # камеры покрыли все кадры -> можно выходить

            assert (len(view_mats) == len(corner_storage))  # заодно проверям, что нашли все view-матрицы
            break

        print("Шаг номер: ", num_iter, ",")
        num_iter += 1

        new_frame, left_lim_1, right_lim_1, left_lim_2, right_lim_2 = \
            choose_best_next_frame(left_lim_1, right_lim_1, left_lim_2, right_lim_2, corner_storage, corners_id_for_3d_points)
        corners_in_frame = corner_storage[new_frame]  # взяли уголки с выбранного ранее кадра

        print("Текущее облако точек имеет размер = ", len(corners_id_for_3d_points), ",")
        print("Обрабатываем кадр номер", new_frame, "...")

        # в следующих 4 строчках получаем 3d и 2d точки, соответствующие друг другу:
        # (по аналогии с тем, что уже делали в триангуляции)
        mask_common_ids_3d_in_frame = np.in1d(corners_id_for_3d_points, corners_in_frame.ids)
        mask_common_ids_frame_in_3d = np.in1d(corners_in_frame.ids, corners_id_for_3d_points)
        points3d_for_frame = found_3d_points[mask_common_ids_3d_in_frame]
        points2d_for_frame = corners_in_frame.points[mask_common_ids_frame_in_3d]

        assert (len(points2d_for_frame) == len(points3d_for_frame))
        assert (len(points2d_for_frame) >= 4)
        res, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=points3d_for_frame,
                                                      imagePoints=points2d_for_frame,
                                                      cameraMatrix=intrinsic_mat,
                                                      reprojectionError=REPROJECTION_ERROR,
                                                      distCoeffs=0)
        if not res:
            for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
                res, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=points3d_for_frame,
                                                              imagePoints=points2d_for_frame,
                                                              cameraMatrix=intrinsic_mat,
                                                              reprojectionError=REPROJECTION_ERROR * i,
                                                              distCoeffs=0)
                print("ВНИМАНИЕ: pnp на базовой ошибке репроекции = ", REPROJECTION_ERROR, " не решилась...")
                print("          Увеличили ошибку репроекции в " , i, " раз и повторили попытку...")
                if res:
                    break
            if not res:
                raise NameError("Не удалось решить pnp - вдимо, странное видео")

        print("Кадр ", new_frame, " обработан; количество инлайеров, по которым решена pnp = ", len(inliers), ",")
        print("Текущиие кадры, для которых нашли положение камеры: [", left_lim_1, " ... ",
              right_lim_1, "], [", left_lim_2, " ... ", right_lim_2, "], а всего кадров: ", len(corner_storage), ".")
        print("-------------------------------------------------------")
        print()

        new_view_camera = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)  # получили view матрицу для нового кадра
        frame_with_found_cam.append(new_frame)  # добавляем это в массив
        view_mats.append(new_view_camera)

        prev_frame = best_frame_for_triangl(new_frame, corner_storage, frame_with_found_cam)  # взяли кадр для триангуляции
        prev_corners = corner_storage[prev_frame]  # уголки в обоих кадрах
        new_corners = corner_storage[new_frame]

        mask_common_ids_prev_in_new = np.in1d(prev_corners.ids, new_corners.ids)  # как и в исходной инициализации,
        mask_common_ids_new_in_prev = np.in1d(new_corners.ids, prev_corners.ids)  # берем соотвествующие 2d точки на
        points2d_new = new_corners.points[mask_common_ids_new_in_prev]  # обоих кадрах
        points2d_prev = prev_corners.points[mask_common_ids_prev_in_new]
        prev_new_common_ids = np.intersect1d(prev_corners.ids, new_corners.ids)  # общие id - это фактически индексы для
        # new_3d_points - тех 3d точек, которые сейчас найдем

        prev_view_camera = view_mats[frame_with_found_cam.index(prev_frame)]  # шашли view-матрицу для prev кадра
        new_3d_points = triangulate_points(points2d_1=points2d_new, points2d_2=points2d_prev,
                                           camera_view_1=new_view_camera, camera_view_2=prev_view_camera,
                                           proj_mat=intrinsic_mat)  # нашли новые 3d точки по этим двум кадрам
        error_on_new_frame = compute_reprojection_errors(new_3d_points, points2d_new, intrinsic_mat, new_view_camera)
        error_on_prev_frame = compute_reprojection_errors(new_3d_points, points2d_prev, intrinsic_mat, prev_view_camera)

        new_3d_points = new_3d_points[(error_on_new_frame < REPROJECTION_ERROR)
                                                & (error_on_prev_frame < REPROJECTION_ERROR)]  # взяли точки с маленькой
        # ошибкой - все как в исходной триангуляции, а так же индексы не забыли изменить:
        prev_new_common_ids = prev_new_common_ids[(error_on_new_frame < REPROJECTION_ERROR)
                                                & (error_on_prev_frame < REPROJECTION_ERROR)]

        # добавляем новые 3d точки - но только те, которых ещё нет:
        mask_inds_not_in_found_3d_points = ~np.in1d(prev_new_common_ids, corners_id_for_3d_points)  # индексы тех
        # уголков из рассматриваемых сейчас, которых еще нет в 3d точках
        found_3d_points = np.append(found_3d_points,
                                    new_3d_points[mask_inds_not_in_found_3d_points], axis=0)
        corners_id_for_3d_points = np.append(corners_id_for_3d_points,
                                             prev_new_common_ids[mask_inds_not_in_found_3d_points])
        assert(len(set(corners_id_for_3d_points)) == len(corners_id_for_3d_points))  # прверяем, что все индексы 3d
        # точек различные


    """frame_count = len(corner_storage)
    view_mats = [pose_to_view_mat3x4(known_view_1[1])] * frame_count"""
    """corners_0 = corner_storage[0]"""
    """point_cloud_builder = PointCloudBuilder(corners_0.ids[:1],
                                            np.zeros((1, 3)))"""
    point_cloud_builder = PointCloudBuilder(np.array(corners_id_for_3d_points, dtype=int),
                                            found_3d_points)
    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()


