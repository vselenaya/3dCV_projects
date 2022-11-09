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
    rodrigues_and_translation_to_view_mat3x4,
    TriangulationParameters,
    build_correspondences,
    triangulate_correspondences
)
#from __test_camtr import create_cli
from ba import run_bundle_adjustment


def choose_best_next_frame_and_solve_pnp(left_lim_1, right_lim_1, left_lim_2, right_lim_2,
                                         found_3d_points, corners_id_for_3d_points,
                                         intrinsic_mat, corner_storage, PNP_ERROR):
    """
    Эта функция выбирает кадр, для которого следующим искать положение камеры в нем.

    Для этого мы будем кадр только среди кадров, соседних с теми, для которых уже нашли позицию камеры (ведь так больше
    шансов найти кадр с наибольши количество 2d-3d соответствий) - как мы уже говорили,
    у нас две области номеров таких кадров: [left_lim_1 ... right_lim_1] и [left_lim_2 ... right_lim_2] -
    - так что, соседние с ними кадры и перебираем.

    А далее для каждого такого кадра решаем задачу pnp - и как только она решится, мы считаем, что нашли подходящий
    кадр - его и выдаем, а заодно границы областей, для которых нашли позицию камеры, а также возвращаем результат
    pnp - параметры позиции камеры и число инлайеров - точек, по которым посчитана pnp

    Изначально мы пытаемся решить pnp с маленькой ошибкой, но если ни на одном из рассматриваемых кадрах это не
    получается сделать, то увеличиваем её...
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

    print("Текущее облако точек имеет размер = ", len(corners_id_for_3d_points), ",")
    print("Рассматриваем кадры с номерами: ", interesting_frames, ",")

    best_frame = -1  # лучший кадр, который и ищем
    res = False  # пока что pnp не решили
    break_from_while = False  # пока из while выходить не нужно
    curr_coeff_er = 1  # именно на этот коэффициент домножаем ошибку решения PNP
    while True:
        for intr_frame in interesting_frames:
            corners_in_frame = corner_storage[intr_frame]  # взяли уголки с выбранного кадра

            print("Обрабатываем кадр номер", intr_frame, "...")
            print("                 попытка решить pnp c ошибкой (в пикселях) = ", PNP_ERROR * curr_coeff_er)

            # в следующих 4 строчках получаем 3d и 2d точки, соответствующие друг другу:
            # (по аналогии с тем, что уже делали в триангуляции)
            mask_common_ids_3d_in_frame = np.in1d(corners_id_for_3d_points, corners_in_frame.ids)
            mask_common_ids_frame_in_3d = np.in1d(corners_in_frame.ids, corners_id_for_3d_points)
            points3d_for_frame = found_3d_points[mask_common_ids_3d_in_frame]
            points2d_for_frame = corners_in_frame.points[mask_common_ids_frame_in_3d]
            if not len(points2d_for_frame) >= 4:  # если не нашли хотя бы 4 точки, то pnp точно не решить - идем дальше
                continue
            assert (len(points2d_for_frame) == len(points3d_for_frame))

            res, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=points3d_for_frame,
                                                          imagePoints=points2d_for_frame,
                                                          cameraMatrix=intrinsic_mat,
                                                          reprojectionError=PNP_ERROR * curr_coeff_er,
                                                          distCoeffs=None,
                                                          iterationsCount=500)  # решаем pnp
            if res:  # если решили:
                best_frame = intr_frame
                if len(inliers) > MIN_INLIERS:
                    break_from_while = True
                    break

        if break_from_while: break

        if PNP_ERROR * curr_coeff_er > 20:  # если ошибка уже очень большая, то выходим - что-то не так с видео
            break

        curr_coeff_er += 0.5
        print("ВНИМАНИЕ: pnp на базовой ошибке репроекции = ", PNP_ERROR,
              " не решилась ни для какого кадра...")
        print("          Увеличили ошибку репроекции в ", curr_coeff_er, "раз и повторили попытку...")

    if not res:  # если из while вышли, но не решили pnp - выводим ошибку
        raise NameError("Не удалось решить pnp - вдимо, странное видео")

    new_left_lim_1, new_right_lim_1, new_left_lim_2, new_right_lim_2 =\
        left_lim_1, right_lim_1, left_lim_2, right_lim_2
    if best_frame == left_lim_1 - 1:  # двигаем нужную границу найденных позиций камеры
        new_left_lim_1 = best_frame
    elif best_frame == right_lim_1 + 1:
        new_right_lim_1 = best_frame
    elif best_frame == left_lim_2 - 1 and right_lim_1 < left_lim_2 - 2:
        new_left_lim_2 = best_frame
    elif best_frame == right_lim_2 + 1:
        new_right_lim_2 = best_frame
    return best_frame, new_left_lim_1, new_right_lim_1, new_left_lim_2, new_right_lim_2, rvec, tvec, inliers


REPROJECTION_ERROR = 1
MIN_TRIANGULATION_ANGLE = 1
MIN_DEPTH = 0
PNP_ERROR = 1
MIN_INLIERS = 20


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
    # corners_id_for_3d_points =
    found_3d_points = PointCloudBuilder()

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
    print("Начало исходной триангуляции...")
    known_frame_1 = known_view_1[0]  # номер первого кадра, для которого известно положения камеры
    known_frame_2 = known_view_2[0]  # номер второго кадра
    known_view3x4_1 = pose_to_view_mat3x4(known_view_1[1])  # известные view-матрицы позиций камеры в обоих кадрах
    known_view3x4_2 = pose_to_view_mat3x4(known_view_2[1])           # (то есть это матрицы внешних мараметров камеры)

    if known_frame_1 > known_frame_2:  # делаем так, чтобы второй изестный кадр по номеру был больше первого
        known_frame_1, known_frame_2 = known_frame_2, known_frame_1
        known_view3x4_1, known_view3x4_2 = known_view3x4_2, known_view3x4_1

    frame_with_found_cam.append(known_frame_1)  # добавляем то, что известно в массивы
    frame_with_found_cam.append(known_frame_2)
    view_mats.append(known_view3x4_1)
    view_mats.append(known_view3x4_2)

    known_1_corners = corner_storage[known_frame_1]  # уголки в кадрах, для которыз позиции камеры известны
    known_2_corners = corner_storage[known_frame_2]            # (имеют тип FrameCorners)

    triang_params = TriangulationParameters(max_reprojection_error=5,
                                            min_triangulation_angle_deg=0,
                                            min_depth=0)

    correspondences_known_1_2 = build_correspondences(known_1_corners, known_2_corners)
    assert (len(correspondences_known_1_2.ids) > 3)

    points3d_for_1and2, ids_common_known_corners, median_cos = \
        triangulate_correspondences(correspondences=correspondences_known_1_2,
                                    view_mat_1=known_view3x4_1, view_mat_2=known_view3x4_2,
                                    intrinsic_mat=intrinsic_mat, parameters=triang_params)
    assert (len(ids_common_known_corners) > 3)  # проверяем, что у нас достаточно много двумерных соответствий

    found_3d_points.add_points(ids=ids_common_known_corners, points=points3d_for_1and2)  # добавляем id уголков

    print("           ... исходная триангуляция завершена")
    print()
    """------------------закончили триангуляцию------------------------------------------------------------------"""

    """--------------начинаем основной цикл решения 2d-3d соответствий-------------------------------------------"""

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

        new_frame, left_lim_1, right_lim_1, left_lim_2, right_lim_2, rvec, tvec, inliers = \
            choose_best_next_frame_and_solve_pnp(left_lim_1, right_lim_1, left_lim_2, right_lim_2,
                                                 found_3d_points.points, found_3d_points.ids,
                                                 intrinsic_mat, corner_storage, PNP_ERROR)
        assert (left_lim_1 <= right_lim_1 < left_lim_2 <= right_lim_2)

        print("Кадр ", new_frame, " обработан; количество инлайеров, по которым решена pnp = ", len(inliers), ",")
        print("Текущиие кадры, для которых нашли положение камеры: [", left_lim_1, " ... ",
              right_lim_1, "], [", left_lim_2, " ... ", right_lim_2, "], а всего кадров: ", len(corner_storage), ".")
        print("-------------------------------------------------------")
        print()

        """-----------теперь попытаемся дотриангулировать еще 3d точек-------------------"""
        new_view_camera = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)  # получили view матрицу для нового кадра
        new_corners = corner_storage[new_frame]

        for prev_frame, prev_view_camera in zip(frame_with_found_cam, view_mats):
            prev_corners = corner_storage[prev_frame]  # уголки в обоих кадрах

            triang_params = TriangulationParameters(max_reprojection_error=REPROJECTION_ERROR,
                                                    min_triangulation_angle_deg=MIN_TRIANGULATION_ANGLE,
                                                    min_depth=MIN_DEPTH)
            correspondences_prev_new = build_correspondences(prev_corners, new_corners)
            if len(correspondences_prev_new.ids) < 4: continue
            new_3d_points, prev_new_common_ids, median_cos = \
                triangulate_correspondences(correspondences=correspondences_prev_new,
                                            view_mat_1=prev_view_camera, view_mat_2=new_view_camera,
                                            intrinsic_mat=intrinsic_mat, parameters=triang_params)

            if len(prev_new_common_ids) > 0:
                # добавляем новые 3d точки - но только те, которых ещё нет:
                found_3d_points.add_points(ids=prev_new_common_ids, points=new_3d_points)

        frame_with_found_cam.append(new_frame)  # добавляем это в массив
        view_mats.append(new_view_camera)


        """
        if num_iter % 10 == 0 and num_iter > 20:
            # print('Frame {}: new points {}, total {}'.format(frame, delta, builder.points.shape[0]))
            view_mats[-20:] = run_bundle_adjustment(
                intrinsic_mat=intrinsic_mat,
                list_of_corners=[corner_storage[i] for i in frame_with_found_cam[-20:]],
                max_inlier_reprojection_error=REPROJECTION_ERROR,
                views=view_mats[-20:],
                pc_builder=found_3d_points)
        """

        num_iter += 1



    """frame_count = len(corner_storage)
    view_mats = [pose_to_view_mat3x4(known_view_1[1])] * frame_count"""
    """corners_0 = corner_storage[0]"""
    """point_cloud_builder = PointCloudBuilder(corners_0.ids[:1],
                                            np.zeros((1, 3)))"""
    point_cloud_builder = found_3d_points
    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()

    temp = sorted(zip(frame_with_found_cam, view_mats), key=lambda x: x[0])
    frame_with_found_cam = [vm[0] for vm in temp]
    view_mats = [vm[1] for vm in temp]

    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()


