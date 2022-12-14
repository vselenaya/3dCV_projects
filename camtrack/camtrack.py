#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2

import tqdm
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
    triangulate_correspondences,
    eye3x4,
    view_mat3x4_to_rodrigues_and_translation
)
#from __test_camtr import create_cli
from ba import run_bundle_adjustment


def retriangulate_points(proj_mat, view_mat_sequence, points2d_sequence):
    assert (len(view_mat_sequence) == len(points2d_sequence))

    Pi_sequence = [proj_mat @ view_mat for view_mat in view_mat_sequence]
    points2d_homog = [np.hstack((point2d, 1)) for point2d in points2d_sequence]

    num = len(Pi_sequence)
    M = np.zeros((3 * num, 4 + num))
    for i, (x, Pi) in enumerate(zip(points2d_homog, Pi_sequence)):  # по аналогиии с практикой 6
        M[3 * i:3 * i + 3, :4] = Pi
        M[3 * i:3 * i + 3, 4 + i] = -x
    V = np.linalg.svd(M)[-1]
    X = V[-1, :4]
    return (X / X[3])[:3]


def choose_best_next_frame_and_solve_pnp(left_lim_1, right_lim_1, left_lim_2, right_lim_2,
                                         found_3d_points, corners_id_for_3d_points,
                                         intrinsic_mat, corner_storage, PNP_ERROR, MIN_INLIERS,
                                         frame_with_found_cam, view_mats):
    """
    Эта функция выбирает кадр, для которого следующим искать положение камеры в нем.

    Для этого мы будем кадр только среди кадров, соседних с теми, для которых уже нашли позицию камеры (ведь так больше
    шансов найти кадр с наибольши количество 2d-3d соответствий) - как мы уже говорили,
    у нас две области номеров таких кадров: [left_lim_1 ... right_lim_1] и [left_lim_2 ... right_lim_2] -
    - так что, соседние с ними кадры (их, очевидно, не больше четырёх) и перебираем.

    А далее для каждого такого кадра решаем задачу pnp - и как только она решится, мы считаем, что нашли подходящий
    кадр - его и выдаем, а заодно границы областей, для которых нашли позицию камеры, а также возвращаем результат
    pnp - параметры позиции камеры и число инлайеров - точек, по которым посчитана pnp

    Изначально мы пытаемся решить pnp с маленькой ошибкой, но если ни на одном из рассматриваемых кадрах это не
    получается сделать, то увеличиваем её...
    """
    interesting_frames = []  # номера кадров, которые можно сейчас рассмотреть - это соседние кадры с кадрами, для
    # которых уже известны положения камеры (так больше шансов найти 2d-2в соответствия)
    if left_lim_1 > 0:
        interesting_frames.append((left_lim_1 - 1, left_lim_1))
    if right_lim_2 < len(corner_storage) - 1:
        interesting_frames.append((right_lim_2 + 1, right_lim_2))
    if right_lim_1 < left_lim_2 - 2:
        interesting_frames.append((right_lim_1 + 1, right_lim_1))
        interesting_frames.append((left_lim_2 - 1, left_lim_2))
    if right_lim_1 == left_lim_2 - 2:
        interesting_frames.append((right_lim_1 + 1, right_lim_1))

    interesting_frames = sorted(interesting_frames, key=lambda pair: pair[0])

    """?????"""
    if (left_lim_1 > 0) and (right_lim_1 < left_lim_2 - 1):
        if left_lim_1 > (left_lim_2 - right_lim_1):
            interesting_frames[0], interesting_frames[1] = interesting_frames[1], interesting_frames[0]
    """-----"""

    print("Текущее облако точек имеет размер = ", len(corners_id_for_3d_points), ",")
    print("Рассматриваем кадры с номерами: ", [pair[0] for pair in interesting_frames], ",")

    best_frame = -1  # лучший кадр, который и ищем
    res = False  # пока что pnp не решили
    break_from_while = False  # пока из while выходить не нужно
    curr_coeff_er = 1  # именно на этот коэффициент домножаем ошибку решения PNP
    while True:
        for intr_frame, neighbor_to_intr_frame in interesting_frames:
            corners_in_frame = corner_storage[intr_frame]  # взяли уголки с выбранного кадра

            print("Обрабатываем кадр номер", intr_frame, "...")
            print("                 попытка решить pnp c ошибкой (в пикселях) = ", PNP_ERROR * curr_coeff_er)

            # в следующих 4 строчках получаем 3d и 2d точки, соответствующие друг другу:
            # (по аналогии с тем, что уже делали в триангуляции)
            common_frame_and_3d = np.intersect1d(corners_id_for_3d_points, corners_in_frame.ids)
            mask_common_ids_3d_in_frame = np.in1d(corners_id_for_3d_points, corners_in_frame.ids)
            mask_common_ids_frame_in_3d = np.in1d(corners_in_frame.ids, corners_id_for_3d_points)
            points3d_for_frame = found_3d_points[mask_common_ids_3d_in_frame]
            points2d_for_frame = corners_in_frame.points[mask_common_ids_frame_in_3d]

            if not len(points2d_for_frame) >= 4:  # если не нашли хотя бы 4 точки, то pnp точно не решить - идем дальше
                continue
            assert (len(points2d_for_frame) == len(points3d_for_frame))

            neighbor_view = view_mats[frame_with_found_cam.index(neighbor_to_intr_frame)]
            neighbor_rvec, neighbor_tvec = view_mat3x4_to_rodrigues_and_translation(neighbor_view)
            res, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=points3d_for_frame,
                                                          imagePoints=points2d_for_frame,
                                                          cameraMatrix=intrinsic_mat,
                                                          reprojectionError=PNP_ERROR * curr_coeff_er,
                                                          distCoeffs=None,
                                                          iterationsCount=3000,
                                                          useExtrinsicGuess=True,
                                                          rvec=neighbor_rvec.copy(),  # начальные значения, чтобы проще решить pnp
                                                          tvec=neighbor_tvec.copy())  # решаем pnp
            tvec = tvec.reshape(-1, 1)

            if inliers is not None:
                current_outliers = np.delete(common_frame_and_3d, inliers.flatten())  # удаляем из всех соответствий
                # инлайеры -> остаются аутлайеры для текущего решения pnp

            if res:  # если решили:
                best_frame = intr_frame  # фиксируем кадр
                if (len(inliers) >= MIN_INLIERS) or (PNP_ERROR * curr_coeff_er > 5):  # если достаточно инлаеров,
                    # по которым решили pnp или ошибка уже достаточно большая, что нам уже не важно, сколько
                    # инлаеров - решить бы хоть как-то, то просто выходим из всех циклов и выдаем затем результат
                    break_from_while = True    # если же нет, то крутимся в циклах, пытаясь решить далее
                    break

        if break_from_while: break

        if PNP_ERROR * curr_coeff_er > 20:  # если ошибка уже очень большая, то выходим - что-то не так с видео
            break

        curr_coeff_er += 0.5  # если ранее не вышли из цикла, то так и не решили задачу... - продолжаем с большим коэфф
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
    return best_frame, new_left_lim_1, new_right_lim_1, new_left_lim_2, new_right_lim_2, \
           rvec, tvec, inliers, current_outliers


def get_initial_frames(corner_storage, intrinsic_mat):
    print("Начинаем инициализацию (поиск положений камеры на двух кадрах)...")

    best_frames = None
    best_R_t = None
    best_inliers = 0

    frames_all = len(corner_storage)

    for i in tqdm.tqdm(range(0, frames_all // 3, 5)):
        for j in range(i + 5, frames_all, 5):
            correspondences = build_correspondences(corner_storage[i], corner_storage[j])
            if len(correspondences.ids) < 200:
                continue
            homography_mat, mask_homography = cv2.findHomography(correspondences.points_1, correspondences.points_2,
                                                                 cv2.RANSAC, 4)
            essential_mat, mask_essential = cv2.findEssentialMat(correspondences.points_1,
                                                                 correspondences.points_2,
                                                                 intrinsic_mat, cv2.RANSAC, 0.999, 1)
            #if mask_essential.sum() / mask_homography.sum() < 0.8:
                #continue
            # inliers_idx = mask_essential.flatten()[(mask_essential.flatten() == 1) & (mask_homography.flatten() == 0)]
            inliers_idx = np.arange(len(mask_essential))[(mask_essential.flatten() == 1) & (mask_homography.flatten() == 0)]

            if len(inliers_idx) < 9:
                continue
            retval, R, t, mask = cv2.recoverPose(essential_mat, correspondences.points_1[inliers_idx],
                                                 correspondences.points_2[inliers_idx], intrinsic_mat)

            # print(retval)
            if best_inliers < retval / len(correspondences.ids):
                best_inliers = retval / len(correspondences.ids)
                best_frames = (i, j)
                best_R_t = (R, t)

    assert (best_R_t is not None)
    
    first_frame, second_frame = best_frames
    R, t = best_R_t
    """
    i, j = 140, 175
    first_frame, second_frame = i, j
    correspondences = build_correspondences(corner_storage[i], corner_storage[j])

    homography_mat, mask_homography = cv2.findHomography(correspondences.points_1, correspondences.points_2,
                                                         cv2.RANSAC, 4)
    essential_mat, mask_essential = cv2.findEssentialMat(correspondences.points_1,
                                                         correspondences.points_2,
                                                         intrinsic_mat, cv2.RANSAC, 0.999, 1)

    inliers_idx = np.arange(len(mask_essential))[(mask_essential.flatten() == 1) & (mask_homography.flatten() == 0)]

    retval, R, t, mask = cv2.recoverPose(essential_mat, correspondences.points_1[inliers_idx],
                                         correspondences.points_2[inliers_idx], intrinsic_mat)
    """
    pose_first = view_mat3x4_to_pose(eye3x4())
    pose_second = Pose(R.T, R.T @ -t)

    print("Инициализация завершена - выбраны кадры: ", first_frame, second_frame)
    print("----------------------------------------------------------------------------------------------------")

    return (first_frame, pose_first), (second_frame, pose_second)


REPROJECTION_ERROR = 0.5  # ошибка репроекции при получении 3d-3d соответствий
MIN_TRIANGULATION_ANGLE = 2  # минимальный угол триангуляции
MIN_DEPTH = 0
PNP_ERROR = 3  # ошибка репроекции при решении задачи pnp
MIN_INLIERS = 20  # минимальное количество инлайеров, которое нас устраивает для решения pnp
MAX_RETRIANGL = 25  # максимальное количество точек, которое ретриангулируем


def track_and_calc_colors(camera_parameters: CameraParameters,  # параметры камеры
                          corner_storage: CornerStorage,  # откуда уголки брать (можно обращаться по индексу -
                          frame_sequence_path: str,                         # - номеру кадра (нумерация с 0))
                          known_view_1: Optional[Tuple[int, Pose]] = None,  # известные положения камеры в двух кадрах
                          #                             ^    ^
                          #                             |    позиция камеры (которые можно перевести во view-матрицу)
                          #                             номер кадра, где задана позиция
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:  # возвращаем список позиций камеры (индексирован номерами кадров) и 3d точки

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(  # матрица внутренних параметров камеры (где фокусное расст и центр камеры)
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = get_initial_frames(corner_storage=corner_storage, intrinsic_mat=intrinsic_mat)
        # raise NotImplementedError()

    # TODO: implement
    """
    Сразу заводим структуру found_3d_points класса PointCloudBuilder(), в котором будем хранить найденные 3d точки
    (они хранятся вместе с id (то есть номерами) тех уголков, для которых найдены 3d точки (то есть те 3d точки, 
    которые проецируется в эти уголки (ведь уголки с одним и тем же id - это фактически изображение одних и тех же
    3d точек (и они могут быть на разных кадрах))
    """
    found_3d_points = PointCloudBuilder()

    """
    А также еще два массива:
    frame_with_found_cam - номера кадров, где уже нашли положение камеры
    view_mats - эти самые найденные положения камеры    
    """
    frame_with_found_cam = []
    view_mats = []

    """
    И еще один массив заводим, в котором будут храниться инлексы тех уголков, которые считаются выбросами (аутлайерами),
    то есть которые не использовались в решении задачи pnp
    """
    IDS_OUTLIERS = np.array([], dtype=np.int64)

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

    triang_params = TriangulationParameters(max_reprojection_error=5,  # параметры для первичной триангуляции задаем
                                            min_triangulation_angle_deg=0,  # самые мягкие, потому что уж первчиная
                                            min_depth=0)  # триангуляция точно должна пройти

    correspondences_known_1_2 = build_correspondences(known_1_corners, known_2_corners)  # строим пересечение уголков
    # на двух кадрах - те ищем одни и те же уголки (у них один и тот же id) на двух кадрах - это наши двумерные
    # соответствия - далее по этим двумерным соответствиям восстанавливаем 3d точки, проецирующиеся в эти уголки
    assert (len(correspondences_known_1_2.ids) > 3)  # проверяем, что у нас достаточно много двумерных соответствий

    points3d_for_1and2, ids_common_known_corners, median_cos = \
        triangulate_correspondences(correspondences=correspondences_known_1_2,
                                    view_mat_1=known_view3x4_1, view_mat_2=known_view3x4_2,
                                    intrinsic_mat=intrinsic_mat, parameters=triang_params)  # по двумерным точкам
    # восстанавливаем 3d точки (points3d_for_1and2),  также получаем инлексы (id) (ids_common_known_corners)
    # для этих точек - эти id- это id уголков, в которые эти 3d-точки проецируются
    assert (len(ids_common_known_corners) > 3)  # проверяем, что у нас достаточно много 3d точек

    found_3d_points.add_points(ids=ids_common_known_corners, points=points3d_for_1and2)  # добавляем точки

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

        new_frame, left_lim_1, right_lim_1, left_lim_2, right_lim_2, rvec, tvec, inliers, curr_outliers = \
            choose_best_next_frame_and_solve_pnp(left_lim_1, right_lim_1, left_lim_2, right_lim_2,
                                                 found_3d_points.points, found_3d_points.ids,
                                                 intrinsic_mat, corner_storage,
                                                 PNP_ERROR, MIN_INLIERS,
                                                 frame_with_found_cam, view_mats)  # ищем лучший кадр, считаем
        # для него положения камеры и заодно двигаем границу областей кадров, для которых нашли положения камер ->
        # -> в результате нашли положение камеры для нового кадра

        assert (left_lim_1 <= right_lim_1 < left_lim_2 <= right_lim_2)  # проверяем корректность наших границ

        IDS_OUTLIERS = np.append(IDS_OUTLIERS, curr_outliers)  # добавляем индексы аутлайеров (-тех точек, на которых
        # не решалась задача pnp - те они возможно выбросы)
        IDS_OUTLIERS.sort()  # обязательно сортируем, так как далее мы используем этот массив в build_correspondences,
        # а внутри этой функции есть snp.intersect, который работает правильно только с отсортированными массивами!
        found_3d_points.delete_points(IDS_OUTLIERS)  # удаляем аутлайеры из 3d-точек, чтобы они не мешали дальше

        print("Кадр ", new_frame, " обработан; количество инлайеров, по которым решена pnp = ", len(inliers), ",")
        print("Текущиие кадры, для которых нашли положение камеры: [", left_lim_1, " ... ",
              right_lim_1, "], [", left_lim_2, " ... ", right_lim_2, "], а всего кадров: ", len(corner_storage), ".")
        print("-------------------------------------------------------")
        print()
        #print(rvec, tvec)
        """-----------теперь попытаемся дотриангулировать еще 3d точек-------------------"""
        new_view_camera = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)  # получили view матрицу для нового кадра
        new_corners = corner_storage[new_frame]  # и уголки

        for prev_frame, prev_view_camera in zip(frame_with_found_cam[::3], view_mats[::3]):  # перебираем все
            # предыдущие кадры,для которых уже извстна view-матрица положения камеры
            # (и сами матрицы тоже перебираем)  -- но так как прямо все кадры перебирать долго, перебираем, например,
            # с шагом 3
            prev_corners = corner_storage[prev_frame]  # уголки в предыдущем кадре

            triang_params = TriangulationParameters(max_reprojection_error=REPROJECTION_ERROR,
                                                    min_triangulation_angle_deg=MIN_TRIANGULATION_ANGLE,
                                                    min_depth=MIN_DEPTH)  # задаем параметры триангуляции
            correspondences_prev_new = build_correspondences(prev_corners, new_corners,
                                                             ids_to_remove=IDS_OUTLIERS)  # снова находим общие уголки
            # на двух кадраха - те 2d-соответствия на этих кадрах

            if len(correspondences_prev_new.ids) < 4: continue  # если их слишком мало - переходим к следующему кадру
            new_3d_points, prev_new_common_ids, median_cos = \
                triangulate_correspondences(correspondences=correspondences_prev_new,
                                            view_mat_1=prev_view_camera, view_mat_2=new_view_camera,
                                            intrinsic_mat=intrinsic_mat, parameters=triang_params)  # получаем 3d-точки,
            # и индексы (id) уголков

            if len(prev_new_common_ids) > 0:  # если нашли больше нуля точек:
                # добавляем новые 3d точки - но только те, которых ещё нет:
                # (то, что добавятся только новые 3d точки, гарантируется самим методо add_points - он добавляет только
                # еще несуществующие 3d-точки):
                found_3d_points.add_points(ids=prev_new_common_ids, points=new_3d_points)

        frame_with_found_cam.append(new_frame)  # добавляем новую камеру это в массив
        view_mats.append(new_view_camera)

        """_________а теперь - ретириангуляция, то есть триангулируем точки по нескольким кадрам"""

        if num_iter % 10 == 0 and num_iter > 10:  # ретриангулируем каждые 10 кадров (раз в 10 кадров)
            count_retriang = 0
            for known_3d_point in found_3d_points.ids[-MAX_RETRIANGL:]:  # каждые 10 точек ретириангулируем
                if count_retriang == MAX_RETRIANGL: break
                points2d_for_this_3d_point = []  # 2d-точки, соответствующие взятой 3d-точке
                view_mats_for_frames_with_this_3d_point = []  # view-матрицы для кадров, в которых эта 3d-точка видна
                for frame, view_m in zip(frame_with_found_cam, view_mats):  # перебираем кадры с известными view-матрицами
                    if known_3d_point in corner_storage[frame].ids:  # если 3d-точка есть на кадре
                        index_3d_point = list(corner_storage[frame].ids).index(known_3d_point)
                        points2d_for_this_3d_point.append(corner_storage[frame].points[index_3d_point])
                        view_mats_for_frames_with_this_3d_point.append(view_m)

                if len(view_mats_for_frames_with_this_3d_point) < 3:  # если маловато кадров - идем дальше
                    continue
                else:
                    new3d = retriangulate_points(proj_mat=intrinsic_mat,
                                                 view_mat_sequence=view_mats_for_frames_with_this_3d_point,
                                                 points2d_sequence=points2d_for_this_3d_point)  # получаем 3d-точку
                    found_3d_points.update_points(ids=np.array([known_3d_point]), points=new3d.reshape(1, 3))  # обновляем
                count_retriang += 1
        """
        if num_iter % 10 == 0 and num_iter > 40:
            # print('Frame {}: new points {}, total {}'.format(frame, delta, builder.points.shape[0]))
            view_mats[-40:] = run_bundle_adjustment(
                intrinsic_mat=intrinsic_mat,
                list_of_corners=[corner_storage[i] for i in frame_with_found_cam[-40:]],
                max_inlier_reprojection_error=REPROJECTION_ERROR,
                views=view_mats[-40:],
                pc_builder=found_3d_points)

        num_iter += 1
        """
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

    """
    ОЧЕНЬ ВАЖНО! 
    во view_mats мы добавляли матрицы камер в том порядке, в котором мы их находили (а могли найти сначла камеру
    для 1 кадра, затем для десятого...), а вернуть нам нужно матрицы в том же порядке, в котором у нас идут кадры - 
    те сначала должна идти view-матрица (которую мы в pose переделываем) для 0-го кадра, затем для 1-ого и тд
    
    Так как параллельно с добавлением матриц во view-mats мы еще и добавляли номера кадров, для которых нашли
    эту view-матрицу, то просто отсортируем view-mats по frame_with_found_cam, в котором эти номера кадров и хранятся
    (то есть переставим элементы view_mats так, чтобы соответствующие номера кадров во frame_with_found_cam встали
    по порядку) - это и делаем далее:
    
    (этой ошибки долго не замечал, из-за этого все время были большие ошибки, так как выводимые pose камер не соответ-
    ствовали кадру)
    """
    temp = sorted(zip(frame_with_found_cam, view_mats), key=lambda x: x[0])
    frame_with_found_cam = [vm[0] for vm in temp]
    view_mats = [vm[1] for vm in temp]

    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()


