#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    # TODO
    """
    общие наблюдения:
    0) смысл этой функции состоит в том, чтобы на каждом изображении последовательности frame_sequence (эта
    последовательность получается как кадры видео или еще как-то) найти уголки и передать
    их в функцию builder вместе с изображением (это мы как бы отрисовали (или просто установили) уголки на изображении);
    так же нам нужно отслеживать потоки - те отслеживать одни и те же уголки, которые переходят из кадра в кадр -
    это мы как-то делаем (вообще есть две функции goodFeaturesToTrack - она по изображению выдаёт массив координат
    уголков и calcOpticalFlowPyrLK - она по предыдущему и следующему изображению (причем они оба должны быть в
    формате np.uint8 - те каждый цвет пикселя - от 0 до 255), а также массиву уголков предыдущего изображения,
    выдает массив координат уголков на следующем изображении, которые получилсиь перемещением уголков предыдущего
    изображения (причем еще есть массив status, который говорит, смогли ли мы отследить уголок старого изображения
    на новом или нет)), а чтобы показать в этом коде, что на двух разных кадрах уголок один и то же, у каждого уголка
    есть id (в функцию builder мы подаём не совсем массив координат уголков, а элемент класса, в котором указаны
    сами координаты уголков, а так же их id (это поле ids)) - это любое число - и если у двух уголков одинаковый id,
    то это один и тот же уголок - так они и отслеживаются.

    1) функция goodFeaturesToTrack, если не смогла найти ни одного уголка (если полностью черный кадр или слишком
    высокий QUALITY_LEVEL или подставили сложную маску (mask) - такую, что в тех областях, где маска просит искать
    уголки, их нет), вернёт None - нужно с этим аккуратнее

    2) функции goodFeaturesToTrack и calcOpticalFlowPyrLK, которые должны выдавать массив координат уголков,
    возвращают массив немного в странном виде - а именно в форме (n, 1, 2) - те у нас есть массив некоторой высоты n
    (n - это количество уголков), далее в каждой его строке записан массив из одного элемента - массива из двух
    элементов - координат уголков... то есть немного непонятно, зачем делать массив из массива из двух элементов,
    поэтому иногда тот массив, что возвращают эти функции, лучше сделать .reshape(-1, 2) - тогда он станет высоты n
    (параметр -1 в reshape автоматически подбирает нужный размер по этой коорд) и ширины 2 - те будет двумерный массив,
    где в каждой строке стоит пара из двух чисел - коодинат уголков
    аналогично с массивом status, который вместе с массивом уголков, возвращает функция
    calcOpticalFlowPyrLK - суть этого массива status в том, что по факту это вектор длины n (те его длтна равна длине
    массива с уголками, который мы подали в функцию calcOpticalFlowPyrLK для отслеживания смещения уголков (и
    соответственно равна длине массивы со смещенными уголками, который нам выдаёт эта функция)) -
    - в этом векторе стоит 1, если новый уголок действительно является отслеженным уголоком с предыдущего изображения
    (те найден поток) и 0 иначе (если астероид взорвался, то мы не можем отследить старые уголки на новой
    картинке после взрыва); но на самом деле это не совсем вектор, а двумерный массив формы (n, 1), где в каждой строке
    стоит один элемент - 0 или 1 - не очень понятно, зачем было делать двумерный массив, почему нельзя просто вектор...-
    -поэтому иногда можно будет сделать .reshape(-1), чтобы превратить status в обычный вектор

    3) у функции goodFeaturesToTrack есть параметр mask - в него можно подавать numpy массив такой же формы, как и
    изображение, на котором ищем уголки, но в этом массиве mask имеют значения два вида элементов - нулевые и ненулевые-
    -если элемент в mask нулевой, это значит, что на соответсвтующем пикселе изображения НЕ нужно искать уголок, а если
    ненулевой элемент - то можно искать; так мы можем указывать области поиска уголков.
    """

    MAX_CORNERS = 5000  # максимальное количество уголков на изображении
    MIN_DISTANCE = 5  # минимальное расстояние в пикселях между уголками, которые ищем
    QUALITY_LEVEL = 0.005  # качество уголков - чем меньше, тем больше уголков найдём, но хуже качество будет
    BLOCK_SIZE = 7
    DRAW_SIZE = 3 * MIN_DISTANCE  # радиус для рисования уголка

    # params for ShiTomasi corner detection
    features = dict(maxCorners=MAX_CORNERS,
                    qualityLevel=QUALITY_LEVEL,
                    minDistance=MIN_DISTANCE,
                    blockSize=BLOCK_SIZE)

    # Parameters for lucas kanade optical flow
    lks = dict(winSize=(15, 15),
               maxLevel=2,
               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # изображение из последовательности - это изображение в виде numpy массив,
    # где цвет пикселя - это float значение от 0 до 1:
    image_0 = frame_sequence[0]  # первый кадр последовательности

    corners = cv2.goodFeaturesToTrack(image=image_0, mask=None,
                                      maxCorners=MAX_CORNERS,
                                      qualityLevel=QUALITY_LEVEL / 3,
                                      minDistance=MIN_DISTANCE,
                                      blockSize=BLOCK_SIZE)  # нашли уголки в нём
    if corners is None:  # если не нашли уголок, укажем какую-то точку на всякий случай
        corners = np.array([[[1, 1]]])
    n = len(corners)
    ids = np.arange(n, dtype=np.int64)  # индексы расставим так (как оказалось, очень важен тип именно int64
                                        # для дальнейших функций по отслеживанию камеры)
    prev_max_ids = ids.max()  # зафиксируем максимальный индекс, который сейчас есть у уголков (просто максимальный
    # индекс, котрый использовался для нумерации уголков)

    frame_corners = FrameCorners(ids=ids,
                                 points=corners,
                                 sizes=np.full(n, DRAW_SIZE))
    builder.set_corners_at_frame(0, frame_corners)  # передали номер кадра и уголки на нём

    for frame, image_1 in enumerate(frame_sequence[1:], 1):  # теперь проходимя по всем кадам
        prev_img = np.uint8(image_0 * 255)  # переводим изображения в формат, когда цвет пикселя - от 0 до 255
        next_img = np.uint8(image_1 * 255)

        # получаем положения уголков (new_corners) на новой картинке по уже найденным уголкам предыдущей картинкия:
        new_corners, status, _ = cv2.calcOpticalFlowPyrLK(prev_img, next_img, corners, None)

        # из всех новых уголков берём только корректно отслеженные - это и будут отслеженные уголки нового кадра:
        ids = ids[status.reshape(-1) == 1]  # id уголков тоже берем, чтобы показать, что это именно отслеженные уголки
        corners = new_corners[status.reshape(-1) == 1]
        n = len(corners)  # количество уголков на новом кадре

        if n < MAX_CORNERS:  # если на новом кадре мало уголков, пробуем найти ещё
            mask = np.full(image_1.shape, 255, dtype=np.uint8)  # создаём массив такой же по размеру как изображение
            for x, y in corners.reshape(-1, 2):
                # рисуем в массиве mask вокруг уголков круг из нулей радиусом = минимальная дистанция между
                # уголками (это нужно, чтобы далее на нашем новом изображении искать уголки не рядом с уже
                # существующими, а на удалении >= как раз это минимальное расстояние между уголками):
                cv2.circle(mask, (int(x), int(y)), MIN_DISTANCE, 0, -1)

            # исполльзуем нашу mask, полученную только что, чтобы искать уголки подальше от уже найденных уголков:
            candidates_corners = cv2.goodFeaturesToTrack(image_1, mask=mask, **features)

            if candidates_corners is not None:  # если новых уголки нашлись, то добавляем их (но не больше, чем нужно)
                corners = np.concatenate([corners, candidates_corners[:MAX_CORNERS - n]])
                n = len(corners)
                # для новых уголков догенерируем id, но только начиная с prev_max_ids + 1, чтобы у новых уголков
                # были другие id, чем у старых:
                ids = np.concatenate([ids, np.arange(prev_max_ids + 1, n - len(ids) + prev_max_ids + 1)])

        prev_max_ids = max(prev_max_ids, ids.max())  # обновляем максимум (берём максимум от старого максимума и
        # текущего максимального индекса, так как если новых уголков не задетектилось, то максимальный использованный
        # индекс должен остаться прежним, а вот текущий максимум может уменьшиться (если какие-то треки со старыми
        # уголками прервались) - и тогда если написать просто prev_max_ids = ids.max(), то это уменьшит максимум, что
        # плохо, ведь новые уголки потом будем обозначать старыми индексами уже исчезнувших с экрана уголков)
        frame_corners = FrameCorners(
            ids=ids,
            points=corners,
            sizes=np.full(n, DRAW_SIZE),
        )
        builder.set_corners_at_frame(frame, frame_corners)  # отрисовываем уголки
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
