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

    MAX_CORNERS = 5000  # максимальное количество уголков на изображении -> берем побольше, чтобы проще отслеживать камеру
    MIN_DISTANCE = 10  # минимальное расстояние в пикселях между уголками, которые ищем (берем больше, чтобы уголки не кучковались на изображении, а более равномерно его покрывали)
    QUALITY_LEVEL = 0.003  # качество уголков - чем меньше, тем больше уголков найдём, но хуже качество будет -> я беру
    # очень маленькое, чтобы получить очень много уголков, а уже потом в отслеживании камеры (camtrack.py) выбросы
    # (плохие уголки будут отсеяны)
    BLOCK_SIZE = 21  # размер блока внутри изображения, по которому считаются частные производные для нахождения уголков
    # (так как тут нет поиска уголков с помощью пирамид (я так и не реализовал этот способ поиска), то берем побольше ->
    # -> чем больше размер этого блока, тем более большие детали изображения сможем задетектировать как уголок, но
    # качество может быть хуже (смысл этого блока - как говорилось в лекции: мы смотрим на часть изображения, попавшее
    # в этот блок - далее можем немного подвигать этот блок на изображении (во всех направлениях) - если "уголковость"
    # того, что оказывается под блоком меняется достаточно сильно, то центр этого блока мы считаем уголком (оценивается
    # "уголковость", как говорилось в лекциях: сначала считаются частные производные по блоку, в котором мы ищем
    # уголок (как раз тому блоку, размер которого сейчас задаем) - а потом в качестве "уголковости" берётся наименьшее
    # собственное число матрицы... - см лекцию) - соответственно, если блок очень большой, то уголок будет не точно
    # найден, но зато проще отследить "уголковость" и найти уголок
    # P.S. на лекциях вместо слова "блок" использовалось слово "окно"
    DRAW_SIZE = 3 * MIN_DISTANCE  # радиус для рисования уголка в визуализации - на сам поиск уголков не влияет

    # params for ShiTomasi corner detection - запихиваем параметры для детектирования уголков (алгоритмом Ши-Томаси) в словарь
    features = dict(maxCorners=MAX_CORNERS,
                    qualityLevel=QUALITY_LEVEL,
                    minDistance=MIN_DISTANCE,
                    blockSize=BLOCK_SIZE)

    # Parameters for lucas kanade optical flow - параметры для отслеживания оптического потока (оптический поток одного
    # уголка, отсеженный в течении нескольких кадров - это и есть трек уголка, который нам нужен (то есть фактичсеки
    # трек уголка - это просто положения (координаты на изображении) одного и того же уголка ("одного и того же
    # уголка" - имеется в виду проекция одного и того же реального объекта (например, уголком может быть клюв
    # пролетающей птицы - и вот на разных кадрах одного видео этот клюв может иметь разные проекции (координаты) на
    # изображении (ведь птица летит), но это будет один и тот же уголок (у нас в коде один и тот же уголок будет иметь
    # один и тот же номер (id)), ведь клюв один и тот же), которая явлется уголком) на разных (последовательных)
    # кадрах)) - оптический поток отслеживается алгоритмом Лукаса-Канаде - его параметры перечисляем в словаре:
    lks = dict(winSize=(25, 25),  # оптять же, размер окна для подсчета частных производных (см лекцию про
               # алгоритм Лукаса Канаде) - снова берем побольше
               maxLevel=5,  # количество пирамид, которые использует Лукас-Канада для отслеживания потока -> берём
               # побольше, но без фанатизма (а то что-нибудь не то отследит или долго будет)
               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01))  # критерий остановки отслеживания
    # потока: либо cv2.TERM_CRITERIA_EPS (- то есть погрешность поиска в пикселях)) станет < 0.01, либо число
    # итераций при отслеживании (то есть cv2.TERM_CRITERIA_COUNT) стало > 100 - это ооочень много на самом деле, и
    # может долго работать иногда, но берём с запасом

    # изображение из последовательности frame_sequence - это изображение в виде numpy массива (разумеется, двумерного
    # массива, так как все изображения у нас чёрно-белые), где каждый элемент имеет float-значение от 0 до 1 и
    # обозначает цвет (в смысле оттенок чёрного и белого - если элемент=0, то пиксель полностью чёрный, если 1 -
    # то пиксель полностью белый) соответствующего пикселя
    image_0 = frame_sequence[0]  # первый кадр последовательности

    # отслеживаем уголки в нём:
    corners = cv2.goodFeaturesToTrack(image=image_0, mask=None,
                                      maxCorners=MAX_CORNERS,
                                      qualityLevel=QUALITY_LEVEL / 3,  # на первом кадре качество берем еще меньше,
                                      # так как в последующих кадрах мы будем искать уголки в тех местах изображения,
                                      # где еще нет уголков, поэтому если на первом кадре будет высокое качество уголков
                                      # , то когда мы будем пытасться найти уголки в тех местах, где их еще нет (а нет
                                      # их там только потому, что качество уголков в этих местах не очень хорошее, а
                                      # потому, мы их не взяли раньше [goodFeaturesToTrack работает так: он ищет уголки
                                      # по всему изображению с качеством qualityLevel, а потом из них берёт
                                      # <= maxCorners наилучших по качеству уголков -> поэтому если в какой-то области
                                      # изображения нет уголков, значит либо их там вовсе нет (что вряд ли - уж
                                      # какие-то наверняка есть), либо goodFeaturesToTrack уже набрал maxCorners
                                      # более качественных уголков на других участках изображения - то есть в этой
                                      # области изображение качество угоклво плохое (хуже, чем в других областях)
                                      # (но качество их все равно не меньше, чем qualityLevel, так как только
                                      # такие и рассматриваем!) - в результате, когда будем искать уголки уже
                                      # конкретно в этой области, мы все равно возьмем эти менее качественные уголки,
                                      # так как выбор стал меньше (не рассматриваем уже те области, откуда сначла
                                      # брали хорошие уголки), но качество этих уголков опять же будет не хуже
                                      # qualityLevel]),
                                      # мы всё равно возбмём уголки еще более низкого качества, поэтому логично просто
                                      # сразу установить для первого кадра качество пониже....
                                      minDistance=MIN_DISTANCE,
                                      blockSize=BLOCK_SIZE)  # нашли уголки (в виде numpy-массива формы (n, 1, 2) - уже говорили в самом начале)
    if corners is None:  # если не нашли уголок, укажем какую-то точку на всякий случай - это можно и не делать, так
        # мы будем работать с достаточно адекватными видео
        corners = np.array([[[1, 1]]])
    n = len(corners)
    ids = np.arange(n, dtype=np.int64)  # индексы (номера, то есть id) для уголков первого кадра расставим так
    # (как оказалось, очень важен тип именно int64 для дальнейших функций по отслеживанию камеры) -  на самом деле не
    # важно, какие именно числа присваивать уголкам в качетсве индексов (номеров) - главное, чтобы
    # у разных уголков - разные номера
    prev_max_ids = ids.max()  # зафиксируем максимальный индекс, который сейчас есть у уголков (просто максимальный
    # индекс, котрый когда-либо использовался для нумерации уголков)

    frame_corners = FrameCorners(ids=ids,
                                 points=corners,
                                 sizes=np.full(n, DRAW_SIZE))  # составили класс уголков, состоящий из номеров уголков
    # (то есть id), записанные в поле ids и их координат на изображении - поле points
    builder.set_corners_at_frame(0, frame_corners)  # передали номер кадра и уголки на нём

    for frame, image_1 in enumerate(frame_sequence[1:], 1):  # теперь проходимся по всем оставшимся кадам
        """
        итак, в начале итерации цикла у нас есть предыдущий кадр image_0 (изначально на первой итерации цикла - это
        просто первый кадр последовательности) и для него мы знаем уголки: массив corners содержит координаты уголков
        на изображении image_0, а массив ids содержит индексы (id) соответсующих уголков
        
        так же у нас есть кадр image_1 (его номер в последовательности = frame) - это следующий за image_0 кадр - и для
        него мы тоже хотим получить массивы corners и ids
        """
        prev_img = np.uint8(image_0 * 255)  # переводим изображения в формат, когда цвет пикселя - от 0 до 255
        next_img = np.uint8(image_1 * 255)  # (0 - чёрный, 255 - белый пиксель) - нужно для корректной работы
        # последующих функций

        # получаем положения (координаты) уголков (- массив new_corners) на новой картинке по уже найденным
        # положениям (-массив corners) этих же уголков на предыдущей картинкия - то есть просто отслеживаем их
        # оптический поток:
        new_corners, status, _ = cv2.calcOpticalFlowPyrLK(prev_img, next_img, corners, None, **lks)
        # получается, что corners и new_corners - это положения одних и тех же уголков (у них одинаковые id, которые
        # хранятся в масссиве ids) на последовательных кадрах

        # из всех новых уголков берём только корректно отслеженные (у них соответсвующий элемент в массиве status
        # равен 1) - это и будут отслеженные уголки нового кадра:
        ids = ids[status.reshape(-1) == 1]  # не забываем про id уголков - берём id корректно отслеженных уголков
        corners = new_corners[status.reshape(-1) == 1]  # а тут берём координаты корректно отслеженных уголков
        n = len(corners)  # количество уголков на новом кадре
        """итак, сейчас у нас для кадра image_1 уже есть уголки (массивы corners и ids как были для кадра image_0),
        полученные отслеживанием оптического потока - но этого недостаточно, ведь поток мог не отследиться, если 
        слишком резкий переход между кадрами был... - поэтому пополним массивы уголков на image_1 дополнительно 
        задетектировав уголки на этом кадре:"""

        if n < MAX_CORNERS:  # если на новом кадре мало уголков после отслеживания потока, пробуем найти ещё, задетектировав их
            mask = np.full(image_1.shape, 255, dtype=np.uint8)  # создаём массив такой же по размеру как изображение и
            # заполняем его ненулевыми числами - например, 255
            for x, y in corners.reshape(-1, 2):  # помним, что corners размера (n,1,2) -> делаем reshape
                # рисуем в массиве mask вокруг уголков круг из нулей радиусом = минимальная дистанция между
                # уголками (это нужно, чтобы далее на нашем новом изображении (на image_1) искать уголки не рядом с уже
                # существующими (которые мы отследили оптическим потоком с предыдущего кадра - c image_0), а на
                # удалении от них >= чем как раз это минимальное расстояние между уголками):
                cv2.circle(mask, (int(x), int(y)), MIN_DISTANCE, 0, -1)

            # используем нашу mask, полученную только что, чтобы искать уголки подальше от уже найденных уголков:
            candidates_corners = cv2.goodFeaturesToTrack(image_1, mask=mask, **features)

            if candidates_corners is not None:  # если новые уголки нашлись, то добавляем их (но не больше, чем нужно)
                corners = np.concatenate([corners, candidates_corners[:MAX_CORNERS - n]])  # добавляем уголки (но суммарно чтобы стало не больше MAX_CORNERS)
                n = len(corners)  # новое количество уголков
                # для новых уголков догенерируем id, но только начиная с prev_max_ids + 1, чтобы у новых уголков
                # были другие id, чем старых (у новых уголков id не должны совпадать с никакими использовавшимися ранее,
                # ведь это другие уголки):
                ids = np.concatenate([ids, np.arange(prev_max_ids + 1, n - len(ids) + prev_max_ids + 1)])  # добавили id

        prev_max_ids = max(prev_max_ids, ids.max())  # обновляем максимум (берём максимум от старого максимума и
        # текущего максимального индекса, так как если новых уголков не задетектилось, то максимальный использованный
        # индекс должен остаться прежним, а вот текущий максимум может уменьшиться (если какие-то треки со старыми
        # уголками прервались (оптический поток не отследился)) - и тогда если написать просто prev_max_ids = ids.max(),
        # то это уменьшит максимум, что плохо, ведь новые уголки потом будем обозначать старыми индексами уже
        # исчезнувших с экрана уголков... поэтому пишем именно так, как написано -> в итоге снова prev_max_ids - это
        # максимальный использованные индекс уголка)
        frame_corners = FrameCorners(
            ids=ids,
            points=corners,
            sizes=np.full(n, DRAW_SIZE),
        )
        builder.set_corners_at_frame(frame, frame_corners)  # добавляем уголки на кадре frame
        image_0 = image_1  # теперь предыдущим кадром становится image_1 и мы продожаем цикл


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
