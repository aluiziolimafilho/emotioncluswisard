import cv2
import random
import time

def sliding_window(image, stepSize, windowSize):
    for y in range(windowSize[0], image.shape[0], stepSize):
        for x in range(windowSize[1], image.shape[1], stepSize):
            clone = image.copy()
            cv2.imshow("test", image[y - windowSize[0]:y, x - windowSize[1]:x])
            cv2.waitKey(0)
            cv2.rectangle(clone, (x - windowSize[1], y - windowSize[0]), (x, y), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
    cv2.destroyAllWindows()

def random_indexes(data_size, number_of_rams):
    number_list = list(range(0, data_size))
    random.shuffle(number_list)
    indexes = []

    for count in range(number_of_rams, data_size + 1, number_of_rams):
        indexes.append(number_list[count - number_of_rams:count])
    return indexes
