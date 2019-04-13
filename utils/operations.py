import math


def safe_divide(numerator, denominator):
    if int(denominator) == 0:
        index = 0
    else:
        index = numerator / denominator
    return index


def division(list1, list2):
    if len(list2) == 0:
        return 0
    else:
        return len(list1) / len(list2)


def corrected_division(list1, list2):
    if len(list2) == 0:
        return 0
    else:
        return len(list1) / math.sqrt(2 * len(list2))


def root_division(list1, list2):
    if len(list2) == 0:
        return 0
    else:
        return len(list1) / math.sqrt(len(list2))


def squared_division(list1, list2):
    if len(list2) == 0:
        return 0
    else:
        return len(list1) ** 2 / len(list2)


def log_division(list1, list2):
    if len(list2) == 0:
        return 0
    else:
        return math.log(len(list1)) / math.log(len(list2))


def uber(list1, list2):
    if len(list1) == 0 or len(list2) == 0:
        return 0
    else:
        return math.log(len(list1)) ** 2 / math.log(len(set(list2)) / len(list1))
