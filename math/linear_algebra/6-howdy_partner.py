#!/usr/bin/env python3
""" This module includes the function cat_arrays """


def cat_arrays(arr1, arr2):
    """ This function concatenates two arrays """

    new_arr = []
    for num1 in arr1:
        new_arr.append(num1)
    for num2 in arr2:
        new_arr.append(num2)
    return new_arr
