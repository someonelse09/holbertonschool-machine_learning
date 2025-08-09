#!/usr/bin/env python3
""" This module includes function add_arrays """


matrix_shape = __import__('2-size_me_please').matrix_shape


def add_arrays(arr1, arr2):
    """ This function allows us to
    add two arrays as long as their shapes are same """

    if (matrix_shape(arr1) != matrix_shape(arr2)):
        return None
    new_list = []
    for i in range(len(arr1)):
        new_list.append(arr1[i] + arr2[i])
    return new_list
