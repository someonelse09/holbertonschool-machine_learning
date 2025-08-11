#!/usr/bin/env python3
""" This module includes
the function called frequency
which decribes an histogram of scores """


import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """ This function plots
    histogram of student's scores """

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor="black")
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.xlim(0, 100)
    plt.xticks(range(0, 101, 10))
    plt.ylim(0, 30)
