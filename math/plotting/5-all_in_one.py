#!/usr/bin/env python3
""" This module includes the function named all_in_one """

import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """ This function collects previous graphs
    in this plotting module into 3x2 grid """

    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 2)
    fig.suptitle("All in One")

    x0 = np.arange(0, 11)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(x0, y0, 'r-')
    ax0.axis((0, 10, None, None))

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.scatter(x1, y1, c='magenta')
    ax1.set_xlabel("Height (in)")
    ax1.set_ylabel("Weight (lbs)")
    ax1.set_title("Men's Height vs Weight")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x2, y2)
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("Fraction Remaining")
    ax2.set_title("Exponential Decay of C-14")
    ax2.axis((0, 28650, None, None))
    ax2.set_yscale("log")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(x3, y31, 'r--')
    ax3.plot(x3, y32, 'g')
    ax3.axis((0, 20000, 0, 1))
    ax3.set_xlabel("Time (years)")
    ax3.set_ylabel("Fraction Remaining")
    ax3.set_title("Exponential Decay of Radioactive Elements")
    ax3.legend(['C-14', 'Ra-226'])

    ax4 = fig.add_subplot(gs[2, :])
    ax4.hist(student_grades, bins=range(0, 101, 10), edgecolor="black")
    ax4.set_xlabel("Grades")
    ax4.set_ylabel("Number of Students")
    ax4.set_title("Project A")
    ax4.set_xlim(0, 100)
    ax4.set_xticks(range(0, 101, 10))
    ax4.set_ylim(0, 30)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
