#!/usr/bin/env python3
""" This module includes function called bars """

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """ This function depicts how 4 type of
    fruits distributed among Farrah, Fred and Felicia
    in the bar form """

    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ('Farrah', 'Fred', 'Felicia')
    plt.bar(people, fruit[0], width=0.5, color='red', label='apples')
    plt.bar(people, fruit[1], width=0.5, color='yellow',
            bottom=fruit[0], label='bananas')
    plt.bar(people, fruit[2], width=0.5, color='#ff8000',
            bottom=fruit[0] + fruit[1], label='oranges')
    plt.bar(people, fruit[3], width=0.5, color='#ffe5b4',
            bottom=fruit[0] + fruit[1] + fruit[2], label='peaches')

    plt.ylabel('Quantity of Fruit')
    plt.yticks(range(0, 81, 10))
    plt.ylim(0, 80)
    plt.title("Number of Fruit per Person")
    plt.legend(['apples', 'bananas', 'orange', 'peaches'])
    plt.show()
