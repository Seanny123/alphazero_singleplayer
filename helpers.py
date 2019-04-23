import random
import os
from shutil import copyfile

import numpy as np
from gym import spaces


def stable_normalizer(x, temp):
    """ Computes x[i]**temp/sum_i(x[i]**temp)  """
    x = (x / np.max(x)) ** temp
    return np.abs(x / np.sum(x))


def argmax(x: np.ndarray):
    """ assumes a 1D vector x  """
    x = x.flatten()
    if np.any(np.isnan(x)):
        print('Warning: Cannot argmax when vector contains nans, results will be wrong')
    try:
        winners = np.argwhere(x == np.max(x)).flatten()
        winner = random.choice(winners)
    except Exception as e:
        print("Argmax error", e)
        winner = np.argmax(x)  # numerical instability ?
    return winner


def check_space(space):
    """ Check the properties of an environment state or action space   """
    if isinstance(space, spaces.Box):
        dim = space.shape
        discrete = False
    elif isinstance(space, spaces.Discrete):
        dim = space.n
        discrete = True
    else:
        raise NotImplementedError('This type of space is not supported')

    return dim, discrete


def store_safely(folder, name, to_store):
    """ to prevent losing information due to interruption of proces    """
    new_name = folder + name + '.npy'
    old_name = folder + name + '_old.npy'
    if os.path.exists(new_name):
        copyfile(new_name, old_name)
    np.save(new_name, to_store)
    if os.path.exists(old_name):
        os.remove(old_name)


class Database:
    """ Database """

    def __init__(self, max_size: int, batch_size: int):
        self.max_size = max_size
        self.batch_size = batch_size
        self.clear()
        self.sample_array = None
        self.sample_index = 0

    def clear(self):
        self.experience = []
        self.insert_index = 0
        self.size = 0

    def store(self, experience):
        if self.size < self.max_size:
            self.experience.append(experience)
            self.size += 1
        else:
            self.experience[self.insert_index] = experience
            self.insert_index += 1
            if self.insert_index >= self.size:
                self.insert_index = 0

    def store_from_array(self, *args):
        for i in range(args[0].shape[0]):
            entry = []
            for arg in args:
                entry.append(arg[i])
            self.store(entry)

    def reshuffle(self):
        self.sample_array = np.arange(self.size)
        random.shuffle(self.sample_array)
        self.sample_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if (self.sample_index + self.batch_size > self.size) and (not self.sample_index == 0):
            self.reshuffle()  # Reset for the next epoch
            raise StopIteration

        if self.sample_index + 2 * self.batch_size > self.size:
            indices = self.sample_array[self.sample_index:]
            batch = [self.experience[i] for i in indices]
        else:
            indices = self.sample_array[self.sample_index:self.sample_index + self.batch_size]
            batch = [self.experience[i] for i in indices]
        self.sample_index += self.batch_size

        arrays = []
        for i in range(len(batch[0])):
            to_add = np.array([entry[i] for entry in batch])
            arrays.append(to_add)
        return tuple(arrays)

    next = __next__
