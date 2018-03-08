import numpy as np
import os
import Augmentor
from itertools import combinations
from tensorflow.contrib.keras import preprocessing


def build_class_generator(class_path, probability, width, height):
    pipeline = Augmentor.Pipeline(class_path)
    pipeline.random_erasing(probability, 0.4)
    pipeline.rotate(probability, 20, 20)
    pipeline.shear(probability, 20, 20)
    pipeline.skew(probability, 0.8)
    pipeline.zoom(probability, 1.1, 1.5)
    pipeline.random_distortion(probability, 3, 3, 3)
    pipeline.random_distortion(probability, 8, 8, 3)
    pipeline.resize(1.0, width, height)
    return pipeline.keras_generator(batch_size=1)


def build_all_generators(data_directory, probability, target_shape):
    width, height, _ = target_shape
    generators = list()
    for class_directory in os.listdir(data_directory):

        class_path = os.path.join(data_directory, class_directory)
        generator = build_class_generator(class_path,
                                          probability,
                                          width, height)
        generators.append(generator)
    return generators


def fill_batch_entry(left_label, right_label, generators, target_shape):

    # Left.
    left_image, _ = next(generators[left_label])
    left_image = left_image.reshape(target_shape)

    # Right.
    right_image, _ = next(generators[right_label])
    right_image = right_image.reshape(target_shape)

    # Similarity.
    is_same = 1.0 if left_label == right_label else 0.0
    return left_image, right_image, is_same


def get_different_pairs(batch_size, amount_classes):
    all_pair_combinations = np.array([*combinations(list(range(amount_classes)), 2)])

    amount_different_pairs = batch_size // 2
    amount_repeats = amount_different_pairs // len(all_pair_combinations)
    remainder = amount_different_pairs - amount_repeats * len(all_pair_combinations)
    pairs = all_pair_combinations.copy()
    for _ in range(amount_repeats - 1):
        pairs = np.concatenate((pairs,  all_pair_combinations))
    return np.concatenate((pairs,  all_pair_combinations[:remainder]))


def test_image_arrays(*args):

    for array in args:
        assert not np.isnan(array).any()

        assert np.max(array) <= 1.0
        assert np.min(array) >= 0.0

        for element in array:
            assert np.mean(element) != 0
            assert np.std(element) != 0


def next_batch(batch_size,
               probability=0.1,
               data_directory='./signs',
               target_shape=(100, 100, 3)):

    # These create augmented images for each class.
    generators = build_all_generators(data_directory,
                                      probability,
                                      target_shape)

    # How many classes do we have?
    amount_classes = len(generators)

    # Index calculations.
    different_pairs = get_different_pairs(batch_size, amount_classes)
    amount_same_pairs = batch_size // 2

    # Initialise state.
    left_images = np.zeros((batch_size,) + target_shape)
    right_images = np.zeros((batch_size,) + target_shape)
    is_similar = np.zeros((batch_size, 1))

    # Similar pairs.
    left_labels = np.arange(batch_size) % amount_classes
    right_labels = np.arange(batch_size) % amount_classes

    # Different pairs.
    left_labels[amount_same_pairs:] = different_pairs.T[0]
    right_labels[amount_same_pairs:] = different_pairs.T[1]

    # Indices for the arrays.
    indices = np.arange(batch_size)
    containers = zip(indices, left_labels, right_labels)

    while True:

        for index, left_label, right_label in containers:

            data = fill_batch_entry(left_label,
                                    right_label,
                                    generators,
                                    target_shape)
            left_images[index] = data[0]
            right_images[index] = data[1]
            is_similar[index, 0] = data[2]

        test_image_arrays(left_images, right_images)

        yield left_images, right_images, is_similar


def unison_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def load_data_from_directory(data_directory, target_shape, grayscale):

    images = []
    labels = []

    class_directories = os.listdir(data_directory)
    class_number = 0

    for class_directory in class_directories:

        class_path = os.path.join(data_directory, class_directory)

        files = os.listdir(class_path)

        for file_name in files:

            if file_name.endswith('.png'):

                file_path = os.path.join(class_path, file_name)
                image = preprocessing.image.load_img(file_path,
                                                     target_size=target_shape,
                                                     grayscale=grayscale)
                image = preprocessing.image.img_to_array(image)
                images.append(image)

                labels.append(class_number)

        class_number += 1

    images = np.array(images) / 255.0
    labels = np.array(labels)

    return images, labels
