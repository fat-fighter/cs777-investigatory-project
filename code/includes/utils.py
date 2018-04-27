import numpy as np

from csv import reader
from random import randrange


np.warnings.filterwarnings('ignore')


def load_csv(filename, mapping=None, normalize=False):
    dataset = list()

    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append([float(_field) for _field in row])

    dataset = np.array(dataset)
    dataset_X, dataset_Y = dataset[:, :-1], dataset[:, -1]

    if mapping != None:
        dataset_X = mapping(dataset_X)

    if not normalize:
        dataset = np.concatenate([dataset_X, dataset_Y[:, None]], axis=1)
        return dataset

    rows_min = np.min(dataset_X, axis=0)
    rows_max = np.max(dataset_X, axis=0)
    dataset_X = np.where(
        rows_max == rows_min,
        1,
        (dataset_X - rows_min) / (rows_max - rows_min)
    )

    dataset = np.concatenate([dataset_X, dataset_Y[:, None]], axis=1)

    return dataset


def folds(dataset, n_folds):
    dataset_split = list()
    dataset = np.copy(dataset)

    np.random.shuffle(dataset)

    fold_size = int(len(dataset) / n_folds)

    fold = list()
    for row in dataset:
        if len(fold) < fold_size:
            fold.append(row)
        else:
            dataset_split.append(np.array(fold))
            fold = list()

    if len(fold) > 0:
        dataset_split.append(np.array(fold))

    return np.array(dataset_split)


def get_data(dataset):

    # DIABETES
    if dataset == "diabetes":
        filename = 'data/indians-diabetes.csv'
        return load_csv(
            filename,
            mapping=lambda data: np.concatenate(
                [data, data ** 2, data ** 3], axis=1
            ),
            normalize=True
        ), 2
    # MNIST
    elif dataset == "mnist":
        import mnist

        images = mnist.train_images()
        labels = mnist.train_labels()

        images = images.reshape(
            (images.shape[0], images.shape[1] * images.shape[2])
        )

        images = images / 255.0

        train_set = np.concatenate([images, labels[:, None]], axis=1)

        images = mnist.test_images()
        labels = mnist.test_labels()

        images = images.reshape(
            (images.shape[0], images.shape[1] * images.shape[2])
        )

        images = images / 255.0

        test_set = np.concatenate([images, labels[:, None]], axis=1)

        return train_set, test_set, 10
