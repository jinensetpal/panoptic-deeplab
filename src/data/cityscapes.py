#!/usr/bin/env python3

from dagshub.data_engine import datasources, datasets
from src.data.common import get_center_targets
import tensorflow as tf
from src import const
from PIL import Image
import pandas as pd
import numpy as np


def enrich(entry):
    split = entry['path'].split('/')
    entry['split'] = None if len(split) == 1 else split[1]
    if entry['split'] != 'invalid':
        entry['location'] = split[2]
        entry['id'] = '.'.join(split[-1].split('_')[:3])
    return entry


def preprocess():
    if not len(datasources.get_datasources(const.REPO_NAME)):
        ds = datasources.create_from_bucket(const.REPO_NAME,
                                            const.DATASOURCE_NAME,
                                            const.BUCKET_NAME)
    else: ds = datasources.get_datasource(const.REPO_NAME, name=const.DATASOURCE_NAME)

    res = []
    for key, frame in ds.all().dataframe.apply(enrich, axis=1).groupby('id'):
        entry = dict(frame.iloc[0])
        for column, path in zip(frame.path.apply(lambda x: x.split('_')[-1].split('.')[0]), frame.path): entry[column] = path
        res.append(entry)

    ds.upload_metadata_from_dataframe(pd.DataFrame(res).drop(['path', 'datapoint_id', 'size', 'dagshub_download_url'], axis=1).dropna(), path_column='leftImg8bit')

    q = ds['split'] != None  # noqa: E711
    q.save_dataset(const.DATASET_NAME)


def semantic_map(filename):
    img = np.array(Image.open(filename))
    res = np.empty((*img.shape, len(const.LABELS)))

    for label in range(-1, len(const.LABELS)):
        res[:, :, label] += (img == label).astype(int)

    return tf.convert_to_tensor(np.resize(res, (*const.IMG_SIZE, const.N_CLASSES)), dtype=tf.float32)


class InstanceTensorizer:
    def __init__(self):
        self.filename = ''

    def process(self, filename):
        self.filename = filename
        self.res = [tf.convert_to_tensor(np.resize(x, (*const.IMG_SIZE, idx+1))) for idx, x in enumerate(get_center_targets(np.array(Image.open(filename))).values())]

    def instance_center(self, filename):
        if self.filename != filename: self.process(filename)
        return self.res[0]

    def center_regression(self, filename):
        if self.filename != filename: self.process(filename)
        return self.res[1]


def image_norm(filename):
    return tf.convert_to_tensor(np.resize(np.array(Image.open(filename).convert('RGB')), const.IMG_SHAPE) / 255)


def get_generators(force_preprocessing=False):
    if force_preprocessing or not len(datasets.get_datasets(const.REPO_NAME)): preprocess()
    ds = datasets.get_dataset(const.REPO_NAME, const.DATASET_NAME)

    instance_tensorizer = InstanceTensorizer()
    kwargs = {'flavor': 'tensorflow',
              'shuffle': True,
              'strategy': 'background',
              'post_hook': lambda x: (x[0], x[1:]),
              'batch_size': const.BATCH_SIZE,
              'metadata_columns': ['labelIds', 'instanceIds', 'instanceIds'],
              'tensorizers': [image_norm, semantic_map, instance_tensorizer.instance_center, instance_tensorizer.center_regression]}
    if const.TESTING: return [(ds['split'] == split).head(10).as_ml_dataloader(**kwargs) for split in ['train', 'val', 'test']]
    return [(ds['split'] == split).all().as_ml_dataloader(**kwargs) for split in ['train', 'val', 'test']]


if __name__ == '__main__':
    train, valid, test = get_generators(force_preprocessing=True)
