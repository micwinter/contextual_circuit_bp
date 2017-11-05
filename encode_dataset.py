import os
from config import Config
from utils import py_utils
from argparse import ArgumentParser
from ops.data_to_tfrecords import data_to_tfrecords


def encode_dataset(dataset):
    """
    Encodes a dataset of cells to tfrecords by name.

    Parameters
    ----------
    dataset : string
        Name of dataset to be encoded. Dataset should be found in
        /dataset_processing and have its own 'init' function.
    """
    config = Config()
    data_class = py_utils.import_module(dataset)
    data_proc = data_class.data_processing()
    files, labels = data_proc.get_data()
    targets = data_proc.targets
    im_size = data_proc.im_size
    preproc_list = data_proc.preprocess
    ds_name = os.path.join(config.tf_records, data_proc.name)
    data_to_tfrecords(
        files=files,
        labels=labels,
        targets=targets,
        ds_name=ds_name,
        im_size=im_size,
        preprocess=preproc_list)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='Name of the dataset.')
    args = parser.parse_args()
    encode_dataset(**vars(args))
