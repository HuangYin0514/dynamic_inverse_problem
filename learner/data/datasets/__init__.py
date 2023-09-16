# encoding: utf-8

from .dynamic_data import DynamicData
from .dynamic_data_uniform_speed import DynamicData_UniformSpeed
from .dynamic_dataset import DynamicDataset

__dataset_factory = {
    "DynamicData": DynamicData,
    "DynamicData_UniformSpeed": DynamicData_UniformSpeed,
}


def get_dataset(dataset_name, config, logger, *args, **kwargs):
    if dataset_name not in __dataset_factory.keys():
        raise ValueError("Dataset '{}' is not implemented".format(dataset_name))
    dataset = __dataset_factory[dataset_name](config, logger, *args, **kwargs)
    return dataset.data
