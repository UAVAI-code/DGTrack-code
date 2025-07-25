
from collections import namedtuple
import importlib
from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "lib.test.evaluation.%sdataset"

dataset_dict = dict(

    visdrone=DatasetInfo(module=pt % "visdrone", class_name="VISDRONEDataset", kwargs=dict()),
    uavdt=DatasetInfo(module=pt % "uavdt", class_name="UAVDTDataset", kwargs=dict()),


)


def load_dataset(name: str):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)
    return dataset.get_sequence_list()


def get_dataset(*args):
    """
    Get a single or set of datasets. 获取数据集名称
    接收可变数量的参数 *args，这些参数是数据集的名称
    """
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset
