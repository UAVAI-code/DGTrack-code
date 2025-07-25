import torch
import torch.utils.data.dataloader
import importlib
import collections

string_classes = str
from lib.utils import TensorDict, TensorList

if float(torch.__version__[:3]) >= 1.9 or len('.'.join((torch.__version__).split('.')[0:2])) > 3:
    int_classes = int
else:
    from torch._six import int_classes


def _check_use_shared_memory():
    if hasattr(torch.utils.data.dataloader,
               '_use_shared_memory'):
        return getattr(torch.utils.data.dataloader, '_use_shared_memory')
    collate_lib = importlib.import_module(
        'torch.utils.data._utils.collate')
    if hasattr(collate_lib, '_use_shared_memory'):
        return getattr(collate_lib, '_use_shared_memory')
    return torch.utils.data.get_worker_info() is not None


def ltr_collate(batch):
    """Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])

    if isinstance(batch[0], torch.Tensor):
        out = None
        if _check_use_shared_memory():
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            if torch.utils.data.dataloader.re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():
            py_type = float if elem.dtype.name.startswith('float') else int
            return torch.utils.data.dataloader.numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], TensorDict):
        return TensorDict({key: ltr_collate([d[key] for d in batch]) for key in batch[0]})
    elif isinstance(batch[0], collections.Mapping):
        return {key: ltr_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], TensorList):
        transposed = zip(*batch)
        return TensorList([ltr_collate(samples) for samples in transposed])
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [ltr_collate(samples) for samples in transposed]
    elif batch[0] is None:
        return batch

    raise TypeError((error_msg.format(type(batch[0]))))


def ltr_collate_stack1(batch):
    """Puts each data field into a tensor. The tensors are stacked at dim=1 to form the batch"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _check_use_shared_memory():
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 1, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            if torch.utils.data.dataloader.re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 1)
        if elem.shape == ():
            py_type = float if elem.dtype.name.startswith('float') else int
            return torch.utils.data.dataloader.numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], TensorDict):
        return TensorDict({key: ltr_collate_stack1([d[key] for d in batch]) for key in batch[0]})
    elif isinstance(batch[0], collections.Mapping):
        return {key: ltr_collate_stack1([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], TensorList):
        transposed = zip(*batch)
        return TensorList([ltr_collate_stack1(samples) for samples in transposed])
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [ltr_collate_stack1(samples) for samples in transposed]
    elif batch[0] is None:
        return batch

    raise TypeError((error_msg.format(type(batch[0]))))


class LTRLoader(torch.utils.data.dataloader.DataLoader):
    """
    1 结合数据集 (dataset) 和采样器 (sampler) 提供单进程或多进程的数据加载。
    2 支持 自定义堆叠维度 (stack_dim)，以决定数据在哪个维度上堆叠形成批次。
    3 支持模式切换（如训练模式 training=True 或测试模式 training=False）。
    4 提供额外的配置选项（如 epoch_interval、stack_dim 等）.

    Note: The only difference with default pytorch DataLoader is that an additional option stack_dim is available to
            select along which dimension the data should be stacked to form a batch.

    Arguments:
        dataset (Dataset): 数据集对象，用于加载数据.
        batch_size (int, optional): 每个批次的样本数量
            (default: 1).
        shuffle (bool, optional): 是否在每个 epoch 开始时打乱数据顺序.
        sampler (Sampler, optional): 自定义采样器，用于定义数据采样的策略.
        batch_sampler (Sampler, optional):自定义批量采样器，直接返回一批索引
        num_workers (int, optional): 用于数据加载的子进程数量。0 表示数据在主线加载.
            (default: 0)
        collate_fn (callable, optional): 用于将样本列表合并为批次的函数.
        stack_dim (int): 指定数据在哪个维度上堆叠以形成批次（默认值为 0）. (default: 0)
        pin_memory (bool, optional): 是否将数据复制到 CUDA 的固定内存中，以加速 GPU 数据传输.
        drop_last (bool, optional): 是否丢弃最后一个不完整的批次（当数据集大小不能被 batch_size 整除时）. (default: False)
        timeout (numeric, optional): 从子进程收集数据的超时时间（以秒为单位）. (default: 0)
        worker_init_fn (callable, optional): 用于初始化每个子进程的函数. (default: None)

    .. note:: By default, each worker will have its PyTorch seed set to
              ``base_seed + worker_id``, where ``base_seed`` is a long generated
              by main process using its RNG. However, seeds for other libraries
              may be duplicated upon initializing workers (w.g., NumPy), causing
              each worker to return identical random numbers. (See
              :ref:`dataloader-workers-random-seed` section in FAQ.) You may
              use ``torch.initial_seed()`` to access the PyTorch seed for each
              worker in :attr:`worker_init_fn`, and use it to set other seeds
              before data loading.

    .. warning:: If ``spawn`` start method is used, :attr:`worker_init_fn` cannot be an
                 unpicklable object, e.g., a lambda function.
    """

    __initialized = False

    def __init__(self,
                 name,
                 dataset,
                 training=True,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 num_workers=0,
                 epoch_interval=1,

                 collate_fn=None,
                 stack_dim=0,
                 pin_memory=False,
                 drop_last=False,
                 timeout=0,
                 worker_init_fn=None
                 ):


        if collate_fn is None:
            if stack_dim == 0:
                collate_fn = ltr_collate
            elif stack_dim == 1:
                collate_fn = ltr_collate_stack1
            else:
                raise ValueError('Stack dim no supported. Must be 0 or 1.')

        super(LTRLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                                        num_workers, collate_fn, pin_memory, drop_last,
                                        timeout, worker_init_fn)

        self.name = name
        self.training = training
        self.epoch_interval = epoch_interval
        self.stack_dim = stack_dim
