import jax.numpy as np
import numpy as onp
from torch.utils import data
from torchvision.datasets import CIFAR10


def numpy_collate(batch):
    if isinstance(batch[0], onp.ndarray):
        return onp.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return onp.array(batch)

class NumpyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                    shuffle=False, sampler=None,
                    batch_sampler=None, num_workers=0,
                    pin_memory=False, drop_last=False,
                    timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                collate_fn=numpy_collate,
                pin_memory=pin_memory,
                drop_last=drop_last,
                timeout=timeout,
                worker_init_fn=worker_init_fn)

class FlattenCastNormalize(object):
    """Casts a PIL Image to a numpy array
    and flattens the image
    and normalizes it so that the values range from 0 to 1"""
    def __call__(self, pic):
        MAX_VALUE = 255.
        return onp.ravel(onp.array(pic, dtype=np.float32))/MAX_VALUE

class ToOneHot(object):
    """ casts an array of labels to a one-hot array """
    def __init__(self, num_classes):
        self.k = num_classes
    def __call__(self, label):
        one_hot_label = onp.zeros(self.k, dtype=onp.float32)
        one_hot_label[label] = 1.
        return one_hot_label
