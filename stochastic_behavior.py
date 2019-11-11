import jax.numpy as np
from jax import random
from jax.experimental.stax import Dense, Relu, LogSoftmax, serial

from torchvision.datasets import CIFAR10
import dataloading as dl

testset = CIFAR10('datasets',
                  download=True,
                  train=False,
                  transform=dl.FlattenCastNormalize(),
                  target_transform=dl.ToOneHot(num_classes=10))

full_test_generator = dl.NumpyLoader(dataset=testset,
                                     batch_size=len(testset),
                                     num_workers=0)
full_test_set = list(full_test_generator)[0]

def accuracy(params, batch):
  inputs, targets = batch
  target_class = np.argmax(targets, axis=1)
  predicted_class = np.argmax(predict(params, inputs), axis=1)
  return np.mean(predicted_class == target_class)

# This is the smallest network I found (in terms of depth) to exhibit this
# problem.
init_random_params, predict = serial(Dense(1024),
                                     Relu,
                                     Dense(1024),
                                     Relu,
                                     Dense(10),
                                     LogSoftmax)

rng = random.PRNGKey(0)

CIFAR10_IMGSIZE = (-1, 3072)
_, init_params = init_random_params(rng, CIFAR10_IMGSIZE)


for i in range(10):
  print(f"Running trial {i}")
  print(accuracy(init_params, full_test_set))
