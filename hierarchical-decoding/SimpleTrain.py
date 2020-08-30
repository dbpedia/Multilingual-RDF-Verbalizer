
from arguments import get_args
from Trainer import train
import numpy as np
import random

if __name__ == "__main__":
  args = get_args()
  global step

train(args)
