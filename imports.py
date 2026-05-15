import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms