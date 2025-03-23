import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import re
from collections import Counter
from sklearn.model_selection import train_test_split
import random
import nltk.translate.bleu_score as bleu