import math
from dataclasses import dataclass
from accLayerSimplified import AccLayer
import numpy as np
import torch
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import random
from ttUtilities.auxFunctions import integerToBinaryArray


@dataclass
class BinaryOutputNeuron:
	name: str
	accLayer: AccLayer
	importancePerClass: dict
	importance: float

	def __post_init__(self):
		self.sopForm = ''
		self.activeTTEntries = 0

