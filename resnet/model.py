from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.utils import _log_api_usage_once
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
