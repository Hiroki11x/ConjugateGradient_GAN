from .AdaBelief import AdaBelief
from .Fromage import Fromage
from .Yogi import Yogi
from .MSVAG import MSVAG
from .RAdam import RAdam
from .AdamW import AdamW

# issue2_hosoiというフォルダの直下に，pytorch_ganとpytorch_dnn_arsenal-masterがある想定
# ローカル環境に応じて変更する必要あり
import sys
import os
sys.path.append(os.environ['ARSENAL_PATH'])
from pytorch_dnn_arsenal.optimizer import KFACOptimizer