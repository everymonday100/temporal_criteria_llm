# paired_bootstrap_check.py
import sys
sys.path.insert(0, '.')
import numpy as np
import torch
from common.model import load_model
from common.generation import extract_answer
from common.datasets import load_gsm_hard, gsm_hard_prompt
from common.controller_model import ControllerModel
from common.stats import paired_bootstrap

# ... загрузить чекпоинт и прогнать D и B на тех же 300 проблемах,
# собрать per-problem correct/incorrect для обоих условий
# затем paired_bootstrap(d_correct, b_correct)