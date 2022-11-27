import torch
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys, glob, re, json, argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from datetime import datetime, timedelta
from time import time
from loguru import logger
from src import models
warnings.filterwarnings('ignore')


def main():
    logger.info(f"Running:---{Path(__file__).name}---")
    dataset = 'audioset'
    state_dict_path = 'assets/models/as_mdl_0_wa.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_class = 527 if dataset == 'audioset' else 200
    audio_model = models.EffNetAttention(label_dim=num_class, b=2, pretrain=False, head_num=4)
    audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(torch.load(state_dict_path), strict=False)

    for param_tensor in audio_model.state_dict():
        print(param_tensor, "\t", audio_model.state_dict()[param_tensor].size())

    for param in model.parameters():
        param.requires_grad = False
    audio_model.fc = nn.Linear(in_features=2048, out_features=num_outputs, bias=True)
    return model
if __name__ == '__main__':
    main()
