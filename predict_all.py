import pandas as pd
import numpy as np
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import glob
from typing import List, Dict, Any
import threading
import time
import json
import torch
import traceback
from model import Kronos, KronosTokenizer, KronosPredictor

model_path = glob.glob('./*/basemodel_*/best_model')
token_path = glob.glob('./tokenizer_*/best_model')

def load_models(tokenizer_path: str, model_path: str, device: str = 'cuda:0') -> tuple[KronosTokenizer, Kronos]:
    """Loads the fine-tuned tokenizer and predictor model."""
    device = torch.device(device)
    print(f"Loading models onto device: {device}...")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path).to(device).eval()
    model = Kronos.from_pretrained(model_path).to(device).eval()
    return tokenizer, model

model_list = []
for i in range(len(model_path)):
    tokenizer, model = load_models(token_path[i], model_path[i])
    model_list.append((tokenizer, model))

def init_predictors_for_thread(model_path, local_device: str = "cuda:0"):
    """为每个线程初始化独立的predictor实例"""
    predictors = []
    for i in range(5):
        tokenizer, model = model_path[i][0], model_path[i][1]
        predictor = KronosPredictor(
            model, tokenizer, 
            device=local_device, 
            max_context=512
        )
        predictors.append(predictor)
    return predictors

predictors_list = init_predictors_for_thread(model_list)

def create_prediction_windows(data, window_size=10, prediction_horizon=5):
    """
    创建用于预测的滑动窗口，返回DataFrame格式的窗口
    """
    rolling_x = []
    rolling_x_timestamps = []
    rolling_y_timestamps = []
    data = data.sort_values('datetime').reset_index(drop=True)
    
    for i in range(len(data) - window_size - prediction_horizon + 1):
        # 输入窗口: 过去window_size个时间点的DataFrame
        input_window = data.iloc[i:i + window_size].copy()
        
        # 目标窗口: 未来prediction_horizon个时间点的DataFrame
        target_window = data.iloc[i + window_size:i + window_size + prediction_horizon].copy()
        rolling_x.append(input_window[['open', 'high', 'low', 'close', 'volume']])
        rolling_x_timestamps.append(input_window['timestamps'])
        rolling_y_timestamps.append(target_window['timestamps'])
    
    return rolling_x, rolling_x_timestamps, rolling_y_timestamps

def extract_open_price(prediction_list: list):
    open_price_list = []
    for prediction_df in prediction_list:
        open_price_list.append(prediction_df.iloc[-1,0])
    return open_price_list

file_path = "./bond-1m"
bond_name = os.listdir(file_path)
out = './bond-1m-result'

for i in bond_name:
    file = os.path.join(file_path, i)
    out_path = os.path.join(out, i)
    df = pd.read_csv(file)
    df['timestamps'] = pd.to_datetime(df['datetime'])
    n_total = df.shape[0] - 10 - 5 -1
    predict_value = np.zeros((n_total, 5))
    rolling_x, rolling_x_timestamps, rolling_y_timestamps = create_prediction_windows(df)
    count = 0

    for predictor in predictors_list:
        temp_pred = predictor.predict_batch(df_list=rolling_x, x_timestamp_list=rolling_x_timestamps, y_timestamp_list=rolling_y_timestamps, pred_len=5)
        temp_open_price = extract_open_price(temp_pred)
        predict_value[:,count] = temp_open_price
    
    predict_value_mean = predict_value.mean(axis=1)
    df['pred'] = np.nan
    df[9:-6, -1] = predict_value_mean
    df = df[['timestamps', "open", 'pred']]
    df.to_csv(out_path, index=False)