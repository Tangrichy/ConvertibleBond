import pandas as pd
import numpy as np
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
from typing import List, Dict, Any
import threading
import time
import json
import torch
import traceback
from model import Kronos, KronosTokenizer, KronosPredictor

# /root/private_data/KronosFT/finetune_csv/FileFT/SH.113525/basemodel_SH.113525
model_path = glob.glob('./basemodel_*/best_model')
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


def process_single_file(file_path: str, token_model_predictors: List[str]) -> Dict[str, Any]:
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 读取Excel文件
            df = pd.read_csv(file_path)
            df['timestamps'] = pd.to_datetime(df['datetime'])
            
            # 执行完整预测流程
            result_df, stats = process_dataframe_complete(df, token_model_predictors)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            return {
                'file_path': file_path,
                'status': 'success',
                'data': result_df,
                'statistics': stats,
                'processing_time': processing_time,
                'data_points': len(df)
            }
            
        except Exception as e:
            return {
                'file_path': file_path,
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }


def process_dataframe_complete(df: pd.DataFrame, predictors: List, lookahead: int = 5) -> tuple:
        """
        对DataFrame执行完整的数据处理流程
        """
        n_total = len(df) - 10 - lookahead + 1
        if n_total <= 0:
            # 如果数据不足，返回原始DataFrame但标记为无效
            result_df = df.copy()
            result_df['pred_open'] = np.nan
            result_df['pred_valid'] = False
            return result_df, {'valid_predictions': 0}
        
        # 预分配数组存储所有预测
        predict_value = np.zeros((n_total, 5))
        
        # 执行滑动窗口预测
        for i in range(n_total):
            input_window = df.iloc[i:i + 10]
            target_window = df.iloc[i + 10:i + 10 + lookahead]
            
            # 批量预测
            batch_pred = batch_predict(
                predictors, input_window, target_window['timestamps']
            )
            predict_value[i] = batch_pred
            
            # 存储详细预测结果
        
        # 计算预测均值
        pred_mean = predict_value.mean(axis=1)
        
        # 创建完整的结果DataFrame
        result_df = df.copy()
        result_df['pred_open'] = np.nan
        result_df['pred_valid'] = False
        
        # 填充预测结果
        valid_indices = list(range(9, len(df) - lookahead))
        result_df.loc[valid_indices, 'pred_open'] = pred_mean[:len(valid_indices)]
        result_df.loc[valid_indices, 'pred_valid'] = True
        result_df = result_df[['datetime', 'open','pred_open', 'pred_valid']]
        
        # 计算统计信息
        stats = {
            'valid_predictions': len(valid_indices),
            'total_windows': n_total,
            'prediction_coverage': len(valid_indices) / len(df) * 100
        }
        
        return result_df, stats


        
def batch_predict(predictor_list, data, y_timestamp, lookahead: int = 5):
    """批量预测函数"""
    batch_predictions = []
        
    for predictor in predictor_list:
        x_df = data[['open', 'high', 'low', 'close', 'volume']]
            
        pred_df = predictor.predict(
                df=x_df,
                x_timestamp=data['timestamps'],
                y_timestamp=y_timestamp,
                pred_len=lookahead,
                T=1.0,
                top_p=0.9,
                sample_count=1
        )
        batch_predictions.append(pred_df.iloc[-1,0].values)
        
    return np.array(batch_predictions)
file_path =  "/root/private_data/KronosFT/bond-1m"
bond_name = os.listdir(file_path)
out_path = "./bond-1m-results"

for i in bond_name[0:100]:
    if i in os.listdir(out_path):
         continue
    file = os.path.join(file_path, i)
    temp_tuple = process_single_file(file_path = file, token_model_predictors = predictors_list)
    if temp_tuple['status'] == 'success':
        print(i[:-4], 'Completed')
        temp_df = temp_tuple['data']
        temp_df.to_csv(os.path.join(out_path, i), index=False)