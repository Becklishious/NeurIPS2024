"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_loading.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
"""

## Necessary Packages
import numpy as np


def MinMaxScaler(data):   
  """Min Max normalizer.
  最大最小归一化
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.min(data, 0)  #计算每列数据与最小值的差
  denominator = np.max(data, 0) - np.min(data, 0)  #最大值与最小值的差
  norm_data = numerator / (denominator + 1e-7)  #计算最小-最大归一化后的数据。为了避免除零错误，分母加上了一个小的正数（1e-7）
  return norm_data


def sine_data_generation (no, seq_len, dim):
  """Sine data generation.
  
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    
  Returns:
    - data: generated data
  """  
  # Initialize the output
  data = list()

  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)            
      phase = np.random.uniform(0, 0.1)
          
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)
        
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)
                
  return data
    

def real_data_loading (data_name, seq_len):
  """Load and preprocess real-world datasets.
  
  Args:
    - data_name: stock or energy
    - seq_len: sequence length
    
  Returns:
    - data: preprocessed data.
  """  
  #0407
  # 加载数据集
  #0409
  assert data_name in ['stock','energy','wind','match','matchgf','audio','test','load','pv','noiseload','inverter1']
  
  if data_name == 'stock':
    ori_data = np.loadtxt('data/stock_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'energy':
    ori_data = np.loadtxt('data/energy_data.csv', delimiter = ",",skiprows = 1)
    #0407风电数据
  elif data_name == 'wind':
    ori_data = np.loadtxt('data/test1.csv', delimiter = ",",skiprows = 1)
    #比赛光伏发电数据
  elif data_name == 'matchgf':
    ori_data = np.loadtxt('data/merged_data_GF.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'match':
    ori_data = np.loadtxt('data/merged_data.xlsx', delimiter = ",",skiprows = 1)
  elif data_name == 'audio':
    ori_data = np.loadtxt('data/audio.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'test':
    ori_data = np.loadtxt('data/His_Power_FD.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'load':
    ori_data = np.loadtxt('data/Load_dingxi.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'pv':
    ori_data = np.loadtxt('data/Pv_dingxi.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'noiseload':
    ori_data = np.loadtxt('data/Load_dingxi_noise.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'inverter1':
    ori_data = np.loadtxt('data/inverter1.csv', delimiter = ",",skiprows = 1)        
        
  # Flip the data to make chronological data 将数据翻转以使其成为时间顺序数据
  ori_data = ori_data[::-1]
  # # Normalize the data 归一化
  #这里将原代码中归一化操作注释掉了 为什么？
  # ori_data = MinMaxScaler(ori_data)
    
  # Preprocess the dataset
  temp_data = []    
  # Cut data by sequence length
  for i in range(0, len(ori_data) - seq_len):
    _x = ori_data[i:i + seq_len]
    temp_data.append(_x)
        
  # Mix the datasets (to make it similar to i.i.d) 混合数据集（使其类似于独立同分布）
  idx = np.random.permutation(len(temp_data))     #对数据列表进行随机排列的索引。
  data = []
  for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])
    
  return data