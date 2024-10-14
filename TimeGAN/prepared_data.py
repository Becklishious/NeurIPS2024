import pandas as pd
import math

# 读取 CSV 文件原数据和生成数据
file_path_ori = 'data/fortrain8000.csv'  # 替换为文件甲的路径
file_path_gen = 'gen_data/generated_data0416.csv'  # 替换为文件乙的路径

data_ori = pd.read_csv(file_path_ori)#8000

data_gen = pd.read_csv(file_path_gen)#10000


# 划分原数据的数据
test_len = data_ori.shape[0] * 0.2
test_len = math.floor(test_len)
print(test_len)
train_data_ori = data_ori[:-test_len]
test_data_ori = data_ori[-test_len:]
# # train_data_ori, test_data_ori = train_test_split(data_ori, test_size=0.2, random_state=42)

# 合并文件甲数据和文件乙数据作为训练集
#train_data = pd.concat([train_data_ori, data_gen], ignore_index=True)

train_data = pd.concat([data_ori, data_gen], ignore_index=True)

test_data = test_data_ori


# 保存训练集和测试集到 CSV 文件
train_data.to_csv('gen_data/train_data0416.csv', index=False)#18000
test_data.to_csv('gen_data/test_data.csv', index=False)
