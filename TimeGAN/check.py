import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('data/RF/train1-2.csv')

# 检查数据中是否存在 NaN 或无穷大值
if df.isnull().values.any():
    print("数据中存在缺失值 (NaN)")
    
# 标识缺失值的位置
missing_values = df.isnull()

# 检查是否有缺失值
if missing_values.values.any():
    # 找到包含缺失值的行和列
    rows_with_missing = missing_values.any(axis=1)
    cols_with_missing = missing_values.any(axis=0)

    print("文件中的缺失值位置：")
    print("行索引：")
    print(df[rows_with_missing].index.tolist())
    print("列索引：")
    print(df.columns[cols_with_missing].tolist())
else:
    print("文件中没有缺失值。")



if df.isin([float('inf'), -float('inf')]).values.any():
    print("数据中存在无穷大值")

# 检查数据类型是否为 float32 并且值是否过大
if df.dtypes.eq('float32').any():
    max_float32 = df.astype('float32').max()
    if (max_float32 == float('inf')).any():
        print("数据中的某些值过大，超出了 float32 的范围")

# 如果需要，你还可以进一步检查其他数据质量问题

# 显示数据前几行以便检查数据整体情况
# print("前几行数据：")
# print(df.head())