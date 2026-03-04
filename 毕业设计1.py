import pandas as pd
import numpy as np
np.random.seed(1000)
# 此处采用模拟200位用户的数据，现实应用可根据统计的真实数据进行计算
n_samples = 200
data = {
    '用户ID': [f'用户{i+1:02d}' for i in range(n_samples)],
    # 1. 静态易受骗系数：uniform生成均匀分布随机数
    '静态易受骗系数': np.round(np.random.uniform(60, 80, size=n_samples)),

    # 2. 设备安全状态系数：uniform生成均匀分布随机数
    '设备安全状态系数': np.round(np.random.uniform(60, 100, size=n_samples)),

    # 3. 访问频率风险系数：采用指数分布制造有偏数据，产生较多中等偏高值，平均值scale设置为40
    '访问频率风险系数': np.round(np.random.exponential(scale=40, size=n_samples)),

    # 4. 访问时长风险系数：采用指数分布制造有偏数据，产生较多中等偏高值，平均值scale设置为6，clip剪切0到12之间的数据
    '访问时长风险系数': np.clip(np.random.exponential(scale=6, size=n_samples),0,12),

    # 5. 交易频率风险系数：采用形状参数较小的帕累托分布，制造极端长尾、高变异，a越小极端数据越多
    '交易频率风险系数': np.round(np.random.pareto(a=10, size=n_samples)*200),

    # 6. 交易金额风险系数：采用形状参数较小的帕累托分布，制造极端长尾、高变异，a越小极端数据越多
    '交易金额风险系数': np.round(np.random.pareto(a=10, size=n_samples)*1000000) ,
}
# 把前面生成的模拟数据转化为一个结构化的表格DataFrame，并把“用户ID”这一列设置为这个表格的索引（行标签）
df = pd.DataFrame(data).set_index('用户ID')
print(df)
######一、熵权法计算步骤
###1.数据标准化
def data_normalization(df, positive_columns, negative_columns):
    """
    param df: 原始数据DataFrame
    param positive_columns: 正向指标列名列表
    param negative_columns: 负向指标列名列表
    return: 标准化后的DataFrame
    """
    df_normalized = df.copy()
    # 正向指标标准化： (x - min) / (max - min)
    for col in positive_columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
    # 负向指标标准化： (max - x) / (max - min)
    for col in negative_columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            df_normalized[col] = (max_val - df[col]) / (max_val - min_val)
    # 避免后续计算log(0)的错误，将0值替换为一个极小的正数
    df_normalized = df_normalized.replace(0, 1e-10)
    return df_normalized
# 以下是正向指标
positive_indices = ['静态易受骗系数', '访问频率风险系数', '访问时长风险系数', '交易频率风险系数', '交易金额风险系数']
# 设备安全状态系数是负向指标
negative_indices = ['设备安全状态系数']
# 进行标准化
df_normalized = data_normalization(df, positive_indices, negative_indices)
###2.计算标准化后的影响因子比重，构建比重矩阵P
P_matrix = df_normalized / df_normalized.sum(axis=0)
###3.计算信息熵值
def calculate_entropy(P_df):
    """
    param P_df: 比重矩阵 DataFrame
    return: 各指标的熵值 Series
    """
    m = len(P_df)  # 样本数量
    k = 1.0 / np.log(m)  # 计算k值
    entropy_vals = (-k * (P_df * np.log(P_df)).sum(axis=0))
    return entropy_vals
entropy_vals = calculate_entropy(P_matrix)
###4.计算信息效用值与权重
d_vals = 1 - entropy_vals
weights = d_vals / d_vals.sum()
calculated_weights = weights.values
print(weights)
######二、定义风险指数计算函数与判定等级
def calculate_risk_index(factor_scores, factor_weights):
    scores_array = np.array(factor_scores)
    weights_array = np.array(factor_weights)
    Y = np.dot(scores_array, weights_array)
    return Y
#根据风险指数数值定义风险等级
def determine_risk_level(risk_index):
    if 0 <= risk_index <= 25:
        return "低风险"
    elif 25 < risk_index <= 50:
        return "中风险"
    elif 50 < risk_index <= 75:
        return "高风险"
    elif 75 < risk_index:
        return "极高风险"
######三、交互式获取用户手动输入的评分
print("【用户风险预警评估系统】")
print("请根据提示，依次输入目标用户的各项风险因子评分")
print()

risk_factors = [
    "静态易受骗系数",
    "设备安全状态系数",
    "访问频率风险系数",
    "访问时长风险系数",
    "交易频率风险系数",
    "交易金额风险系数"
]

user_scores = []

score_input1 = input(f"请输入「静态易受骗系数」的评分: ").strip()
user_scores.append(int(score_input1))

score_input2 = input(f"请输入「设备安全状态系数」的评分: ").strip()
score_input2a = 100 - int(score_input2)
user_scores.append(int(score_input2a))

score_input3 = input(f"请输入「高风险网站平台访问频率」的次数: ").strip()
user_scores.append(int(score_input3))

score_input4 = input(f"请输入「高风险网站平台访问时长」的数值: ").strip() #按照5小时来衡量100分
score_input4a = float(score_input4)*20
user_scores.append(float(score_input4a))

score_input5 = input(f"请输入「高风险网站平台交易频率」的次数: ").strip()#按照现实经验，在高风险网站平台交易与受诈骗有很大的关联性，以5次折合100分
score_input5a = int(score_input5)*20
user_scores.append(int(score_input5a))

score_input6 = input(f"请输入「高风险网站平台交易总金额」的数值: ").strip() #以10000元折合成100分
score_input6a = int(score_input6)/100
user_scores.append(int(score_input6a))

print(user_scores)
print()
print("="*50)
print("输入完成，开始计算...")
print()

#计算并输出结果
user_risk_index = calculate_risk_index(user_scores, calculated_weights)
user_risk_level = determine_risk_level(user_risk_index)

print("【评估结果】")
for i, (factor, score) in enumerate(zip(risk_factors, user_scores)):
    print(f"  {factor}: {score}")
print(f"\n计算出的风险指数 Y = {user_risk_index:.2f}")
print(f"该用户的风险等级为: 【{user_risk_level}】")
print("="*50)