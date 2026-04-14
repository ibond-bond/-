# app.py
import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件最大16MB

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------- 全局变量，存储通过上传数据计算出的权重 ----------
calculated_weights = None   # 一开始没有权重


def calculate_entropy_weights(df):
    """输入原始数据DataFrame，返回权重列表（顺序与 ordered_columns 一致）"""
    # 固定列顺序（与期望显示顺序一致）
    ordered_columns = [
        '静态易受骗系数',
        '设备安全状态系数',
        '访问频率风险系数',
        '访问时长风险系数',
        '交易频率风险系数',
        '交易金额风险系数'
    ]
    positive_cols = ['静态易受骗系数', '访问频率风险系数', '访问时长风险系数',
                     '交易频率风险系数', '交易金额风险系数']
    negative_cols = ['设备安全状态系数']

    # 检查列是否存在
    for col in ordered_columns:
        if col not in df.columns:
            raise ValueError(f"Excel中缺少必要的列：{col}")

    # 按固定顺序选取并转换为数值类型
    df = df[ordered_columns].copy()
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()  # 删除含有无效值的行

    # 数据标准化（与原脚本一致）
    df_norm = df.copy()
    for col in positive_cols:
        min_v = df[col].min()
        max_v = df[col].max()
        if max_v == min_v:
            df_norm[col] = 0  # 若所有值相等，标准化为0
        else:
            df_norm[col] = (df[col] - min_v) / (max_v - min_v)
    for col in negative_cols:
        min_v = df[col].min()
        max_v = df[col].max()
        if max_v == min_v:
            df_norm[col] = 0
        else:
            df_norm[col] = (max_v - df[col]) / (max_v - min_v)

    # 避免 log(0) 错误
    df_norm = df_norm.replace(0, 1e-10)

    # 计算比重矩阵
    P = df_norm / df_norm.sum(axis=0)

    # 计算熵值
    m = len(P)
    k = 1.0 / np.log(m)
    entropy = (-k * (P * np.log(P)).sum(axis=0))

    # 计算权重
    d = 1 - entropy
    weights = d / d.sum()

    # 返回权重列表（顺序与 ordered_columns 一致）
    return weights.values.tolist(), ordered_columns

def calculate_risk_index(scores, w):
    return np.dot(np.array(scores), np.array(w))


def determine_risk_level(risk_index):
    if risk_index <= 25:
        return "低风险"
    elif risk_index <= 50:
        return "中风险"
    elif risk_index <= 75:
        return "高风险"
    else:
        return "极高风险"


# ---------- 页面路由 ----------
@app.route('/')
def index():
    return render_template('index.html')


# ---------- 文件上传与权重计算 ----------
@app.route('/upload', methods=['POST'])
def upload_file():
    global calculated_weights
    if 'file' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    if not file.filename.endswith(('.xlsx', '.xls')):
        return jsonify({'error': '请上传Excel文件（.xlsx 或 .xls）'}), 400

    try:
        # 保存上传的文件
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # 使用 pandas 读取 Excel
        df = pd.read_excel(filepath)

        # 计算权重
        weights, col_names = calculate_entropy_weights(df)
        calculated_weights = weights

        # 返回成功信息，并附带计算出的权重
        # required_cols 的顺序（与 calculate_entropy_weights 中一致）
        required_cols = [
            '静态易受骗系数',
            '访问频率风险系数',
            '访问时长风险系数',
            '交易频率风险系数',
            '交易金额风险系数',
            '设备安全状态系数'
        ]
        # 期望的显示顺序
        display_order = [
            '静态易受骗系数',
            '设备安全状态系数',
            '访问频率风险系数',
            '访问时长风险系数',
            '交易频率风险系数',
            '交易金额风险系数'
        ]

        # 先构建名称到权重的字典
        weight_dict = dict(zip(required_cols, weights))

        # 构建有序数组返回前端
        weights_ordered = []
        for i, name in enumerate(col_names):
            weights_ordered.append({
                'name': name,
                'value': round(weights[i], 6)  # 保留6位小数以便对比
            })
        return jsonify({
            'success': True,
            'message': '数据训练完成，权重已更新。',
            'weights': weights_ordered  # 注意这里是数组，不是对象
        })
    except Exception as e:
        return jsonify({'error': f'处理文件时出错：{str(e)}'}), 500


# ---------- 风险评估接口 ----------
@app.route('/calculate', methods=['POST'])
def calculate():
    global calculated_weights
    if calculated_weights is None:
        return jsonify({'error': '请先上传数据训练模型！'}), 400

    try:
        data = request.get_json()
        # 前端传来的原始值
        raw_score1 = float(data['static'])
        raw_score2 = 100 - float(data['device'])   # 设备安全负向转换
        raw_score3 = float(data['freq_visit'])
        raw_score4 = float(data['duration']) * 20
        raw_score5 = float(data['freq_trade']) * 20
        raw_score6 = float(data['amount']) / 100

        user_scores = [raw_score1, raw_score2, raw_score3,
                       raw_score4, raw_score5, raw_score6]

        risk_index = calculate_risk_index(user_scores, calculated_weights)
        risk_level = determine_risk_level(risk_index)

        return jsonify({
            'risk_index': round(risk_index, 2),
            'risk_level': risk_level
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)