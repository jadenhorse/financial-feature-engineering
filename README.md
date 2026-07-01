# Financial Feature Engineering Toolkit 📊

> **金融风控特征工程工具包** — 将原始金融数据转化为高区分度风控特征，覆盖时序、交叉、行为、文本四大特征族，支持自动化特征选择与评估。

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 业务背景

在信贷风控建模中，特征工程决定了模型上限。本工具包将百融云创SaaS平台中反复验证的特征工程经验系统化，提供从原始数据到生产级特征的完整管线，将特征开发周期从**周级缩短到小时级**。

## ✨ 核心功能

| 功能 | 描述 |
|------|------|
| 📈 时序特征族 | 滑动窗口统计（均值/极值/趋势/波动率），支持1/3/6/12月窗口 |
| 🔀 交叉特征族 | 类别×数值交叉编码，自动生成WOE/IV评估 |
| 🧠 行为特征族 | RFM模型 + 行为序列编码（最近/频率/金额/衰减） |
| 📝 文本特征族 | 关键词TF-IDF + 语义标签映射，零依赖实现 |
| 🏆 特征选择器 | IV过滤 + 相关性去冗余 + 稳定性评估（PSI）三阶段筛选 |
| 📊 特征评估报告 | 自动生成特征分布、区分度、稳定性全量报告 |

## 📐 系统架构

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  原始数据层   │ →  │  特征生成层   │ →  │  特征筛选层   │ →  │  输出交付层   │
│              │    │              │    │              │    │              │
│ · 交易流水   │    │ · 时序统计   │    │ · IV过滤     │    │ · 特征报告   │
│ · 行为日志   │    │ · 交叉编码   │    │ · 相关性去冗 │    │ · WOE字典    │
│ · 文本数据   │    │ · RFM计算    │    │ · PSI稳定性  │    │ · 模型就绪DF │
│ · 标签数据   │    │ · 文本映射   │    │ · Top-K排序  │    │ · 上线清单   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

## 🚀 快速开始

```bash
pip install -r requirements.txt

# 运行示例
python -m feature_engineering.run_example

# 或在Jupyter中交互探索
jupyter notebook notebooks/feature_exploration.ipynb
```

## 📊 特征族详解

### 1. 时序特征族 (Temporal Features)

```
输入: 用户月度交易流水
输出: 多窗口统计特征

窗口: 1月 / 3月 / 6月 / 12月
统计: mean, std, min, max, trend, cv
衍生: 环比变化率, 同比变化率, 波动率指数
```

### 2. 交叉特征族 (Cross Features)

```
输入: 类别变量 + 数值变量
输出: WOE编码交叉特征

方法: 类别×数值 → 分箱 → WOE计算 → IV评估
阈值: IV > 0.02 (弱), > 0.1 (中), > 0.3 (强)
```

### 3. 行为特征族 (Behavioral Features)

```
输入: 用户行为日志（登录/点击/申请）
输出: RFM + 序列特征

R(最近): 距今天数
F(频率): 窗口内次数
M(金额): 窗口内总额
衰减: exp(-λ·Δt) 加权
```

### 4. 文本特征族 (Text Features)

```
输入: 非结构化文本（新闻/公告/备注）
输出: 关键词特征 + 语义标签

方法: 领域词典匹配 + TF-IDF权重
标签: 融资/风险/中标/负面/正面
```

## 🧪 测试

```bash
python -m pytest tests/ -v
```

覆盖：时序特征计算、WOE/IV编码、RFM计算、特征选择管线、输出schema。

## 📁 项目结构

```
financial-feature-engineering/
├── feature_engineering/
│   ├── __init__.py
│   ├── temporal.py          # 时序特征生成器
│   ├── cross.py             # 交叉特征 + WOE/IV编码
│   ├── behavioral.py        # RFM + 行为序列特征
│   ├── text_features.py     # 文本关键词 + 语义标签
│   ├── selector.py          # 三阶段特征选择器
│   └── report.py            # 特征评估报告生成
├── tests/
│   └── test_features.py     # 单元测试
├── examples/
│   └── sample_features.json # 示例输出
├── notebooks/
│   └── feature_exploration.ipynb
└── requirements.txt
```

## 🔧 技术亮点

1. **纯Python实现** — 核心特征计算零外部依赖，pandas/numpy可选加速
2. **风控领域知识编码** — IV/PSI/WOE等金融风控标准指标内置
3. **三阶段特征筛选** — IV过滤→相关性去冗→PSI稳定性，确保上线质量
4. **可解释性优先** — 每个特征附带业务含义说明和区分度评估

## 👤 适用场景

- 银行信贷风控模型特征工程
- 保险理赔欺诈检测特征开发
- 供应链金融风险评估
- 金融数据产品特征平台搭建

## 📄 License

MIT
