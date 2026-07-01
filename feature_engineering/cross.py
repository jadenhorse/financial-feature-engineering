"""
交叉特征编码器 — WOE/IV编码与特征交叉

提供WOE（Weight of Evidence）编码和IV（Information Value）评估，
支持类别变量与数值变量的自动交叉特征生成。
"""

import math
from typing import Optional


def calculate_woe_iv(
    labels: list[int],
    values: list,
    n_bins: int = 10,
    min_samples: int = 30,
) -> tuple[dict, float]:
    """
    计算WOE编码和IV值

    Args:
        labels: 二分类标签列表 (0=好客户, 1=坏客户)
        values: 对应的特征值列表
        n_bins: 分箱数量
        min_samples: 每箱最小样本数

    Returns:
        (woe_dict, iv_value) — WOE编码字典和IV值
    """
    if len(labels) != len(values):
        raise ValueError("labels和values长度不一致")

    total_good = sum(1 for y in labels if y == 0)
    total_bad = sum(1 for y in labels if y == 1)

    if total_good == 0 or total_bad == 0:
        return {}, 0.0

    # 判断数值型还是类别型
    numeric_values = []
    is_numeric = True
    for v in values:
        try:
            numeric_values.append(float(v))
        except (ValueError, TypeError):
            is_numeric = False
            break

    if is_numeric:
        bins = _bin_numeric(numeric_values, labels, n_bins, min_samples)
    else:
        bins = _bin_categorical(values, labels, min_samples)

    # 计算WOE和IV
    woe_dict = {}
    iv_total = 0.0

    for bin_key, (good_count, bad_count) in bins.items():
        good_rate = good_count / total_good if total_good > 0 else 0.0001
        bad_rate = bad_count / total_bad if total_bad > 0 else 0.0001

        # 防止除零
        good_rate = max(good_rate, 0.0001)
        bad_rate = max(bad_rate, 0.0001)

        woe = math.log(good_rate / bad_rate)
        iv = (good_rate - bad_rate) * woe

        woe_dict[bin_key] = round(woe, 6)
        iv_total += iv

    return woe_dict, round(iv_total, 6)


def _bin_numeric(
    values: list[float],
    labels: list[int],
    n_bins: int,
    min_samples: int,
) -> dict:
    """等频分箱"""
    paired = sorted(zip(values, labels), key=lambda x: x[0])
    bin_size = max(len(paired) // n_bins, 1)

    bins = {}
    for i in range(0, len(paired), bin_size):
        chunk = paired[i : i + bin_size]
        bin_min = chunk[0][0]
        bin_max = chunk[-1][0]
        bin_key = f"[{bin_min:.2f}, {bin_max:.2f})"

        good = sum(1 for _, y in chunk if y == 0)
        bad = sum(1 for _, y in chunk if y == 1)
        bins[bin_key] = (good, bad)

    return bins


def _bin_categorical(
    values: list,
    labels: list[int],
    min_samples: int,
) -> dict:
    """类别分箱"""
    bins = {}
    for v, y in zip(values, labels):
        key = str(v)
        if key not in bins:
            bins[key] = (0, 0)
        good, bad = bins[key]
        if y == 0:
            bins[key] = (good + 1, bad)
        else:
            bins[key] = (good, bad + 1)

    # 合并小样本箱
    merged = {}
    overflow_good, overflow_bad = 0, 0
    for key, (good, bad) in bins.items():
        if good + bad < min_samples:
            overflow_good += good
            overflow_bad += bad
        else:
            merged[key] = (good, bad)

    if overflow_good + overflow_bad > 0:
        merged["__other__"] = (overflow_good, overflow_bad)

    return merged


class CrossFeatureEncoder:
    """交叉特征编码器 — 类别×数值自动交叉"""

    def __init__(self, min_iv: float = 0.02):
        """
        Args:
            min_iv: IV过滤阈值，低于此值的交叉特征将被过滤
        """
        self.min_iv = min_iv
        self.woe_mappings: dict[str, tuple[dict, float]] = {}

    def fit(
        self,
        data: list[dict],
        cat_cols: list[str],
        num_cols: list[str],
        label_col: str = "label",
    ) -> "CrossFeatureEncoder":
        """
        拟合交叉特征编码

        Args:
            data: 数据列表，每个元素为字典
            cat_cols: 类别变量列名
            num_cols: 数值变量列名
            label_col: 标签列名

        Returns:
            self（支持链式调用）
        """
        labels = [row[label_col] for row in data]

        for cat_col in cat_cols:
            cat_values = [str(row.get(cat_col, "")) for row in data]
            woe_dict, iv = calculate_woe_iv(labels, cat_values)
            if iv >= self.min_iv:
                self.woe_mappings[cat_col] = (woe_dict, iv)

            for num_col in num_cols:
                num_values = [row.get(num_col, 0) for row in data]
                cross_name = f"{cat_col}_{num_col}"
                # 交叉特征：类别分组内的数值WOE编码
                cross_values = [
                    f"{cat}_{num}" for cat, num in zip(cat_values, num_values)
                ]
                woe_dict, iv = calculate_woe_iv(labels, cross_values, n_bins=5)
                if iv >= self.min_iv:
                    self.woe_mappings[cross_name] = (woe_dict, iv)

        return self

    def transform(self, data: list[dict]) -> list[dict]:
        """
        将WOE编码应用于数据

        Args:
            data: 待转换的数据列表

        Returns:
            添加了WOE编码特征的数据列表
        """
        result = []
        for row in data:
            new_row = dict(row)
            for feat_name, (woe_dict, _) in self.woe_mappings.items():
                # 简化：使用默认WOE值0
                woe_value = 0.0
                for bin_key, woe_val in woe_dict.items():
                    # 精确匹配或范围匹配
                    if feat_name in row:
                        if str(row[feat_name]) in bin_key or str(row[feat_name]) == bin_key:
                            woe_value = woe_val
                            break
                new_row[f"{feat_name}_woe"] = woe_value
            result.append(new_row)
        return result

    def get_iv_summary(self) -> dict[str, float]:
        """获取所有特征的IV值摘要"""
        return {name: iv for name, (_, iv) in self.woe_mappings.items()}


def iv_strength(iv: float) -> str:
    """判断IV值的预测力强度"""
    if iv < 0.02:
        return "无预测力"
    elif iv < 0.1:
        return "弱预测力"
    elif iv < 0.3:
        return "中预测力"
    else:
        return "强预测力"
