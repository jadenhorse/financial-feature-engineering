"""
时序特征生成器 — 滑动窗口统计特征

支持多窗口（1/3/6/12月）的统计特征生成，
包括均值、标准差、极值、趋势、变异系数等。
"""

import math
from typing import Optional


class TemporalFeatureGenerator:
    """时序特征生成器 — 从时间序列数据中提取统计特征"""

    DEFAULT_WINDOWS = [1, 3, 6, 12]

    def __init__(self, windows: Optional[list[int]] = None):
        self.windows = windows or self.DEFAULT_WINDOWS

    def compute_window_stats(self, values: list[float]) -> dict:
        """
        计算单个窗口的统计特征

        Args:
            values: 窗口内的数值列表

        Returns:
            包含均值、标准差、极值、趋势、变异系数的字典
        """
        if not values:
            return self._empty_stats()

        n = len(values)
        mean = sum(values) / n

        # 标准差
        if n > 1:
            variance = sum((x - mean) ** 2 for x in values) / (n - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0

        # 变异系数
        cv = std / mean if mean != 0 else 0.0

        # 趋势（线性回归斜率的简化版）
        if n > 1:
            x_mean = (n + 1) / 2
            x_var = sum((i + 1 - x_mean) ** 2 for i in range(n))
            xy_cov = sum((i + 1 - x_mean) * (values[i] - mean) for i in range(n))
            trend = xy_cov / x_var if x_var != 0 else 0.0
        else:
            trend = 0.0

        return {
            "mean": round(mean, 6),
            "std": round(std, 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "range": round(max(values) - min(values), 6),
            "trend": round(trend, 6),
            "cv": round(cv, 6),
            "count": n,
        }

    def generate(self, monthly_values) -> dict:
        """
        从月度数据生成多窗口时序特征

        Args:
            monthly_values: {月份序号: 数值} 的字典，或数值列表（按时间顺序）

        Returns:
            多窗口特征字典，如 {"mean_1": 100, "mean_3": 120, ...}
        """
        if not monthly_values:
            return {}

        # 支持list输入：自动转换为dict
        if isinstance(monthly_values, list):
            monthly_values = {i + 1: v for i, v in enumerate(monthly_values)}

        sorted_months = sorted(monthly_values.keys(), reverse=True)
        latest_month = sorted_months[0]
        features = {}

        for window in self.windows:
            start_month = latest_month - window + 1
            window_values = [
                monthly_values[m]
                for m in sorted_months
                if start_month <= m <= latest_month
            ]

            if not window_values:
                continue

            stats = self.compute_window_stats(window_values)

            features[f"mean_{window}"] = stats["mean"]
            features[f"std_{window}"] = stats["std"]
            features[f"min_{window}"] = stats["min"]
            features[f"max_{window}"] = stats["max"]
            features[f"range_{window}"] = stats["range"]
            features[f"trend_{window}"] = stats["trend"]
            features[f"cv_{window}"] = stats["cv"]

        # 环比变化率
        if len(sorted_months) >= 2:
            curr = monthly_values[sorted_months[0]]
            prev = monthly_values[sorted_months[1]]
            if prev != 0:
                features["amt_mom_rate"] = round((curr - prev) / abs(prev), 6)
            else:
                features["amt_mom_rate"] = 0.0

        return features

    @staticmethod
    def _empty_stats() -> dict:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "range": 0.0,
            "trend": 0.0,
            "cv": 0.0,
            "count": 0,
        }
