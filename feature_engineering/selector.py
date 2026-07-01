"""
特征选择器 — 三阶段特征筛选

IV过滤 → 相关性去冗余 → PSI稳定性评估
确保上线特征的质量和可解释性。
"""

import math
from typing import Optional


class FeatureSelector:
    """三阶段特征选择器"""

    def __init__(
        self,
        min_iv: float = 0.02,
        max_corr: float = 0.7,
        max_psi: float = 0.25,
    ):
        """
        Args:
            min_iv: IV最低阈值（默认0.02，弱预测力下限）
            max_corr: 最大允许相关系数（默认0.7）
            max_psi: 最大允许PSI值（默认0.25，稳定性阈值）
        """
        self.min_iv = min_iv
        self.max_corr = max_corr
        self.max_psi = max_psi

    def filter_by_iv(self, iv_dict: dict[str, float]) -> list[str]:
        """
        第一阶段：IV过滤

        Args:
            iv_dict: {特征名: IV值} 的字典

        Returns:
            通过IV过滤的特征名列表
        """
        return [name for name, iv in iv_dict.items() if iv >= self.min_iv]

    def remove_correlated(
        self,
        features: list[str],
        corr_matrix: dict[str, dict[str, float]],
        iv_dict: Optional[dict[str, float]] = None,
    ) -> list[str]:
        """
        第二阶段：相关性去冗余

        当两个特征相关性超过阈值时，保留IV更高的特征。

        Args:
            features: 待筛选的特征列表
            corr_matrix: 相关系数矩阵 {feat1: {feat2: corr}}
            iv_dict: IV值字典，用于决定保留哪个特征

        Returns:
            去冗余后的特征列表
        """
        selected = list(features)
        removed = set()

        for i, feat_a in enumerate(features):
            if feat_a in removed:
                continue
            for feat_b in features[i + 1 :]:
                if feat_b in removed:
                    continue

                corr = corr_matrix.get(feat_a, {}).get(feat_b, 0)
                if abs(corr) > self.max_corr:
                    # 保留IV更高的特征
                    if iv_dict:
                        iv_a = iv_dict.get(feat_a, 0)
                        iv_b = iv_dict.get(feat_b, 0)
                        drop = feat_b if iv_a >= iv_b else feat_a
                    else:
                        # 无IV信息时保留排在前面的
                        drop = feat_b

                    removed.add(drop)
                    if drop in selected:
                        selected.remove(drop)

        return selected

    def filter_by_psi(
        self,
        psi_dict: dict[str, float],
        features: Optional[list[str]] = None,
    ) -> list[str]:
        """
        第三阶段：PSI稳定性过滤

        PSI < 0.1: 稳定
        0.1 <= PSI < 0.25: 轻微不稳定
        PSI >= 0.25: 不稳定，应移除

        Args:
            psi_dict: {特征名: PSI值} 的字典
            features: 待筛选的特征列表（默认使用psi_dict的所有key）

        Returns:
            通过PSI过滤的特征名列表
        """
        if features is None:
            features = list(psi_dict.keys())

        return [
            name for name in features
            if name in psi_dict and psi_dict[name] < self.max_psi
        ]

    def select(
        self,
        iv_dict: dict[str, float],
        corr_matrix: Optional[dict[str, dict[str, float]]] = None,
        psi_dict: Optional[dict[str, float]] = None,
    ) -> list[str]:
        """
        完整三阶段特征选择

        Args:
            iv_dict: IV值字典
            corr_matrix: 相关系数矩阵（可选）
            psi_dict: PSI值字典（可选）

        Returns:
            最终选中的特征列表
        """
        # 阶段1: IV过滤
        selected = self.filter_by_iv(iv_dict)

        # 阶段2: 相关性去冗余
        if corr_matrix:
            selected = self.remove_correlated(selected, corr_matrix, iv_dict)

        # 阶段3: PSI稳定性
        if psi_dict:
            selected = self.filter_by_psi(psi_dict, selected)

        return selected

    @staticmethod
    def psi_stability_level(psi: float) -> str:
        """判断PSI稳定性等级"""
        if psi < 0.1:
            return "稳定"
        elif psi < 0.25:
            return "轻微不稳定"
        else:
            return "不稳定"


def compute_psi(
    expected: list[float],
    actual: list[float],
    n_bins: int = 10,
) -> float:
    """
    计算PSI（Population Stability Index）

    Args:
        expected: 期望分布（训练集）的值
        actual: 实际分布（测试集/线上）的值
        n_bins: 分箱数量

    Returns:
        PSI值
    """
    if not expected or not actual:
        return 0.0

    # 基于expected的分箱边界
    sorted_exp = sorted(expected)
    bin_edges = []
    for i in range(1, n_bins):
        idx = int(len(sorted_exp) * i / n_bins)
        bin_edges.append(sorted_exp[min(idx, len(sorted_exp) - 1)])

    # 添加首尾
    all_values = sorted_exp + sorted(actual)
    bin_edges = [min(all_values) - 0.001] + bin_edges + [max(all_values) + 0.001]

    # 计算各箱占比
    def get_proportions(values, edges):
        counts = [0] * (len(edges) - 1)
        for v in values:
            for i in range(len(edges) - 1):
                if edges[i] <= v < edges[i + 1]:
                    counts[i] += 1
                    break
        total = sum(counts)
        return [c / total if total > 0 else 0 for c in counts]

    exp_props = get_proportions(expected, bin_edges)
    act_props = get_proportions(actual, bin_edges)

    # 计算PSI
    psi = 0.0
    for ep, ap in zip(exp_props, act_props):
        ep = max(ep, 0.0001)
        ap = max(ap, 0.0001)
        psi += (ap - ep) * math.log(ap / ep)

    return round(psi, 6)
