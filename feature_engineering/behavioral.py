"""
RFM特征生成器 — 行为序列特征

基于RFM（Recency/Frequency/Monetary）模型，
支持指数衰减加权和多窗口行为特征生成。
"""

import math
from datetime import datetime, timedelta
from typing import Optional


class RFMFeatureGenerator:
    """RFM行为特征生成器 — 从用户行为日志中提取RFM及衍生特征"""

    def __init__(self, decay_lambda: float = 0.1, windows: Optional[list[int]] = None):
        """
        Args:
            decay_lambda: 时间衰减系数，越大衰减越快
            windows: 统计窗口列表（天数），默认[7, 30, 90, 180]
        """
        self.decay_lambda = decay_lambda
        self.windows = windows or [7, 30, 90, 180]

    def compute_rfm(
        self,
        events: list[dict],
        reference_date: Optional[str] = None,
    ) -> dict:
        """
        计算单个用户的RFM特征

        Args:
            events: 行为事件列表，每个事件包含 date, amount 字段
            reference_date: 参考日期（默认今天）

        Returns:
            RFM特征字典
        """
        if not events:
            return self._empty_rfm()

        ref = (
            datetime.strptime(reference_date, "%Y-%m-%d")
            if reference_date
            else datetime.now()
        )

        # 按日期排序
        sorted_events = sorted(events, key=lambda e: e.get("date", ""), reverse=True)
        latest_date_str = sorted_events[0].get("date", "")
        if not latest_date_str:
            return self._empty_rfm()

        latest_date = datetime.strptime(latest_date_str, "%Y-%m-%d")
        recency = (ref - latest_date).days

        amounts = [e.get("amount", 0) for e in sorted_events]
        frequency = len(events)
        monetary = sum(amounts)

        # 指数衰减加权金额
        decay_amounts = []
        for e in sorted_events:
            event_date = datetime.strptime(e.get("date", ref.strftime("%Y-%m-%d")), "%Y-%m-%d")
            delta_days = (ref - event_date).days
            decay_weight = math.exp(-self.decay_lambda * delta_days / 30)
            decay_amounts.append(e.get("amount", 0) * decay_weight)

        decay_monetary = sum(decay_amounts)

        # 平均金额
        avg_amount = monetary / frequency if frequency > 0 else 0

        # 金额标准差
        if frequency > 1:
            variance = sum((a - avg_amount) ** 2 for a in amounts) / (frequency - 1)
            amount_std = math.sqrt(variance)
        else:
            amount_std = 0.0

        features = {
            "recency_days": recency,
            "frequency": frequency,
            "monetary_total": round(monetary, 2),
            "monetary_avg": round(avg_amount, 2),
            "monetary_std": round(amount_std, 2),
            "monetary_decay": round(decay_monetary, 2),
            "monetary_max": round(max(amounts), 2) if amounts else 0,
            "monetary_min": round(min(amounts), 2) if amounts else 0,
        }

        # 多窗口行为特征
        for window in self.windows:
            window_events = [
                e
                for e in sorted_events
                if (ref - datetime.strptime(e.get("date", ref.strftime("%Y-%m-%d")), "%Y-%m-%d")).days
                <= window
            ]
            w_freq = len(window_events)
            w_amounts = [e.get("amount", 0) for e in window_events]
            w_total = sum(w_amounts)

            features[f"freq_{window}d"] = w_freq
            features[f"amt_{window}d"] = round(w_total, 2)
            features[f"amt_avg_{window}d"] = round(w_total / w_freq, 2) if w_freq > 0 else 0

        return features

    def batch_compute(
        self,
        user_events: dict[str, list[dict]],
        reference_date: Optional[str] = None,
    ) -> dict[str, dict]:
        """
        批量计算多用户RFM特征

        Args:
            user_events: {user_id: [events]} 的字典
            reference_date: 参考日期

        Returns:
            {user_id: {rfm_features}} 的字典
        """
        results = {}
        for user_id, events in user_events.items():
            results[user_id] = self.compute_rfm(events, reference_date)
        return results

    @staticmethod
    def _empty_rfm() -> dict:
        return {
            "recency_days": -1,
            "frequency": 0,
            "monetary_total": 0.0,
            "monetary_avg": 0.0,
            "monetary_std": 0.0,
            "monetary_decay": 0.0,
            "monetary_max": 0.0,
            "monetary_min": 0.0,
        }
