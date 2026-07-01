"""
特征评估报告生成器

自动生成特征分布、区分度、稳定性全量报告，
支持文本和JSON格式输出。
"""

import json
from datetime import datetime
from typing import Optional

from .cross import iv_strength
from .selector import FeatureSelector, compute_psi


class FeatureReportGenerator:
    """特征评估报告生成器"""

    def __init__(self):
        self.report_data = {}

    def generate(
        self,
        iv_dict: dict[str, float],
        psi_dict: Optional[dict[str, float]] = None,
        corr_matrix: Optional[dict[str, dict[str, float]]] = None,
        feature_descriptions: Optional[dict[str, str]] = None,
    ) -> dict:
        """
        生成特征评估报告

        Args:
            iv_dict: {特征名: IV值}
            psi_dict: {特征名: PSI值}（可选）
            corr_matrix: 相关系数矩阵（可选）
            feature_descriptions: {特征名: 业务描述}（可选）

        Returns:
            完整报告字典
        """
        descriptions = feature_descriptions or {}

        # 特征明细
        feature_details = []
        for name, iv in sorted(iv_dict.items(), key=lambda x: x[1], reverse=True):
            detail = {
                "name": name,
                "iv": iv,
                "iv_strength": iv_strength(iv),
                "description": descriptions.get(name, ""),
            }
            if psi_dict and name in psi_dict:
                detail["psi"] = psi_dict[name]
                detail["stability"] = FeatureSelector.psi_stability_level(psi_dict[name])
            feature_details.append(detail)

        # 高相关特征对
        high_corr_pairs = []
        if corr_matrix:
            seen = set()
            for feat_a, corrs in corr_matrix.items():
                for feat_b, corr in corrs.items():
                    pair = tuple(sorted([feat_a, feat_b]))
                    if pair not in seen and abs(corr) > 0.7 and feat_a != feat_b:
                        high_corr_pairs.append({
                            "feature_a": pair[0],
                            "feature_b": pair[1],
                            "correlation": corr,
                        })
                        seen.add(pair)

        # 统计摘要
        iv_values = list(iv_dict.values())
        summary = {
            "total_features": len(iv_dict),
            "strong_features": sum(1 for iv in iv_values if iv >= 0.3),
            "medium_features": sum(1 for iv in iv_values if 0.1 <= iv < 0.3),
            "weak_features": sum(1 for iv in iv_values if 0.02 <= iv < 0.1),
            "useless_features": sum(1 for iv in iv_values if iv < 0.02),
            "avg_iv": round(sum(iv_values) / len(iv_values), 4) if iv_values else 0,
            "max_iv": round(max(iv_values), 4) if iv_values else 0,
            "high_corr_pairs": len(high_corr_pairs),
        }

        if psi_dict:
            psi_values = list(psi_dict.values())
            summary["stable_features"] = sum(1 for p in psi_values if p < 0.1)
            summary["unstable_features"] = sum(1 for p in psi_values if p >= 0.25)

        report = {
            "report_time": datetime.now().isoformat(),
            "summary": summary,
            "feature_details": feature_details,
            "high_correlation_pairs": high_corr_pairs,
        }

        self.report_data = report
        return report

    def to_text(self, report: Optional[dict] = None) -> str:
        """将报告转换为可读文本格式"""
        data = report or self.report_data
        if not data:
            return "No report data available."

        lines = []
        lines.append("=" * 60)
        lines.append("  特征评估报告")
        lines.append(f"  生成时间: {data['report_time']}")
        lines.append("=" * 60)

        # 摘要
        s = data["summary"]
        lines.append(f"\n📊 摘要")
        lines.append(f"  总特征数: {s['total_features']}")
        lines.append(f"  强预测力(IV≥0.3): {s['strong_features']}")
        lines.append(f"  中预测力(0.1≤IV<0.3): {s['medium_features']}")
        lines.append(f"  弱预测力(0.02≤IV<0.1): {s['weak_features']}")
        lines.append(f"  无预测力(IV<0.02): {s['useless_features']}")
        lines.append(f"  平均IV: {s['avg_iv']}")
        lines.append(f"  最高IV: {s['max_iv']}")

        if "stable_features" in s:
            lines.append(f"  稳定特征(PSI<0.1): {s['stable_features']}")
            lines.append(f"  不稳定特征(PSI≥0.25): {s['unstable_features']}")

        # Top特征
        lines.append(f"\n🏆 Top特征 (按IV排序)")
        for i, feat in enumerate(data["feature_details"][:10], 1):
            line = f"  {i}. {feat['name']} — IV={feat['iv']:.4f} ({feat['iv_strength']})"
            if "psi" in feat:
                line += f" | PSI={feat['psi']:.4f} ({feat['stability']})"
            if feat["description"]:
                line += f" | {feat['description']}"
            lines.append(line)

        # 高相关特征对
        if data["high_correlation_pairs"]:
            lines.append(f"\n⚠️ 高相关特征对 (|r|>0.7)")
            for pair in data["high_correlation_pairs"]:
                lines.append(
                    f"  {pair['feature_a']} ↔ {pair['feature_b']} "
                    f"(r={pair['correlation']:.4f})"
                )

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def to_json(self, report: Optional[dict] = None, filepath: Optional[str] = None) -> str:
        """将报告导出为JSON"""
        data = report or self.report_data
        json_str = json.dumps(data, ensure_ascii=False, indent=2)

        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)

        return json_str
