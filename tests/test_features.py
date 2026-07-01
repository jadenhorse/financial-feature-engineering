# Financial Feature Engineering - Unit Tests
"""
特征工程工具包单元测试：时序特征、交叉特征、行为特征、文本特征、特征选择、报告生成
"""

import unittest
import math

from feature_engineering.temporal import TemporalFeatureGenerator
from feature_engineering.cross import CrossFeatureEncoder, calculate_woe_iv, iv_strength
from feature_engineering.behavioral import RFMFeatureGenerator
from feature_engineering.text_features import TextFeatureExtractor
from feature_engineering.selector import FeatureSelector, compute_psi
from feature_engineering.report import FeatureReportGenerator


class TestTemporalFeatureGenerator(unittest.TestCase):
    """时序特征生成器测试"""

    def setUp(self):
        self.gen = TemporalFeatureGenerator(windows=[1, 3, 6])

    def test_basic_features(self):
        values = [100, 120, 110, 130, 125, 140, 150]
        features = self.gen.generate(values)
        self.assertIn("mean_1", features)
        self.assertIn("std_3", features)
        self.assertIn("max_6", features)
        self.assertIn("trend_6", features)

    def test_empty_values(self):
        features = self.gen.generate([])
        self.assertEqual(features, {})

    def test_single_value(self):
        features = self.gen.generate([100])
        self.assertIn("mean_1", features)
        self.assertAlmostEqual(features["mean_1"], 100.0)

    def test_cv_calculation(self):
        values = [100, 200, 300]
        features = self.gen.generate(values)
        # CV = std/mean
        self.assertIn("cv_3", features)
        self.assertGreater(features["cv_3"], 0)


class TestCrossFeatureEncoder(unittest.TestCase):
    """交叉特征编码器 + WOE/IV测试"""

    def test_woe_iv_basic(self):
        # 简单二分类数据
        labels = [1, 0, 1, 0]
        values = ["A", "A", "B", "B"]
        woe_dict, iv = calculate_woe_iv(labels, values)
        self.assertIsInstance(woe_dict, dict)
        self.assertIsInstance(iv, float)

    def test_iv_strength(self):
        self.assertEqual(iv_strength(0.01), "无预测力")
        self.assertEqual(iv_strength(0.05), "弱预测力")
        self.assertEqual(iv_strength(0.15), "中预测力")
        self.assertEqual(iv_strength(0.5), "强预测力")

    def test_encoder_transform(self):
        encoder = CrossFeatureEncoder(min_iv=0.01)
        data = [
            {"cat": "A", "num": 10, "label": 0},
            {"cat": "A", "num": 20, "label": 0},
            {"cat": "A", "num": 30, "label": 0},
            {"cat": "B", "num": 40, "label": 1},
            {"cat": "B", "num": 50, "label": 0},
            {"cat": "B", "num": 60, "label": 1},
            {"cat": "C", "num": 70, "label": 1},
            {"cat": "C", "num": 80, "label": 1},
            {"cat": "C", "num": 90, "label": 1},
        ]
        encoder.fit(data, cat_cols=["cat"], num_cols=["num"])
        transformed = encoder.transform(data)
        self.assertEqual(len(transformed), 9)


class TestRFMFeatureGenerator(unittest.TestCase):
    """RFM行为特征生成器测试"""

    def setUp(self):
        self.gen = RFMFeatureGenerator()

    def test_basic_rfm(self):
        transactions = [
            {"date": "2026-06-15", "amount": 1000},
            {"date": "2026-06-20", "amount": 2000},
            {"date": "2026-06-28", "amount": 500},
        ]
        features = self.gen.compute_rfm(transactions, reference_date="2026-07-01")
        self.assertIn("recency_days", features)
        self.assertIn("frequency", features)
        self.assertIn("monetary_total", features)
        self.assertEqual(features["frequency"], 3)
        self.assertAlmostEqual(features["monetary_total"], 3500.0)

    def test_empty_transactions(self):
        features = self.gen.compute_rfm([], reference_date="2026-07-01")
        self.assertEqual(features["recency_days"], -1)
        self.assertEqual(features["frequency"], 0)

    def test_rfm_segments(self):
        transactions = [
            {"date": "2026-06-30", "amount": 50000},
            {"date": "2026-06-29", "amount": 30000},
            {"date": "2026-06-28", "amount": 20000},
        ]
        features = self.gen.compute_rfm(transactions, reference_date="2026-07-01")
        self.assertIn("frequency", features)
        self.assertEqual(features["frequency"], 3)


class TestTextFeatureExtractor(unittest.TestCase):
    """文本特征提取器测试"""

    def setUp(self):
        self.ext = TextFeatureExtractor()

    def test_basic_extraction(self):
        text = "该公司因经营异常被列入失信名单，涉及诉讼金额500万元。"
        features = self.ext.extract_features(text)
        self.assertIn("text_char_count", features)
        self.assertIn("text_keyword_count", features)
        self.assertGreater(features["text_keyword_count"], 0)

    def test_empty_text(self):
        features = self.ext.extract_features("")
        self.assertEqual(features["text_char_count"], 0)

    def test_financial_keywords(self):
        text = "企业完成A轮融资，获得亿元级投资"
        features = self.ext.extract_features(text)
        self.assertIn("label_融资", features)
        self.assertEqual(features["label_融资"], 1)


class TestFeatureSelector(unittest.TestCase):
    """三阶段特征选择器测试"""

    def test_iv_filter(self):
        selector = FeatureSelector(min_iv=0.02)
        iv_dict = {"feat_a": 0.5, "feat_b": 0.01, "feat_c": 0.1}
        result = selector.filter_by_iv(iv_dict)
        self.assertIn("feat_a", result)
        self.assertNotIn("feat_b", result)
        self.assertIn("feat_c", result)

    def test_correlation_removal(self):
        selector = FeatureSelector(max_corr=0.7)
        features = ["feat_a", "feat_b", "feat_c"]
        corr = {"feat_a": {"feat_b": 0.9, "feat_c": 0.3},
                "feat_b": {"feat_a": 0.9, "feat_c": 0.2}}
        iv = {"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.1}
        result = selector.remove_correlated(features, corr, iv)
        # feat_a和feat_b高度相关，保留IV更高的feat_a
        self.assertIn("feat_a", result)
        self.assertNotIn("feat_b", result)

    def test_psi_filter(self):
        selector = FeatureSelector(max_psi=0.25)
        psi = {"feat_a": 0.05, "feat_b": 0.3, "feat_c": 0.15}
        result = selector.filter_by_psi(psi)
        self.assertIn("feat_a", result)
        self.assertNotIn("feat_b", result)
        self.assertIn("feat_c", result)

    def test_full_pipeline(self):
        selector = FeatureSelector()
        iv = {"a": 0.5, "b": 0.01, "c": 0.3, "d": 0.2}
        corr = {"a": {"c": 0.85}, "c": {"a": 0.85}}
        psi = {"a": 0.05, "c": 0.3, "d": 0.1}
        result = selector.select(iv, corr, psi)
        # b: IV<0.02 → out; c: 与a高相关且IV低 → out; c: PSI=0.3 → out anyway
        self.assertIn("a", result)
        self.assertIn("d", result)

    def test_psi_stability_level(self):
        self.assertEqual(FeatureSelector.psi_stability_level(0.05), "稳定")
        self.assertEqual(FeatureSelector.psi_stability_level(0.15), "轻微不稳定")
        self.assertEqual(FeatureSelector.psi_stability_level(0.5), "不稳定")


class TestComputePSI(unittest.TestCase):
    """PSI计算函数测试"""

    def test_identical_distributions(self):
        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        actual = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        psi = compute_psi(expected, actual)
        self.assertLess(psi, 0.1)  # 相同分布PSI应接近0

    def test_shifted_distributions(self):
        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10
        actual = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15] * 10
        psi = compute_psi(expected, actual)
        self.assertGreater(psi, 0.1)  # 分布偏移PSI应较大


class TestFeatureReportGenerator(unittest.TestCase):
    """特征评估报告生成器测试"""

    def test_generate_report(self):
        gen = FeatureReportGenerator()
        iv = {"feat_a": 0.5, "feat_b": 0.01, "feat_c": 0.15}
        report = gen.generate(iv)
        self.assertIn("summary", report)
        self.assertIn("feature_details", report)
        self.assertEqual(report["summary"]["total_features"], 3)
        self.assertEqual(report["summary"]["strong_features"], 1)

    def test_to_text(self):
        gen = FeatureReportGenerator()
        iv = {"feat_a": 0.5}
        report = gen.generate(iv)
        text = gen.to_text(report)
        self.assertIn("特征评估报告", text)
        self.assertIn("feat_a", text)

    def test_to_json(self):
        gen = FeatureReportGenerator()
        iv = {"feat_a": 0.5}
        report = gen.generate(iv)
        json_str = gen.to_json(report)
        self.assertIn("feat_a", json_str)


if __name__ == "__main__":
    unittest.main()
