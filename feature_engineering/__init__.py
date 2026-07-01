"""
Financial Feature Engineering Toolkit
金融风控特征工程工具包

将原始金融数据转化为高区分度风控特征，
覆盖时序、交叉、行为、文本四大特征族。

Author: Ma Zhibin (马志斌)
Background: Applied Statistics MS, Financial Risk Control Data PM
"""

from .temporal import TemporalFeatureGenerator
from .cross import CrossFeatureEncoder, calculate_woe_iv
from .behavioral import RFMFeatureGenerator
from .text_features import TextFeatureExtractor
from .selector import FeatureSelector
from .report import FeatureReportGenerator

__all__ = [
    "TemporalFeatureGenerator",
    "CrossFeatureEncoder",
    "calculate_woe_iv",
    "RFMFeatureGenerator",
    "TextFeatureExtractor",
    "FeatureSelector",
    "FeatureReportGenerator",
]
