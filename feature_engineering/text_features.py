"""
文本特征提取器 — 关键词TF-IDF + 语义标签映射

零依赖实现，基于领域词典的文本特征提取，
支持金融风控场景的语义标签自动映射。
"""

import math
import re
from typing import Optional


# 领域关键词词典
DOMAIN_KEYWORDS = {
    "融资": ["融资", "Pre-A", "A轮", "B轮", "C轮", "天使轮", "战略投资", "领投", "投融资"],
    "风险": ["经营异常", "行政处罚", "失信", "诉讼", "被执行人", "风险", "异常名录", "违约"],
    "中标": ["中标", "入选", "公示", "认定", "补贴", "专项资金", "拟支持"],
    "负面": ["欺诈", "逃废债", "骗贷", "逾期", "坏账", "欠款", "跑路"],
    "正面": ["增长", "盈利", "上市", "合作", "签约", "获批", "认证"],
}

# 停用词
STOP_WORDS = set(
    "的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 "
    "自己 这 他 她 它 们 那 些 什么 怎么 又 被 从 而 对 与 为 以 但 如 把 其 此".split()
)


class TextFeatureExtractor:
    """文本特征提取器 — 领域词典 + TF-IDF + 语义标签"""

    def __init__(self, keywords: Optional[dict[str, list[str]]] = None):
        """
        Args:
            keywords: 自定义领域词典，默认使用内置金融词典
        """
        self.keywords = keywords or DOMAIN_KEYWORDS
        self.idf_cache: dict[str, float] = {}

    def extract_keywords(self, text: str) -> dict[str, int]:
        """
        提取文本中的关键词及其出现次数

        Args:
            text: 输入文本

        Returns:
            {关键词: 出现次数} 的字典
        """
        if not text:
            return {}

        keyword_counts = {}
        for category, words in self.keywords.items():
            for word in words:
                count = text.count(word)
                if count > 0:
                    keyword_counts[word] = count

        return keyword_counts

    def compute_tf_idf(self, text: str, idf_dict: Optional[dict[str, float]] = None) -> dict[str, float]:
        """
        计算文本的TF-IDF特征

        Args:
            text: 输入文本
            idf_dict: 预计算的IDF字典，如未提供则使用纯TF

        Returns:
            {关键词: TF-IDF值} 的字典
        """
        keyword_counts = self.extract_keywords(text)
        if not keyword_counts:
            return {}

        total_keywords = sum(keyword_counts.values())

        tfidf = {}
        for word, count in keyword_counts.items():
            tf = count / total_keywords
            idf = idf_dict.get(word, 1.0) if idf_dict else 1.0
            tfidf[word] = round(tf * idf, 6)

        return tfidf

    def map_semantic_labels(self, text: str) -> dict[str, bool]:
        """
        将文本映射到语义标签

        Args:
            text: 输入文本

        Returns:
            {标签名: 是否命中} 的字典
        """
        if not text:
            return {cat: False for cat in self.keywords}

        labels = {}
        for category, words in self.keywords.items():
            labels[category] = any(word in text for word in words)

        return labels

    def extract_features(self, text: str, idf_dict: Optional[dict[str, float]] = None) -> dict:
        """
        完整文本特征提取：关键词 + TF-IDF + 语义标签

        Args:
            text: 输入文本
            idf_dict: 预计算的IDF字典

        Returns:
            综合特征字典
        """
        keyword_counts = self.extract_keywords(text)
        tfidf = self.compute_tf_idf(text, idf_dict)
        labels = self.map_semantic_labels(text)

        # 文本基础统计
        char_count = len(text) if text else 0
        word_count = len(re.findall(r"[\u4e00-\u9fa5]+|[a-zA-Z]+|[0-9]+", text)) if text else 0

        features = {
            "text_char_count": char_count,
            "text_word_count": word_count,
            "text_keyword_count": sum(keyword_counts.values()),
            "text_unique_keywords": len(keyword_counts),
        }

        # 语义标签特征
        for label, hit in labels.items():
            features[f"label_{label}"] = int(hit)

        # TF-IDF top关键词
        sorted_tfidf = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (word, score) in enumerate(sorted_tfidf):
            features[f"tfidf_top{i+1}_score"] = score

        return features

    def fit_idf(self, corpus: list[str]) -> dict[str, float]:
        """
        从语料库计算IDF值

        Args:
            corpus: 文本列表

        Returns:
            IDF字典
        """
        n_docs = len(corpus)
        if n_docs == 0:
            return {}

        doc_freq = {}
        for text in corpus:
            seen_words = set()
            for category, words in self.keywords.items():
                for word in words:
                    if word in text and word not in seen_words:
                        doc_freq[word] = doc_freq.get(word, 0) + 1
                        seen_words.add(word)

        idf_dict = {}
        for word, df in doc_freq.items():
            idf_dict[word] = round(math.log((n_docs + 1) / (df + 1)) + 1, 6)

        self.idf_cache = idf_dict
        return idf_dict
