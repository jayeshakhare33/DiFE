"""
Feature Engineering Module
Extracts graph-based features (embeddings, degree, centrality, clustering, etc.)
"""

from .feature_extractor import FeatureExtractor, EdgeFeatureExtractor
from .graph_embeddings import GraphEmbeddings

__all__ = ['FeatureExtractor', 'EdgeFeatureExtractor', 'GraphEmbeddings']

