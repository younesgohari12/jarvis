from __future__ import annotations
from jarvis.agent.semantic_router_v15 import SemanticIntentRouterV15, SemanticIntentPrediction

class SemanticIntentRouterV18(SemanticIntentRouterV15):
    """v18 retrained semantic router. Runtime guards remain compatible with v15,
    while learned weights are trained on dataset_v009 and carry v18 metadata."""
    FORMAT='jarvis-semantic-router-v18'

__all__=['SemanticIntentRouterV18','SemanticIntentPrediction']
