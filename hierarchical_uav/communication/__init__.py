"""
Communication Module for Hierarchical UAV System

Implements:
1. Self-matching gating mechanism
2. Query/Key generation networks
3. gRPC-based communication between L-UAV and H-UAV
"""

from .self_matching import SelfMatchingGate, QueryKeyGenerator
from .grpc_server import HUAVServer
from .grpc_client import LUAVClient

__all__ = [
    'SelfMatchingGate',
    'QueryKeyGenerator',
    'HUAVServer',
    'LUAVClient'
]
