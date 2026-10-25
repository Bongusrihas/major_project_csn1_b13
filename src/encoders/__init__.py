"""Graph Encoders: Original Paper R-GCN vs Proposed LECG R-GCN."""
from src.encoders.flat_encoder import FlatEncoder
from src.encoders.rgcn_lecg import StructuralGraphEncoderLECG
from src.encoders.rgcn_original import RelationalGraphConvOriginal, StructuralGraphEncoderOriginal

__all__ = [
    "StructuralGraphEncoderOriginal",
    "RelationalGraphConvOriginal",
    "StructuralGraphEncoderLECG",
    "FlatEncoder",
]
