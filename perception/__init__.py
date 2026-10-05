"""Perception layer for beacon detection, neural discrimination, and gating."""
from .classical_detector import ClassicalBeaconDetector
from .discriminator import NeuralBeaconDiscriminator, BeaconDiscriminatorNet
from .hybrid_detector import HybridBeaconDetector
from .gating import MahalanobisGater
