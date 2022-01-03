from .losses import (TripletLossNoHardMining, ProtoRegularization, ProtoCFLoss,
                     ProtoCFLoss2, LessForget, InterClassSeparation, TripletLoss,
                     PNLoss, CosineTripletLossNoHardMining, BufferCenterLoss, NovelPNLoss,
                     NormFixLoss, CosineBufferCenterLoss, CosineNovelPNLoss, PrototypesCenterLoss,
                     PTFixPNLoss, IncrementalPTFixPNLoss, SimilarBufferCenterLoss, CosineSimilarBufferCenterLoss,
                     IncrementalPNormPTLoss, IncrementalNetworkFixLoss, CenterLoss, ProtoFixLoss, TripletLoss_pytorch)

from .noise_losses import (CosineTripletNoiseLoss, CosineFixNoiseLoss, CosineIncrementalMetricCFLoss,
                           NoiseProtoCFLoss, IncrementalMetricCFLoss, AllGreatNoiseLoss,
                           TripletNoiseLoss, CosineLessForget, CosineNoiseProtoCFLoss,
                           NoiseBufferCenterLoss, NoiseCosBufferCenterLoss, NoisePTFixLoss, NoiseProtoFixLoss,
                           NoiseSimilarBufferCenterLoss, NoiseCosSimilarBufferCenterLoss)

__all__ = [
    'TripletLossNoHardMining',
    'ProtoRegularization',
    'ProtoCFLoss',
    'ProtoCFLoss2',
    'LessForget',
    'InterClassSeparation',
    'AllGreatNoiseLoss',
    'TripletNoiseLoss',
    'NoiseProtoCFLoss',
    'IncrementalMetricCFLoss',
    'TripletLoss',
    'PNLoss',
    'CosineTripletLossNoHardMining',
    'CosineTripletNoiseLoss',
    'CosineFixNoiseLoss',
    'CosineIncrementalMetricCFLoss',
    'CosineLessForget',
    'CosineNoiseProtoCFLoss',
    'BufferCenterLoss',
    'NoiseBufferCenterLoss',
    'NovelPNLoss',
    'NormFixLoss',
    'CosineBufferCenterLoss',
    'CosineNovelPNLoss',
    'NoiseCosBufferCenterLoss',
    'NoisePTFixLoss',
    'PrototypesCenterLoss',
    'PTFixPNLoss',
    'IncrementalPTFixPNLoss',
    'SimilarBufferCenterLoss',
    'NoiseSimilarBufferCenterLoss',
    'CosineSimilarBufferCenterLoss',
    'NoiseCosSimilarBufferCenterLoss',
    'IncrementalPNormPTLoss',
    'IncrementalNetworkFixLoss',
    'NoiseProtoFixLoss',
    'CenterLoss',
    'ProtoFixLoss',
    'TripletLoss_pytorch'
]