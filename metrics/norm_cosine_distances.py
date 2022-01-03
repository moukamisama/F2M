import torch.nn.functional as F
import torch.nn as nn

def norm_cosine_distances(output1, output2):
    cos = nn.CosineSimilarity(dim=1)
    logits = cos(output1, output2)

    return 1 - logits

def pair_norm_cosine_distances(output1, output2):
    """Estimate the eculidean distances between output1 and output2

    Args:
        output1 (a * m Tensor)
        output2 (b * m Tensor)
    Returns:
        pair eculidean distances (a * b Tensor)
    """

    a = output1.shape[0]
    b = output2.shape[0]

    output1 = output1.unsqueeze(1).expand(a, b, -1)
    output2 = output2.unsqueeze(0).expand(a, b, -1)

    cos = nn.CosineSimilarity(dim=2)
    logits = cos(output1, output2)
    return 1 - logits

def pair_norm_cosine_distances_dim3(output1, output2):
    """Estimate the eculidean distances between output1 and output2

    Args:
        output1 (batch * a * m Tensor)
        output2 (batch * b * m Tensor)
    Returns:
        pair eculidean distances (batch * a * b Tensor)
    """
    batch1, a, _ = output1.size()
    batch2, b, _ = output2.size()
    assert batch1 == batch2
    output1 = output1.unsqueeze(2).expand(batch1, a, b, -1)
    output2 = output2.unsqueeze(1).expand(batch2, a, b, -1)

    cos = nn.CosineSimilarity(dim=3)
    logits = cos(output1, output2)
    return 1 - logits