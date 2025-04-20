import torch
from src.models.dncnn import DnCNN

def test_output_shape():
    model = DnCNN()
    x = torch.randn(1, 3, 256, 256)
    assert model(x).shape == x.shape