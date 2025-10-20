# test_backbones.py
# All comments are in English as requested.

import torch
from .BuildNet import buildnet, model_dict

def try_forward(name: str, in_channel: int, img_size: int = 32):
    """Try a forward pass with a random tensor to validate shape and runtime errors."""
    x = torch.randn(2, in_channel, img_size, img_size)  # small batch for sanity check

    # We disable classifier to focus on encoder+projection validity
    # If you want to verify classifier too, set include_classifier=True.
    net = buildnet(
        name=name,
        head='mlp',
        feat_dim=128,
        num_classes=10,
        softmax=False,
        include_classifier=False,
        in_channel=in_channel,
    )

    with torch.no_grad():
        pred, feat = net(x)

    # pred is None when include_classifier=False
    pred_shape = None if pred is None else tuple(pred.shape)
    feat_shape = tuple(feat.shape)

    print(f"[OK] model={name:>14s} | in_channel={in_channel} | "
          f"feat={feat_shape} | pred={pred_shape}")

def main():
    # Test each registered backbone name from model_dict
    names = list(model_dict.keys())

    print("=== Test with 3-channel input (e.g., CIFAR) ===")
    for n in names:
        try:
            try_forward(n, in_channel=3)
        except Exception as e:
            print(f"[ERR] model={n:>14s} | in_channel=3 | error: {repr(e)}")

    print("\n=== Test with 1-channel input (e.g., MNIST) ===")
    for n in names:
        try:
            try_forward(n, in_channel=1)
        except Exception as e:
            print(f"[ERR] model={n:>14s} | in_channel=1 | error: {repr(e)}")

if __name__ == "__main__":
    main()
