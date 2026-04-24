
import torch

from asr_rl_pk.modules.mlp import MlpModel

def conv2d_output_shape(h, w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Returns output H, W after convolution/pooling on input H, W.
    """
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 2
    sh, sw = stride if isinstance(stride, tuple) else (stride,) * 2
    ph, pw = padding if isinstance(padding, tuple) else (padding,) * 2
    d = dilation
    h = (h + (2 * ph) - (d * (kh - 1)) - 1) // sh + 1
    w = (w + (2 * pw) - (d * (kw - 1)) - 1) // sw + 1
    return h, w

class Conv2dModel(torch.nn.Module):
    """2-D Convolutional model component, with option for max-pooling vs
    downsampling for strides > 1.  Requires number of input channels, but
    not input shape.  Uses ``torch.nn.Conv2d``.
    """

    def __init__(
            self,
            in_channels,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            use_maxpool=False,  # if True: convs use stride 1, maxpool downsample.
            head_sizes=None,  # Put an MLP head on top.
            normlayer= None, # If None, will not be used
            ):
        super().__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]

        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [in_channels] + channels[:-1]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones

        conv_layers = [
            torch.nn.Conv2d(
                in_channels=ic,
                out_channels=oc,
                kernel_size=k,
                stride=s,
                padding=p
            )
            for (ic, oc, k, s, p) in zip(in_channels, channels, kernel_sizes, strides, paddings)
        ]

        sequence = []
        for conv_layer, oc, maxp_stride in zip(conv_layers, channels, maxp_strides):
            sequence.append(conv_layer)

            if normlayer is not None:
                if isinstance(normlayer, str):
                    if normlayer == "BatchNorm2d":
                        sequence.append(torch.nn.BatchNorm2d(oc))
                    else:
                        raise ValueError(f"Unsupported normlayer string: {normlayer}")
                else:
                    # normlayer 是 callable，例如 lambda oc: nn.GroupNorm(4, oc)
                    sequence.append(normlayer(oc))

            sequence.append(nonlinearity())

            if maxp_stride > 1:
                sequence.append(torch.nn.MaxPool2d(maxp_stride))

        self.conv = torch.nn.Sequential(*sequence)