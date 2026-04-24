
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
        conv_layers = [torch.nn.Conv2d(in_channels=ic, out_channels=oc,
            kernel_size=k, stride=s, padding=p) for (ic, oc, k, s, p) in
            zip(in_channels, channels, kernel_sizes, strides, paddings)]
        sequence = list()
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
                sequence.append(torch.nn.MaxPool2d(maxp_stride))  # No padding.
        self.conv = torch.nn.Sequential(*sequence)

    def forward(self, input):
        """Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W]."""
        return self.conv(input)

    def conv_out_size(self, h, w, c=None):
        """Helper function ot return the output size for a given input shape,
        without actually performing a forward pass through the model."""
        for child in self.conv.children():
            try:
                h, w = conv2d_output_shape(h, w, child.kernel_size,
                    child.stride, child.padding)
            except AttributeError:
                pass  # Not a conv or maxpool layer.
            try:
                c = child.out_channels
            except AttributeError:
                pass  # Not a conv layer.
        return h * w * c

    def conv_out_resolution(self, h, w):
        """Helper function that return the resolution (H, W) for a giben input resolution"""
        for child in self.conv.children():
            try:
                h, w = conv2d_output_shape(h, w, child.kernel_size,
                    child.stride, child.padding)
            except AttributeError:
                pass  # Not a conv or maxpool layer.
            try:
                c = child.out_channels
            except AttributeError:
                pass  # Not a conv layer.
        return h, w

class Conv2dHeadModel(torch.nn.Module):
    """Model component composed of a ``Conv2dModel`` component followed by 
    a fully-connected ``MlpModel`` head.  Requires full input image shape to
    instantiate the MLP head.
    """

    def __init__(
            self,
            image_shape,
            channels,
            kernel_sizes,
            strides,
            hidden_sizes,
            output_size=None,  # if None: nonlinearity applied to output.
            paddings=None,
            nonlinearity=torch.nn.LeakyReLU,
            use_maxpool=False,
            normlayer= None, # if None, will not be used
            ):
        super().__init__()
        if isinstance(nonlinearity, str): nonlinearity = getattr(torch.nn, nonlinearity)
        c, h, w = image_shape
        self.conv = Conv2dModel(
            in_channels=c,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            nonlinearity=nonlinearity,
            use_maxpool=use_maxpool,
            normlayer=normlayer,
        )
        conv_out_size = self.conv.conv_out_size(h, w)
        if hidden_sizes or output_size:
            self.head = MlpModel(conv_out_size, hidden_sizes,
                output_size=output_size, nonlinearity=nonlinearity)
            if output_size is not None:
                self._output_size = output_size
            else:
                self._output_size = (hidden_sizes if
                    isinstance(hidden_sizes, int) else hidden_sizes[-1])
        else:
            self.head = lambda x: x
            self._output_size = conv_out_size

    def forward(self, input):
        """Compute the convolution and fully connected head on the input;
        assumes correct input shape: [B,C,H,W]."""
        return self.head(self.conv(input).view(input.shape[0], -1))

    @property
    def output_size(self):
        """Returns the final output size after MLP head."""
        return self._output_size


class VisualMultiHeadEncoder(torch.nn.Module):
    def __init__(
        self,
        image_shape,
        channels,
        kernel_sizes,
        strides,
        shared_hidden_sizes,
        num_terrain_classes,
        geom_output_size,
        paddings=None,
        nonlinearity=torch.nn.LeakyReLU,
        use_maxpool=False,
        normlayer=None,
    ):
        super().__init__()

        c, h, w = image_shape

        self.backbone = Conv2dModel(
            in_channels=c,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            nonlinearity=nonlinearity,
            use_maxpool=use_maxpool,
            normlayer=normlayer,
        )

        conv_out_size = self.backbone.conv_out_size(h, w)

        self.shared_mlp = MlpModel(
            input_size=conv_out_size,
            hidden_sizes=shared_hidden_sizes,
            output_size=None,   # 最後一層也保留 activation
            nonlinearity=nonlinearity,
        )

        shared_out_size = self.shared_mlp.output_size

        self.terrain_head = torch.nn.Linear(shared_out_size, num_terrain_classes)
        self.geom_head = torch.nn.Linear(shared_out_size, geom_output_size)

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.view(x.shape[0], -1)
        feat = self.shared_mlp(feat)

        terrain_logits = self.terrain_head(feat)
        geom = self.geom_head(feat)

        return {
            "terrain_logits": terrain_logits,
            "geom": geom,
        }
    def pack_encoder_output(self, out_dict):
        terrain_logits = out_dict["terrain_logits"]   # (B, 7)
        geom = out_dict["geom"]                       # (B, 5) or (B, 8)

        terrain_prob = torch.softmax(terrain_logits, dim=-1)

        return torch.cat([terrain_prob, geom], dim=-1)
"""how to use 
self.visual_encoder = VisualMultiHeadEncoder(
    image_shape=(visual_channels, height, width),
    channels=vk["channels"],
    kernel_sizes=vk["kernel_sizes"],
    strides=vk["strides"],
    shared_hidden_sizes=[128, 64],
    num_terrain_classes=6,
    geom_output_size=5,   # 例如 [sin_yaw, cos_yaw, dist, width, height]
    paddings=vk.get("paddings", [2, 1, 1]),
    nonlinearity=torch.nn.LeakyReLU,
    use_maxpool=False,
    normlayer=lambda oc: torch.nn.GroupNorm(4, oc),
)
"""

class Conv2dBackboneModel(torch.nn.Module):
    """
    ConvNet with fixed output size = 64
    """

    def __init__(
        self,
        image_shape,  # (C, H, W)
        channels=(32, 64, 64),
        kernel_sizes=(5, 3, 3),
        strides=(2, 2, 2),
        paddings=(2, 1, 1),
        nonlinearity=torch.nn.ELU,
        normlayer=None,
        use_maxpool=True,
    ):
        super().__init__()

        if isinstance(nonlinearity, str):
            nonlinearity = getattr(torch.nn, nonlinearity)
        if isinstance(normlayer, str):
            normlayer = getattr(torch.nn, normlayer)

        c, _, _ = image_shape
        in_channels = [c] + list(channels[:-1])

        layers = []
        for ic, oc, k, s, p in zip(in_channels, channels, kernel_sizes, strides, paddings):

            if use_maxpool:
                conv_stride = 1
                pool_stride = s
            else:
                conv_stride = s
                pool_stride = 1

            layers.append(torch.nn.Conv2d(ic, oc, kernel_size=k, stride=conv_stride, padding=p))

            if normlayer is not None:
                layers.append(normlayer(oc))

            layers.append(nonlinearity())

            if use_maxpool and pool_stride > 1:
                layers.append(torch.nn.MaxPool2d(pool_stride))

        self.conv = torch.nn.Sequential(*layers)

        # 關鍵：Global Average Pool
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self._output_size = channels[-1]  # = 64

    def forward(self, x):
        x = self.conv(x)          # [B, 64, H', W']
        x = self.pool(x)          # [B, 64, 1, 1]
        x = x.view(x.size(0), -1) # [B, 64]
        return x

    @property
    def output_size(self):
        return self._output_size