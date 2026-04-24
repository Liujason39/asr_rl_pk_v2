import torch
import torch.nn as nn


def conv2d_output_shape(h, w, kernel_size=1, stride=1, padding=0, dilation=1):
    """Returns output H, W after convolution/pooling on input H, W."""
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    sh, sw = stride if isinstance(stride, tuple) else (stride, stride)
    ph, pw = padding if isinstance(padding, tuple) else (padding, padding)
    d = dilation
    h = (h + (2 * ph) - (d * (kh - 1)) - 1) // sh + 1
    w = (w + (2 * pw) - (d * (kw - 1)) - 1) // sw + 1
    return h, w


class Conv2dModel(nn.Module):
    """Pure 2D convolution stack with optional max-pooling."""

    def __init__(
        self,
        in_channels,
        channels,
        kernel_sizes,
        strides,
        paddings=None,
        nonlinearity=nn.ReLU,
        use_maxpool=False,
        normlayer=None,
    ):
        super().__init__()

        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        if isinstance(nonlinearity, str):
            nonlinearity = getattr(nn, nonlinearity)
        if isinstance(normlayer, str):
            normlayer = getattr(nn, normlayer)

        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings), \
            "channels, kernel_sizes, strides, paddings 長度必須一致"

        in_channel_list = [in_channels] + list(channels[:-1])
        ones = [1 for _ in range(len(strides))]

        if use_maxpool:
            maxpool_strides = strides
            conv_strides = ones
        else:
            maxpool_strides = ones
            conv_strides = strides

        layers = []
        for ic, oc, k, s, p, mp_s in zip(
            in_channel_list, channels, kernel_sizes, conv_strides, paddings, maxpool_strides
        ):
            layers.append(nn.Conv2d(ic, oc, kernel_size=k, stride=s, padding=p))

            if normlayer is not None:
                layers.append(normlayer(oc))

            layers.append(nonlinearity())

            if mp_s > 1:
                layers.append(nn.MaxPool2d(kernel_size=mp_s, stride=mp_s))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

    def conv_out_size(self, h, w, c=None):
        """Return flattened output size."""
        for layer in self.conv.children():
            try:
                h, w = conv2d_output_shape(
                    h, w,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                )
            except AttributeError:
                pass
            try:
                c = layer.out_channels
            except AttributeError:
                pass

        if c is None:
            raise ValueError("無法推得 conv output channels，請檢查網路結構。")
        return h * w * c

    def conv_out_resolution(self, h, w):
        c = None
        for layer in self.conv.children():
            try:
                h, w = conv2d_output_shape(
                    h, w,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                )
            except AttributeError:
                pass
            try:
                c = layer.out_channels
            except AttributeError:
                pass
        return h, w, c


class Conv2dBackboneModel(nn.Module):
    """
    只保留 ConvNet backbone，不接 MLP head。
    輸出為 flatten 後的 feature，對應論文中的 d_tilde。
    """

    def __init__(
        self,
        image_shape,
        channels,
        kernel_sizes,
        strides,
        paddings=None,
        nonlinearity=nn.ReLU,
        use_maxpool=False,
        normlayer=None,
        flatten_output=True,
    ):
        super().__init__()

        if isinstance(nonlinearity, str):
            nonlinearity = getattr(nn, nonlinearity)

        c, h, w = image_shape
        self.flatten_output = flatten_output

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

        if flatten_output:
            self._output_size = self.conv.conv_out_size(h, w)
        else:
            out_h, out_w, out_c = self.conv.conv_out_resolution(h, w)
            self._output_size = (out_c, out_h, out_w)

    def forward(self, x):
        x = self.conv(x)
        if self.flatten_output:
            x = x.view(x.shape[0], -1)
        return x

    @property
    def output_size(self):
        return self._output_size