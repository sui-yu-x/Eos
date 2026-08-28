# Palm detection model - NPU adapted version (NCHW)
# MONO grayscale single-channel version
# Changes: tf.pad -> Conv2D 1x1, removed FPN(Conv2DTranspose), single-scale output, data_format=channels_first
# All padding is explicit and symmetric to avoid NPU incompatibility
from tensorflow.keras.layers import Input, Conv2D, Add, ReLU, MaxPooling2D, DepthwiseConv2D, ZeroPadding2D, SpatialDropout2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.initializers import Constant

DF = 'channels_first'


def conv_blocks(x, num_filter, num_iterations=1):
    for _ in range(num_iterations):
        x = ReLU()(x)
        shortcut = x
        x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True, data_format=DF)(x)
        x = Conv2D(num_filter, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True, data_format=DF)(x)
        x = Add()([shortcut, x])
    return x


def conv_blocks_with_pooling(x, num_filter, channel_align=False):
    x = ReLU()(x)
    shortcut = x
    shortcut = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format=DF)(shortcut)

    if channel_align:
        shortcut = Conv2D(num_filter, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False, data_format=DF)(shortcut)

    # symmetric pad + valid instead of 'same' to avoid asymmetric padding with stride=2
    x = ZeroPadding2D(padding=(1, 1), data_format=DF)(x)
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='valid', use_bias=True, data_format=DF)(x)
    x = Conv2D(num_filter, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True, data_format=DF)(x)
    x = Add()([x, shortcut])
    return x


def detection_head(x, num_anchors=2, num_points=2, mid_channels=64):
    share = Conv2D(mid_channels, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True, data_format=DF)(x)
    cls = Conv2D(mid_channels, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True, data_format=DF)(share)
    # Prior bias (~1% positive) helps stabilize dense detection classification early in training.
    classificator = Conv2D(
        num_anchors,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        use_bias=True,
        bias_initializer=Constant(-4.6),
        data_format=DF,
    )(cls)
    classificator = Activation('sigmoid')(classificator)

    reg = Conv2D(mid_channels, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True, data_format=DF)(share)
    regressor = Conv2D(num_anchors * (4 + num_points * 2), kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True, data_format=DF)(reg)
    return classificator, regressor

# num_iterations可修改，默认为5。
# model = palm_detection_model()，默认每个 stage 都是 5
# model = palm_detection_model(num_iterations=7)，所有 stage 统一 7，之前就是7
# model = palm_detection_model(num_iterations=(5, 5, 4, 4, 3))，每个 stage 分别配置（共 5 段）

def _normalize_stage_iterations(num_iterations):
    """
    Normalize stage iterations to a 5-item tuple.

    Accepts:
    - int: use the same value for all 5 stages
    - list/tuple with 5 ints: per-stage values
    """
    if isinstance(num_iterations, int):
        if num_iterations < 1:
            raise ValueError('num_iterations must be >= 1')
# 括号带逗号 → 是单元素元组，乘以 5 会重复元素
        return (num_iterations,) * 5

    if isinstance(num_iterations, (list, tuple)):
        if len(num_iterations) != 5:
            raise ValueError('num_iterations list/tuple must have 5 elements')
        normalized = []
        for value in num_iterations:
            value = int(value)
            if value < 1:
                raise ValueError('Each stage num_iterations must be >= 1')
            normalized.append(value)
        return tuple(normalized)

    raise TypeError('num_iterations must be int or list/tuple of 5 ints')

# 后面 256 通道的几段最耗算力，优先砍后面几段，降计算最明显。
# 前两段保留 7，尽量稳住低层特征，关键点精度通常更稳。
def palm_detection_model(input_size=(1, 224, 384), num_iterations=(7, 7, 6, 5, 4)):
    it1, it2, it3, it4, it5 = _normalize_stage_iterations(num_iterations)

    inputs = Input(input_size)
    # symmetric pad + valid instead of 'same'
    x = ZeroPadding2D(padding=(1, 1), data_format=DF)(inputs)
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='valid', use_bias=True, data_format=DF)(x)

    # Spatial shapes preserve the 384:224 input aspect ratio. The two heads are
    # stride 16 (14x24) and stride 32 (7x12); the architecture and parameters
    # are otherwise identical to first_train.
    x = conv_blocks(x, 32, num_iterations=it1)
    x = conv_blocks_with_pooling(x, 64, channel_align=True)   # 32ch -> 64ch

    # block 7 ~ 12 (1, 64, 56, 56)
    x = conv_blocks(x, 64, num_iterations=it2)
    x = conv_blocks_with_pooling(x, 128, channel_align=True)  # 64ch -> 128ch

    # block 13 ~ 18 (1, 128, 28, 28)
    x = conv_blocks(x, 128, num_iterations=it3)
    x = SpatialDropout2D(0.05, data_format=DF)(x)
    x = conv_blocks_with_pooling(x, 256, channel_align=True)  # 128ch -> 256ch

    # block 19 ~ 24 (1, 256, 14, 14)
    x = conv_blocks(x, 256, num_iterations=it4)
    x = SpatialDropout2D(0.1, data_format=DF)(x)
    shortcut = x  # 14x14 feature for the second detection head
    x = conv_blocks_with_pooling(x, 256, channel_align=False)

    # block 25 ~ 30 (1, 256, 7, 7)
    x = conv_blocks(x, 256, num_iterations=it5)
    x = ReLU()(x)
    x = SpatialDropout2D(0.15, data_format=DF)(x)

    # Two detection heads: 14x14 and 7x7
    classificator_7, regressor_7 = detection_head(x, num_anchors=2, num_points=2)
    classificator_14, regressor_14 = detection_head(shortcut, num_anchors=2, num_points=2)
    model = Model(inputs, [regressor_14, classificator_14, regressor_7, classificator_7])
    return model
