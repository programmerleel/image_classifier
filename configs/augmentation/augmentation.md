# 模糊噪声
AdvancedBlur: 0.5
Blur: 0.5
# 下采样 上采样 降低图像质量
Downscale: 0.5
Defocus: 0.5
GlassBlur
GaussianBlur
GaussNoise
# 降低 jpeg webp格式图像质量
ImageCompression: 0
ISONoise
MultiplicativeNoise
MedianBlur
MotionBlur
ZoomBlur

# 几何变换
Affine
Flip
GridDistortion
HorizontalFlip
OpticalDistortion
Perspective
PiecewiseAffine
RandomRotate90
Rotate
SafeRotate
ShiftScaleRotate
Transpose
VerticalFlip

# 剪裁填充
BBoxSafeRandomCrop
Crop
CenterCrop
CropAndPad
CropNonEmptyMaskIfExists
PadIfNeeded
RandomCrop
RandomCropFromBorders
RandomCropNearBBox
RandomResizedCrop
RandomSizedBBoxSafeCrop
RandomSizedCrop

# 放缩
LongestMaxSize
RandomScale
Resize
SmallestMaxSize

# 图像直方图
CLAHE*
Equalize*
HistogramMatching*

# 图像基本属性
ColorJitter*
HueSaturationValue*
RandomToneCurve*
RandomBrightnessContrast*

# 数据类型变化吧
FromFloat*
ToGray*
ToFloat*
ToTensorV2
ToRGB*

# 特殊效果
ChannelDropout*
CoarseDropout
Cutout*
GridDropout
MaskDropout
PixelDropout

# 其余
RandomFog*
RandomRain*
RandomShadow*
RandomSnow*
RandomSunFlare*
Spatter*
RandomGravel*
ChannelShuffle*
Emboss*
ElasticTransform
FDA*
FancyPCA*
InvertImg*
Lambda
Normalize*
PixelDistributionAdaptation*
Posterize*
RandomGamma*
RandomGridShuffle
RingingOvershoot*
RGBShift*
Sharpen*
Solarize*
Superpixels*
ToSepia*
TemplateTransform*
UnsharpMask*

# 工具
Compose
OneOf
OneOrOther
ReplayCompose
Sequential
SomeOf