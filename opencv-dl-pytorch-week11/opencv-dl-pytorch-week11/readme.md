# Semantic Segmentation

## FCN (Fully Convolutio Network)

[credit](https://medium.com/@msmapark2/fcn-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-fully-convolutional-networks-for-semantic-segmentation-81f016d76204#:~:text=%EC%9E%85%EB%A0%A5%20%EC%9D%B4%EB%AF%B8%EC%A7%80%EC%97%90%20%EB%8C%80%ED%95%B4%20%ED%94%BD%EC%85%80,%EC%97%90%20%EB%A7%9E%EC%B6%B0%20%EB%B3%80%ED%98%95%EC%8B%9C%ED%82%A8%20%EA%B2%83%EC%9D%B4%EB%8B%A4.)

1 x 1 convolution으로 Fully Connected Network 대체. Pixel-wise Prediction 가능.

Coarse map에서 Dense map을 얻는 몇 가지 방법이 있다:

- Interpolation
- `Deconvolution`
- Unpooling
- Shift and stitch

물론 Pooling을 사용하지 않거나, Pooling의 stride를 줄임으로써 Feature map의 크기가 작아지는 것을 처음부터 피할 수도 있다.
그러나, 이 경우 필터가 더 세밀한 부분을 볼 수는 있지만 Receptive Field가 줄어들어 이미지의 컨텍스트를 놓치게 된다.
또한, Pooling의 중요한 역할 중 하나는 특징맵의 크기를 줄임으로써 학습 파라미터의 수를 감소시키는 것인데, 이러한 과정이 사라지면 파라미터의 수가 급격히 증가하고 이로인해 더 많은 학습시간을 요구하게 된다.

### Skip Architecture

Deep & Coarse(추상적인) 레이어의 의미적(Semantic) 정보와 Shallow & fine 층의 외관적(appearance) 정보를 결합한 Skip architecture를 정의한다. 시각화 모델을 통해 입력 이미지에 대해 얕은 층에서는 주로 직선 및 곡선, 색상 등의 낮은 수준의 특징에 활성화되고, 깊은 층에서는 보다 복잡하고 포괄적인 개체 정보에 활성화된다는 것을 확인할 수 있다. 또한 얕은 층에선 local feature를 깊은 층에선 global feature를 감지한다고 볼 수 있다. FCNs 연구팀은 이러한 직관을 기반으로 앞에서 구한 Dense map에 얕은 층의 정보를 결합하는 방식으로 Segmentation의 품질을 개선하였다.


## U Net

FCN 기반의 모델

[좋은 설명](https://medium.com/@msmapark2/u-net-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-u-net-convolutional-networks-for-biomedical-image-segmentation-456d6901b28a)

## DeepLab

[Credit](https://blog.lunit.io/2018/07/02/deeplab-v3-encoder-decoder-with-atrous-separable-convolution-for-semantic-image-segmentation/)

### Dilated Convolution

Atrous convolution을 활용함으로써 얻을 수 있는 이점은, 기존 convolution과 동일한 양의 파라미터와 계산량을 유지하면서도, field of view (한 픽셀이 볼 수 있는 영역) 를 크게 가져갈 수 있게 됩니다. 보통 semantic segmentation에서 높은 성능을 내기 위해서는 convolutional neural network의 마지막에 존재하는 한 픽셀이 입력값에서 어느 크기의 영역을 커버할 수 있는지를 결정하는 receptive field 크기가 중요하게 작용합니다. Atrous convolution을 활용하면 파라미터 수를 늘리지 않으면서도 receptive field를 크게 키울 수 있기 때문에 DeepLab series에서는 이를 적극적으로 활용하려 노력합니다.

### Spatial Pyramid Pooling


Spatial Pyramid Pooling (SPP) is a pooling layer that removes the fixed-size constraint of the network, i.e. a CNN does not require a fixed-size input image. Specifically, we add an SPP layer on top of the last convolutional layer. The SPP layer pools the features and generates fixed-length outputs, which are then fed into the fully-connected layers (or other classifiers).
![image](https://paperswithcode.com/media/methods/Screen_Shot_2020-06-21_at_3.05.44_PM.png)

Semantic segmentaion의 성능을 높이기 위한 방법 중 하나로, spatial pyramid pooling 기법이 자주 활용되고 있는 추세입니다. DeepLab V2에서는 feature map으로부터 여러 개의 rate가 다른 atrous convolution을 병렬로 적용한 뒤, 이를 다시 합쳐주는 atrous spatial pyramid pooling (ASPP) 기법을 활용할 것을 제안하고 있습니다. 최근 발표된 PSPNet에서도 atrous convolution을 활용하진 않지만 이와 유사한 pyramid pooling 기법을 적극 활용하고 있습니다. 이러한 방법들은 multi-scale context를 모델 구조로 구현하여 보다 정확한 semantic segmentation을 수행할 수 있도록 돕게 됩니다. DeepLab 에서는 ASPP를 기본 모듈로 사용하고 있습니다.

### Depthwise Convolution

depthwise convolution으로 나온 결과에 대해, 1×1×C 크기의 convolution filter를 적용한 것을 depthwise separable convolution 이라 합니다. 이처럼 복잡한 연산을 수행하는 이유는 기존 convolution과 유사한 성능을 보이면서도 사용되는 파라미터 수와 연산량을 획기적으로 줄일 수 있기 때문입니다.

Depthwise separable convolution은 기존 convolution filter가 spatial dimension과 channel dimension을 동시에 처리하던 것을 따로 분리시켜 각각 처리한다고 볼 수 있습니다. 이 과정에서, 여러 개의 필터가 spatial dimension 처리에 필요한 파라미터를 하나로 공유함으로써 파라미터의 수를 더 줄일 수 있게 됩니다. 두 축을 분리시켜 연산을 수행하더라도 최종 결과값은 결국 두 가지 축 모두를 처리한 결과값을 얻을 수 있으므로, 기존 convolution filter가 수행하던 역할을 충분히 대체할 수 있게 됩니다.

픽셀 각각에 대해서 label을 예측해야 하는 semantic segmentation은 난이도가 높은 편에 속하기 때문에 CNN 구조가 깊어지고 receptive field를 넓히기 위해 더 많은 파라미터들을 사용하게 되는 상황에서, separable convolution을 잘 활용할 경우 모델에 필요한 parameter 수를 대폭 줄일 수 있게 되므로 보다 깊은 구조로 확장하여 성능 향상을 꾀하거나, 기존 대비 메모리 사용량 감소와 속도 향상을 기대할 수 있습니다

-> MobileNet (경량화된 CNN) [link](https://arxiv.org/abs/1704.04861)


### Torchvision FCN

```python
fcn_resnet101 = models.segmentation.fcn_resnet101(pretrained=True).eval()

trf = T.Compose([T.Resize(640),
                 T.ToTensor(), 
                 T.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])

inp = trf(img).unsqueeze(0)

print('Image size after transformation: {}'.format(inp.size()))
# torch.Size([1, C, H, W])

out = fcn_resnet101(inp)['out']
print (out.shape)
om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy() # [H, W]
print (np.unique(om)) # [0 C-1]
```

### Torchvision DeeplabV3

21개 class로 pre-학습 (1 for bg)

```python
deeplabv3_resnet101 = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

```

-----------------------------------------------------------------

## ConvTranspose2D

[link](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)

Deconvolution Layer. This module can be seen as the gradient of Conv2d with respect to its input. It is also known as a fractionally-strided convolution or a deconvolution (although it is not an actual deconvolution operation).

`Decoder 예시`

```python
class DecoderBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()

        # 1x1 projection module to reduce channels
        self.proj = nn.Sequential(
            # convolution
            nn.Conv2d(channels_in, channels_in // 4, kernel_size=1, bias=False),
            # batch normalization
            nn.BatchNorm2d(channels_in // 4),
            # relu activation
            nn.ReLU()
        )

        # fully convolutional module
        self.deconv = nn.Sequential(
            # deconvolution
            nn.ConvTranspose2d(
                channels_in // 4,
                channels_in // 4,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                groups=channels_in // 4,
                bias=False
            ),
            # batch normalization
            nn.BatchNorm2d(channels_in // 4),
            # relu activation
            nn.ReLU()
        )

        # 1x1 unprojection module to increase channels
        self.unproj = nn.Sequential(
            # convolution
            nn.Conv2d(channels_in // 4, channels_out, kernel_size=1, bias=False),
            # batch normalization
            nn.BatchNorm2d(channels_out),
            # relu activation
            nn.ReLU()
        )

    # stack layers and perform a forward pass
    def forward(self, x):

        proj = self.proj(x)
        deconv = self.deconv(proj)
        unproj = self.unproj(deconv)

        return unproj
```
--------------------------------------------

### <font style="color:green">Dice Coefficient</font>

The **`Sørensen-Dice coefficient`** is applied for the pixel-wise comparison between a model prediction (segmented input) and the ground truth. There are several formulas for coefficient computation: the original was based on the
cardinality of two sets, for example, $A$ and $B$: $$\frac{2|A\cap B|}{|A|+|B|},$$ where $|A|$, $|B|$ are the cardinalities of the $A$, $B$ sets accordingly.

<img src='https://www.learnopencv.com/wp-content/uploads/2020/04/c3-w11-dice-coeff.png'>

--------------------------------

## Instance Segmentation using Mask RCNN

**Instant Segmentation -> Object Detection + Semantic Segmentation**

Instance Segmentation is challenging because it requires the correct detection of all objects in an image while also precisely segmenting each instance. It therefore combines elements from the classical computer vision tasks of object detection, where the goal is to classify individual objects and localize each using a bounding box, and semantic segmentation, where the goal is to classify each pixel into a fixed set of categories without differentiating object instances.

<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/c3-w11-instance-egmentation.png" width=1000>

**Faster RCNN**  consists of two stages. The first stage, called a Region Proposal Network (RPN),
proposes candidate object bounding boxes. The second stage, which is in essence Fast R-CNN, extracts features using RoIPool from each candidate box and performs classification and bounding-box regression. Mask R-CNN adopts the same two-stage procedure, with an identical first stage (which is RPN). In the second stage, in parallel to predicting the class and box offset, Mask R-CNN also outputs a binary mask for each RoI.

### Torchvision

[implementation](https://pytorch.org/vision/stable/_modules/torchvision/models/detection/mask_rcnn.html)

```python
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
```

num_classes=91

[Documentation](https://pytorch.org/vision/0.8/models.html#mask-r-cnn)
