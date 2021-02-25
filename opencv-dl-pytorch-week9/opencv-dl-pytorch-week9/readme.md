## YOLO

- single CNN -> SxS Cell 로 해석 (e.g S=7)
- Output Feature : [S, S, 30] ([x,y,w,h,c] x b(=2, bbox 개수 per cell), 20 for class score)
- C (confidence) 기준 thr, Class마다 정렬
- NMS

- small object 여러개 있으면 감지 불가, Scale 변화 대응 못함.

## SSD
`pytorch Hub`

- SSD implementation of Nvidia is a variant of [the original paper](https://arxiv.org/pdf/1512.02325.pdf). 


## <font style="color:green">Model Description</font>

---

<img src=https://www.learnopencv.com/wp-content/uploads/2020/03/c3-w9-ssd.png width=1000>

<center>Image credits: Liu et al</center>

---


This SSD300 model is based on the
[SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) paper, which
describes SSD as "a method for detecting objects in images using a single deep neural network".
The input size is fixed to `300x300`.

The main difference between this model and the one described in the paper is in the backbone.
Specifically, the VGG model is obsolete and is replaced by the ResNet-50 model.

From the
[Speed/accuracy trade-offs for modern convolutional object detectors](https://arxiv.org/abs/1611.10012)
paper, the following enhancements were made to the backbone:
*   The `conv5_x`, `avgpool`, `fc` and softmax layers were removed from the original classification model.
*   All strides in `conv4_x` are set to `1x1`. 

The backbone is followed by 5 additional convolutional layers.
In addition to the convolutional layers, we attached 6 detection heads:
*   The first detection head is attached to the last `conv4_x` layer.
*   The other five detection heads are attached to the corresponding `5` additional layers.

Detector heads are similar to the ones referenced in the paper, however,
they are enhanced by additional BatchNorm layers after each convolution.

**If you are interested in original SSD paper implementation, you can find [here](https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py).**

## RetinaNet

### Feature Pyramid Networks (FPN)

[좋은 설명](https://eehoeskrap.tistory.com/300)
각 추출된 결과들인 low-resolution 및 high-resolution 들을 묶는 방식이다. 

각 레벨에서 독립적으로 특징을 추출하여 객체를 탐지하게 되는데 

상위 레벨의 이미 계산 된 특징을 재사용 하므로 멀티 스케일 특징들을 효율적으로 사용 할 수 있다. 
CNN 자체가 레이어를 거치면서 피라미드 구조를 만들고 forward 를 거치면서 더 많은 의미(Semantic)를 가지게 된다. 

출처: https://eehoeskrap.tistory.com/300 [Enough is not enough]

- 1x1 -> high semantic info
- big -> good localization info