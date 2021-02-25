## Precision

* **True Positives (TP)** - correctly detected objects.
* **False Positives (FP)** - incorrectly detected objects.
* **False Negative (FN)** - GT boxes that do not have a
  corresponding detection with high enough IoU.
* **Precision** is the ratio of true positives in the obtained results. In
  other words, it is the percentage of correct predictions among all
  predictions:
  $ \text{precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} =
  \frac{\text{TP}}{D},$ where $D$ is a total number of detections.
* **Recall** - the amount of true positives we found among all the GT boxes in the data.
  $\text{recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} =
  \frac{\text{TP}}{G},$ where $G$ is a total number of ground truth objects.
* **Intersection over Union** is a way to measure the level of overlap
  between bounding boxes, for example, between ground truth and predicted
  detection boxes: $\text{IoU} = \frac{\text{intersection_area}}{\text{union_area}}$


# Object Detection

1. Sliding Window
2. R CNN (External Region Proposal)
3. FAST RCNN (ROI Procjection & Pooling -> feature map을 크기에 맞게 조정 -> FC Layer)
4. Faster RCNN (fully Differentiable)

## Faster RCNN

- base CNN을 사용 -> RPN (Region Proposal Network)  
- k anchor -> classification (binary) & Regression layer  
- RPN은 [k, n(24,,), n] / [4 * k, n, n]의 output 생성  
- RPN 의 bbox 기반으로 ROI Pooling으로 FC Layer Input 사이즈 통일.  
- `torch.nn.AdaptiveMaxPool2d()` 사용 가능 

## <font style="color:blue">Object Detection with Faster-RCNN</font>
The model comes built-in with the Torchvision library. We can simply load the model using `torchvision.models`. We also want to use the pre-trained weights so that we can check the detection with our own images. (We will see how to train a custom Object Detector using your own data in the next section.)

### <font style="color:green">Input </font>
The pretrained Faster-RCNN ResNet-50 model we are going to use expects the input image tensor to be in the form ```[n, c, h, w]``` 
where 
- n is the number of images
- c is the number of channels , for RGB images it is 3
- h is the height of the image
- w is the width of the image

### <font style="color:green">Output </font>
The model will return
- Bounding boxes [x0, y0, x1, y1]  are all predicted classes of shape (N,4) where N is the number of classes predicted by the model to be present in the image.
- Labels of all predicted classes.
- Scores of each predicted label.

```python
# get the pretrained model from torchvision.models
# Note: pretrained=True will get the pretrained weights for the model.
# model.eval() to use the model for inference
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
```

~~torchvision 최고~~