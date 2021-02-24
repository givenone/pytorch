# TensorBoard 사용법.

What is TensorBoard?
TensorBoard is a visualization toolkit for machine learning experimentations. It has the following features:

- Tracking and visualizing metrics such as loss and accuracy

- Visualizing the model graph (ops and layers)

- Viewing histograms of weights, biases, or other tensors as they change over time

- Projecting embeddings to a lower-dimensional space - it gives a visual reprsentation of how the model classifies different instances of objects

- Displaying images, text, and audio data


# Transfer Learning

마지막 fc layer만 새로 학습 -> turn off autograd.

```python
def pretrained_resnet18(transfer_learning=True, num_class=3):
    resnet = models.resnet18(pretrained=True)
    
    if transfer_learning:
        for param in resnet.parameters():
            param.requires_grad = False
            
    last_layer_in = resnet.fc.in_features
    resnet.fc = nn.Linear(last_layer_in, num_class)
    
    return resnet
```