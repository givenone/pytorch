## TorchVision

Function Syntax
torchvision.datasets.DATASET(root, train=True, transform=None, target_transform=None, download=False)
Where,

- DATASET is the name of the dataset, which can be MNIST, FashionMNIST, COCO etc. You can get the full list here
- root is the folder where the dataset is kept. Note that this will be used for downloading if you opt for that.
-train is a flag which specifies whether to use the train data or test data.
- download is a flag which is turned on when you want to download the data. Note that the data is not downloaded if it is already present in the root folder mentioned above.
- transform applies a series of image transforms on the input images. Some examples are cropping, resizing, etc.
- target_transform takes the target or labels and transforms it as required.

### Why is it useful?  

Suppose you are working on a problem and you achieve a decent accuracy. Now, you want to test your model on a different/harder data. So, you need to search for a dataset, go through how it is organized, download it on your system, prepare it so that it fits your training pipeline and then you are ready to use the new dataset.

However, if you are using torchvision datasets, you can skip all the above steps and treat the new dataset as a drop-in replacement for your old dataset. This is possible because almost all the datasets available on Torchvision have a similar API.

They have common arguments like transform and target_transform which apply some transform to the input as well as the labels/targets.

### Transforms

Transforms are image transforms that are applied while training a network. Simple operations like cropping, resizing, normalizing are all examples of a transform. We can apply multiple transforms to an image by chaining the transforms using the Compose Class.

Please have alook at the different transforms available in torchvision here

Some frequently used transforms are:

torchvision.transforms.ToTensor - It takes in a PIL image of dimension [H X W X C] in the range [0,255] and converts it to a float Tensor of dimension [C X H X W] in the range [0,1].
torchvision.transforms.Compose - It is used to chain many transforms together so that they can be applied in a single go


### Models

Just like torchvision.datasets consists of popular datasets which are useful for experimentation, torchvision.models consists of popular models for computer vision tasks such as:

Classification
Detection
Segmentation
Video Classification
The list keeps growing with time.

Function Syntax
model = torchvision.models.MODEL(pretrained=True)

### Ops

The ops module implements some functions used for specific computer vision tasks. Some of them are:

Non Maximum suppression - Used in Object detection pipelines
Region of Interest Pooling - Used in Fast RCNN paper
Region of Interest Alignment - Used in Mask RCNN paper
These are just mentioned for the sake if completeness and are rarely used.

