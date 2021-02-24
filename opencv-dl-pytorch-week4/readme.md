## <font style="color:blue">LR Scheduler</font>

We know that decreasing the learning rate with epochs or after a few epochs does help in convergence. Thankfully, this can be done in PyTorch using the `torch.optim.lr_scheduler` class which changes the learning rate for the optimizer depending upon the type of scheduler chosen. 

In this notebook, we will see how to use different learning rate schedulers. 

We will use the "Fashion MNIST" dataset, and SGD optimizer with momentum to illustrate the convergence of different learning rate schedulers.

We will use the same LeNet architecture.

- scheduler를 만들 때 optimzer를 넣어준다.
- lr는 lambda로 조정 가능

-------------------------------------------------  

## <font style="color:green">11.3. Exponential Decay (ExponentialLR)</font>

$$
\alpha = \alpha_0 * \gamma^n
$$

where, $\alpha_0 = \text{inital learning rate} $

$n = \text{epoch}$

$\gamma = \text{decay_rate}$

**ExponentialLR method:**

```
torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
```
- `optimizer` (Optimizer) – Wrapped optimizer.

- `gamma` (python:float) – Multiplicative factor of learning rate decay.

- `last_epoch` (python:int) – The index of last epoch. Default: -1.

Find details [here](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ExponentialLR).

## <font style="color:green">11.4. ReduceLROnPlateau</font>

This is the most interesting scheduler. In all the above LR-schedulers, reducing strategy is already defined. So it is a matter of experience and experiments to find the right hyperparameters of LR-scheduler. `ReduceLROnPlateeau` does solve this problem.

For the given number of epochs, if the model does not improve, it reduces the learning rate.

## **ReduceLROnPlateau method:**
(validation loss 기반으로 알아서 LR 조정.. 발전 없을 때 (진동) lr 감소시킴.)

```
torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
```
- `optimizer` (Optimizer) – Wrapped optimizer.

- `mode` (str) – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.

- `factor` (python:float) – Factor by which the learning rate will be reduced. `new_lr = lr * factor`. Default: `0.1`.

- `patience` (python:int) – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: `10`.

- `verbose` (bool) – If True, prints a message to stdout for each update. Default: False.

- `threshold` (python:float) – Threshold for measuring the new optimum, to only focus on significant changes. Default: `1e-4`.

- `threshold_mode` (str) – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.

- `cooldown` (python:int) – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: `0`.

- `min_lr` (python:float or list) – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: `0`.

- `eps` (python:float) – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.

Find details [here](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau)


--------------------------------------------------------

## Regularization 

### Batch Norm & DropOut

- 순서에 유념..?

```python
if batch_norm:
            self._body = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),

                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(dropout)
            )
```

### Batch Normalization

Training Deep Neural Networks is complicated by the fact that the distribution of each layer’s inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities.

Batch normalization, or batchnorm for short, is proposed as a technique to help coordinate the update of multiple layers in the model.

Batch normalization provides an elegant way of reparametrizing almost any deep network. The reparametrization significantly reduces the problem of coordinating updates across many layers.

- train & Inference 시 차이점.


### Dropout

With unlimited computation, the best way to “regularize” a fixed-sized model is to average the predictions of all possible settings of the parameters, weighting each setting by its posterior probability given the training data.

Weight들의 Co-Adaptation 효과를 최소화. Feature Visualization 시 더 선명한 feature를 확인할 수 있음. 즉, Hidden Cell의 Activation 증가.