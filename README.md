# All-In-One Adam Optimizer in PyTorch

All-In-One Adam Optimizer where several novelties are combined from following papers:

1) Decoupled Weight Decay Regularization for Adam
https://arxiv.org/abs/1711.05101

Authors shown that the real reason why Momentum optimizer is often outperforming Adam in generalization was due to the fact that Adam does not perform well under L2 regularization and developed decoupled weight decay as a solution.

2) Online Learning Rate Adaptation with Hypergradient Descent
https://arxiv.org/abs/1703.04782

This is enabled via "hypergrad" parameter by setting it to any value except zero. It enables the optimizer to update the learning-rate itself by the technique proposed in the paper, instead of giving an external schedule which would require lots of additional hyperparameters. It is especially useful when one doesn't have the chance to hypertune a schedule.

3) Closing the Generalization Gap of Adaptive Gradient Methods in Training Deep Neural Networks
https://arxiv.org/abs/1806.06763

This can be set by the "partial" parameter, which controls how likely the optimizer acts similar to Adam (1.0) and SGD (0.0), which is very useful if hypertuned. One can also update (decay) this parameter online to switch between Adam and SGD optimizers in an easy way, which has been recommended by previous research for a better generalization.

# AdaBound with Decoupled Weight Decay

Adaptive Gradient Methods with Dynamic Bound of Learning Rate
https://github.com/Luolc/AdaBound

Exploiting Uncertainty of Loss Landscape for Stochastic Optimization
https://github.com/bsvineethiitg/adams
