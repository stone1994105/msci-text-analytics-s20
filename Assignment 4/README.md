# MSCI-641 Assignment 4 Report
Name: Tianyu Shi<br />
Student ID: 20570373<br />

|                                           | Sigmoid |   ReLU  |   Tanh  |
| ----------------------------------------- | ------- | ------- | ------- |
| No regularization, No Dropout             |  0.804  |  0.807  |  0.807  |
| No regularization, Dropout = 0.3          |  0.814  |  0.812  |  0.813  |
| No regularization, Dropout = 0.4          |  0.813  |  0.811  |  0.813  |
| No regularization, Dropout = 0.5          |  0.813  |  0.812  |  0.812  |
| No regularization, Dropout = 0.6          |  0.813  |  0.811  |  0.813  |
| L2-regularization = 0.01, No Dropout      |  0.811  |  0.815  |  0.810  |
| L2-regularization = 0.01, Dropout = 0.3   |  0.806  |  0.813  |  0.809  |
| L2-regularization = 0.01, Dropout = 0.4   |  0.809  |  0.813  |  0.806  |
| L2-regularization = 0.01, Dropout = 0.5   |  0.810  |  0.812  |  0.815  |
| L2-regularization = 0.01, Dropout = 0.6   |  0.809  |  0.814  |  0.813  |
| L2-regularization = 0.001, No Dropout     |  0.813  |  0.813  |  0.810  |
| L2-regularization = 0.001, Dropout = 0.3  |  0.814  |  0.812  |  0.810  |
| L2-regularization = 0.001, Dropout = 0.4  |  0.815  |  0.814  |  0.814  |
| L2-regularization = 0.001, Dropout = 0.5  |  0.814  |  0.813  |  0.814  |
| L2-regularization = 0.001, Dropout = 0.6  |  0.814  |  0.813  |  0.815  |

## Effect of activation function:
Let's compare the three cases with no dropout, and the ReLU activation function has higher accuracy then the other two activation functions. One benefit of ReLu is that it converges faster than sigmoid and tanh. Another good point of using ReLU is that it has no vanishing gradient problems.

For more details, here are some links that compares different activation functions<br />
https://towardsdatascience.com/comparison-of-activation-functions-for-deep-neural-networks-706ac4284c8a<br />
https://towardsdatascience.com/complete-guide-of-activation-functions-34076e95d044

## Effect of L2-norm regularization:
The result in the table shows that regularization could increase the accuracy. The L2-norm regularization term penalizes large weight values, and prevents the neural network from overfitting.

## Effect of dropout:

Dropout is also a regularization method that randomly ignores some number of layer outputs, and it can also improve accuracy.

For furthur study, the optimal combination of L2-norm regularization and dropout rate should be investigated.
