# Comparsion of TensorFlow Wrappers

Run Keras, TensorLayer and Tflearn with same model and data on a same GPU machine.

The parameter initialization may have slightly different, but would not effect the speed.

Feel free to PUSH !

## Speed of MLP

GPU: GTX980

TensorFlow: r0.10

Data: MNIST  train:50k  val:10k  test:10k

Model: 784-800-800-10

Num of epochs: 200

Batch size: 500

Keras: 282.475250s  = 1.41 s/epoch

TensorLayer: 116.670947s = 0.58 s/epoch

Tflearn:
# Arcyfelix's test
### Setup
GPU: GTX970

Driver Version: 375.39

TensorFlow: 1.0.1

Data: MNIST  train: 50k  val: 10k  test: 10k
### Speed of MLP
Num of epochs: 200
Batch size: 500
FC-x = Fully Connected / Dense with Relu activation with x number of neurons
DP = Dropout
| Architecture  / Library | Keras | TFLearn | TensorLayer |
| --- | --- | --- | --- |
INPUT + **FC-800** + DP + **FC-800** + DP + OUTPUT| **173.825s** | 337.312s| To be tested |
INPUT + **FC-2000** + DP + **FC-2000** + DP + OUTPUT | **377.443s** | 477.034s | To be tested |
INPUT + **FC-4000** + DP + **FC-4000** + DP + OUTPUT | 1007.613s| **872.662s** | To be tested |
INPUT + **FC-4000** + DP + **FC-4000** + **FC-4000** + DP + OUTPUT|1715.068s | **1313.363s** | To be tested |

### Speed of CNN
Num of epochs: 20
Batch size: 100
Conv2d[kernel-x, kernel-y]-filters = Convolutional layer with padding = 'same'
Architecture  / Library                                                            |  Keras   | TFLearn  | TensorLayer  |
-----------------------------------------------------------------------------------|----------|-----|------------- |
INPUT + **Conv2d[3,3]-8** + **Conv2d[3,3]-8** + FC-100 + DP + FC-100 + DP + OUTPUT | **79.999s** | 84.487s | To be tested |
INPUT + **Conv2d[3,3]-32** + **Conv2d[3,3]-32** + FC-100 + DP + FC-100 + DP + OUTPUT | 132.741s | **125.306s** | To be tested |
INPUT + **Conv2d[3,3]-64** + **Conv2d[3,3]-64** + FC-100 + DP + FC-100 + DP + OUTPUT | 230.574s | **204.685s** | To be tested |
INPUT + **Conv2d[3,3]-128** + **Conv2d[3,3]-128** + FC-100 + DP + FC-100 + DP + OUTPUT | 477.009s | **407.489s** | To be tested |
INPUT + **Conv2d[3,3]-256** + **Conv2d[3,3]-256** + FC-100 + DP + FC-100 + DP + OUTPUT | 1186.775s | **1037.954s** | To be tested |

## Speed of LSTM
