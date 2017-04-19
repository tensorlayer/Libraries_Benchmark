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
## Arcyfelix's test
GPU: GTX970

Driver Version: 375.39

TensorFlow: 1.0.1

Data: MNIST  train:50k  val:10k  test:10k

Num of epochs: 200

Batch size: 500

Keras: 171.689685 - 0.858448 s/epoch

TensorLayer: -

Tflearn: 342.936505s - 1.714684s s/epoch

## Speed of CNN


## Speed of LSTM
