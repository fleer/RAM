# A Keras implementation of the Recurrent Attention Model

The **Recurrent Attention Model**...
Instead processing the whole image, it uses *glimpses* at different locations.

The code is inspired by [2] & [3]

## Installation
Needed packages:
1. [numpy](http://www.numpy.org/)
2. [tensorflow](https://www.tensorflow.org/)
3. [Keras](https://keras.io/)
4. [opencv](https://opencv.org/)
5. [matplotlib](http://matplotlib.org/) for plotting
6. [h5py](http://www.h5py.org/)

Install the packages via `pip`.

```
pip install numpy tensorflow keras opencv-python matplotlib h5py
```

## Usage
The parameter for the training are all defined in the configuration file, 
such as `run_mnist.py`.
There all parameters of the network, such as number of glimpses, number of zooms, 
optimizer, learning rate, etc. can be specified.

In the `main()` function, the number of runs can be specified, together with the 
filename of the results `.json` file.

After the training for the specified number of epochs, the network-model is 
saved as a `.h5py` file, while the trained weights are saved as `.json` file.
They can be loaded for further training or evaluation.

## Classification of the standard MNIST Dataset
To Train the network for classifying the standard MNIST Dataset, 
start the code via the corresponding confiuration file:
```
python run_mnist.py
```
The chosen parameters are nearly the same as in [3].

### Results
The plot below shows the training accuracy for the first 400 epochs:
![Example](./MNIST_Results/MNIST_accuracy.png)

### Examples Images
To create images of the glimpses that the network uses after training, simply execute the evaluation script `evaluate.py run_mnist`.

|Originial Image | Glimpse 0| Glimpse 1| Glimpse 2 |Glimpse 3|
|:--:|:--:|:--:|:--:|:--:|
|<img src="./MNIST_Results/Images/symbol_0.png" alt="Symbol0" width="140">|<img src="./MNIST_Results/Images/symbol_0_glimpse_0_zoom_1.png" alt="Glimpse0" width="140">|<img src="./MNIST_Results/Images/symbol_0_glimpse_1_zoom_1.png" alt="Glimpse1" width="140">|<img src="./MNIST_Results/Images/symbol_0_glimpse_2_zoom_1.png" alt="Glimpse2" width="140">|<img src="./MNIST_Results/Images/symbol_0_glimpse_3_zoom_1.png" alt="Glimpse3" width="140">|
|<img src="./MNIST_Results/Images/symbol_1.png" alt="Symbol1" width="140">|<img src="./MNIST_Results/Images/symbol_1_glimpse_0_zoom_1.png" alt="Glimpse0" width="140">|<img src="./MNIST_Results/Images/symbol_1_glimpse_1_zoom_1.png" alt="Glimpse1" width="140">|<img src="./MNIST_Results/Images/symbol_1_glimpse_2_zoom_1.png" alt="Glimpse2" width="140">|<img src="./MNIST_Results/Images/symbol_1_glimpse_3_zoom_1.png" alt="Glimpse3" width="140">|

## Classification of the translated MNIST Dataset
To Train the network for classifying the translated MNIST Dataset, 
start the code via the corresponding confiuration file:
```
python run_translated_mnist.py
```
The chosen parameters are nearly the same as in [3].


--------
[1] Mnih, Volodymyr, Nicolas Heess, and Alex Graves. "Recurrent models of visual attention." Advances in neural information processing systems. 2014.

[2] https://github.com/jlindsey15/RAM

[3] http://torch.ch/blog/2015/09/21/rmva.html
