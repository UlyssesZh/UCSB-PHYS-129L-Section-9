# Week 9B

## Task 1A

See `graph_mlp.png`.

## Task 1B

See `mlp.py`.

## Task 2A

See `graph_cnn.png`.

## Task 2B

See `cnn.py`.

## Task 3

See `mnist.py` for NumPy implementation,
and see `mnist_torch.py` for PyTorch implementation.
The network structures and training datasets are the same for both implementations,
but the training process is different.

It will attempt to download MNIST dataset if it is not found in the current directory.

In `mnist.py`, it will attempt to train a CNN model and a MLP model
if the weights are not found in the current directory.
The training takes 10 to 20 minutes for each model.
You can download weights trained on my computer from the GitHub Release page.
The training in `mnist_torch.py` is faster, taking less than 5 seconds for each model (with CUDA).

It is recommended to install the package `tqdm` to see the progress bar.
