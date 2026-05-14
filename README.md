# Shortcut Learning on MNIST

Final project for CS 4210. This project looks at shortcut learning in machine
learning by training a small CNN on MNIST and then testing it on rotated and
noisy versions of the same digits to see how much the accuracy drops.

The idea is that even though the model gets really high accuracy on the normal
test set, it is not actually learning what we think it is. It is just picking
up on patterns that only exist in the training distribution (upright, centered,
clean digits), so as soon as those patterns change a little the accuracy falls
off.

## How to run

The whole project is one Jupyter Notebook: `shortcut-learning-mnist.ipynb`.

Easiest option is Kaggle:
1. Go to https://www.kaggle.com/code and click "New Notebook"
2. File -> Import Notebook -> upload `shortcut-learning-mnist.ipynb`
4. Run All

You can also run it locally if you have Python with the packages below.
Open the file in Jupyter and Run All.

## Required packages

- torch
- torchvision
- numpy
- pandas
- matplotlib
- scikit-learn

On Kaggle these are all preinstalled.
Locally you can install them with:
pip install torch torchvision numpy pandas matplotlib scikit-learn

## Where the main code is

Everything is inside `shortcut-learning-mnist.ipynb`. The notebook goes
top to bottom in this order:

- Imports and MNIST loading
- Train / validation split
- Small CNN model class
- Training loop (baseline model, 5 epochs)
- Evaluate on the clean MNIST test set
- One rotation check at 60 degrees
- Rotation sweep (0 to 90 degrees in 15 degree steps)
- Gaussian noise sweep
- Qualitative examples (clean vs rotated vs noisy)
- Augmented model training (random rotation + noise during training)
- Comparison of baseline vs augmented on the rotation sweep

The most important results are the rotation sweep and the baseline vs
augmented comparison at the end of the notebook.

## Results summary

- Clean MNIST test accuracy: about 98.80%
- Test accuracy after rotating every image by 60 degrees: about 22.72%
- Test accuracy after rotating every image by 90 degrees: about 13.69%
- Augmented model (trained with random rotation + noise) stays around
  90-97% across the same rotation sweep, which supports the shortcut
  learning idea.
