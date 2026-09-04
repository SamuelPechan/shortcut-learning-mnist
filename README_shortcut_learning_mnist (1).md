# Testing CNN Robustness on MNIST

This project investigates how well a convolutional neural network (CNN) trained on standard MNIST images performs when the input distribution changes. Although the baseline model achieves high accuracy on clean test data, its performance drops substantially when the digits are rotated or corrupted with Gaussian noise.

I then train a second model using randomized rotations and noise augmentation to test whether exposure to these variations during training improves robustness.

This project was completed as a final project for CS 4210.

## Key Result

The baseline CNN achieved **98.85% accuracy** on clean MNIST images but only **24.93% accuracy** when the images were rotated by 60 degrees. After training with randomized rotations and Gaussian noise, accuracy on the same 60-degree test set increased to **96.39%**, while clean-image accuracy remained **96.83%**.

## Research Question

Does high accuracy on the standard MNIST test set mean that a model has learned features that remain reliable when the appearance of the digits changes?

The experiment evaluates whether the model depends heavily on properties of the original training distribution, such as upright orientation and clean backgrounds.

## Methodology

1. Load the MNIST training and test datasets with Torchvision.
2. Split the original training set into training and validation subsets.
3. Train a baseline CNN on clean, upright MNIST images for five epochs.
4. Evaluate the baseline model on the clean test set.
5. Measure baseline accuracy across rotations from 0 to 90 degrees.
6. Measure baseline accuracy across increasing levels of Gaussian noise.
7. Train a second CNN using random rotations between -90 and 90 degrees and Gaussian noise augmentation.
8. Compare the baseline and augmented models across the same rotation sweep.

Both models use the same CNN architecture so that the primary experimental difference is the training data.

## Model Architecture

The CNN contains:

- Two convolutional layers
- Batch normalization after each convolution
- ReLU activation functions
- Max pooling after each convolutional block
- Dropout before the fully connected layers
- Two fully connected layers
- Ten output classes for digits 0 through 9

Training configuration:

- Optimizer: Adam
- Learning rate: 0.0005
- Loss function: cross-entropy loss
- Batch size: 64
- Epochs: 5
- Random seed: 42

## Results

### Baseline Performance Under Rotation

| Rotation | Baseline accuracy |
|---:|---:|
| 0 degrees | 98.85% |
| 15 degrees | 95.76% |
| 30 degrees | 78.82% |
| 45 degrees | 52.94% |
| 60 degrees | 24.93% |
| 75 degrees | 15.77% |
| 90 degrees | 14.47% |

The baseline model performs well on small rotations, but accuracy declines sharply as the rotation moves farther from the upright images seen during training.

### Baseline Performance Under Gaussian Noise

| Noise standard deviation | Baseline accuracy |
|---:|---:|
| 0.00 | 98.85% |
| 0.10 | 98.07% |
| 0.20 | 82.99% |
| 0.30 | 51.58% |
| 0.50 | 22.62% |
| 0.75 | 12.99% |
| 1.00 | 11.75% |

Performance remains strong under light noise but deteriorates as the noise increasingly obscures the image structure.

### Baseline vs. Augmented Model

| Rotation | Baseline accuracy | Augmented accuracy |
|---:|---:|---:|
| 0 degrees | 98.85% | 96.83% |
| 15 degrees | 95.76% | 96.39% |
| 30 degrees | 78.82% | 96.51% |
| 45 degrees | 52.94% | 96.50% |
| 60 degrees | 24.93% | 96.39% |
| 75 degrees | 15.77% | 95.79% |
| 90 degrees | 14.47% | 91.15% |

The augmented model sacrifices approximately two percentage points of clean-image accuracy but remains much more accurate across the rotation sweep. This demonstrates how training-data diversity can improve robustness to changes that are not represented in the original training distribution.

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Jupyter Notebook

## How to Run

### Option 1: Kaggle

1. Open [Kaggle Notebooks](https://www.kaggle.com/code).
2. Create a new notebook.
3. Import `shortcut-learning-mnist.ipynb`.
4. Select **Run All**.

Kaggle includes the required packages and can provide GPU acceleration when available.

### Option 2: Local installation

Clone the repository:

```bash
git clone https://github.com/SamuelPechan/shortcut-learning-mnist.git
cd shortcut-learning-mnist
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `shortcut-learning-mnist.ipynb` and select **Restart & Run All**.

MNIST is downloaded automatically through Torchvision the first time the notebook runs.

## Repository Contents

```text
shortcut-learning-mnist/
|-- README.md
|-- requirements.txt
`-- shortcut-learning-mnist.ipynb
```

## Limitations

MNIST is a relatively simple image-classification dataset, so these findings may not generalize directly to more complex computer-vision tasks. Large rotations can also make certain digits difficult to interpret or introduce label ambiguity, particularly for digits such as 6 and 9.

The experiment therefore demonstrates sensitivity to distribution shift and orientation changes. It does not establish that every performance failure is caused by shortcut learning. Additional experiments with other datasets, transformations, model architectures, and multiple training runs would be needed to make broader conclusions.

## Conclusion

High performance on a standard test set does not guarantee that a model will remain reliable when its inputs differ from the training distribution. The baseline CNN reached 98.85% accuracy on clean MNIST images but failed under larger rotations and stronger noise. Training with randomized transformations substantially improved rotational robustness while preserving high clean-image accuracy.

The results highlight the importance of testing machine-learning systems beyond their original data distribution and designing training data that reflects the conditions a model may encounter after deployment.
