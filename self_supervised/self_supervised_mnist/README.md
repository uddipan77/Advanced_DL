# Self-Supervised Learning on MNIST

This project demonstrates a self-supervised learning approach on the MNIST dataset. The main goal is to train a model using a pretext task (rotation prediction) and then fine-tune the model for digit classification.

## Project Structure

```plaintext
self_supervised_mnist/
│
├── main.py
│
├── data/
│   ├── __init__.py
│   └── dataset.py
│
├── model/
│   ├── __init__.py
│   └── model.py
│
├── train/
│   ├── __init__.py
│   └── train.py
│
└── finetune/
    ├── __init__.py
    └── finetune.py





### Overview of Files

- **main.py**: The entry point of the project. It first trains the model on the rotation prediction task and then fine-tunes it on digit classification.

- **data/dataset.py**: Contains the custom dataset class `RotatedMNISTDataset` which augments the MNIST dataset for the rotation prediction task.

- **model/model.py**: Defines the Convolutional Neural Network (CNN) architecture used for both the pretext task and digit classification.

- **train/train.py**: Contains the training logic for the rotation prediction task, which serves as the pretext task.

- **finetune/finetune.py**: Handles the fine-tuning of the model on digit classification using the pre-trained weights from the pretext task.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- torchvision
- Other dependencies can be installed via `requirements.txt` (if available)

### Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/uddipan77/your-repo-name.git
    cd your-repo-name
    ```

2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Download the MNIST Dataset:**

    The MNIST dataset will be automatically downloaded to `C:/Users/uddip/Downloads/MNIST` when you run the training script.

### Running the Project

1. **Train the Model on the Pretext Task (Rotation Prediction) and Fine-Tune:**

    ```bash
    python main.py
    ```

### Project Workflow

- **Step 1**: The model is trained on a pretext task (rotation prediction) using the rotated MNIST dataset.
- **Step 2**: The model is fine-tuned on the MNIST digit classification task using the pre-trained weights from the pretext task.

### Model Architecture

The CNN model consists of two convolutional layers followed by fully connected layers. For the pretext task, the model predicts four possible rotation angles (0°, 90°, 180°, 270°). For digit classification, the final layer is modified to predict 10 classes (digits 0-9).

## Results

After fine-tuning, the model achieves an accuracy of approximately **XX.XX%** on the MNIST test set (replace with your actual results).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The PyTorch framework
- The creators of the MNIST dataset
