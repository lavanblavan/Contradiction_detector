# Semantic Contradiction Detector

A specialized Natural Language Processing (NLP) pipeline designed to identify logical and semantic contradictions within single documents (e.g., product reviews, customer feedback, or claims). This project utilizes a **Dual-Expert Ensemble** approach, leveraging both RoBERTa and DeBERTa architectures to catch complex performance, sentiment, and temporal conflicts.

## 🚀 Key Features

* **Ensemble Reasoning**: Combines `roberta-large-mnli` for sentiment/performance nuances and `DeBERTa-v3` for logical, numerical, and temporal reasoning.
* **GPU Acceleration**: Optimized with `torch.float16` (Half-Precision) support to maximize throughput on modern CUDA-enabled hardware.
* **Detailed Results**: Returns a boolean contradiction flag, a confidence score (0-1), and the specific sentence pairs that triggered the detection.
* **Automated Preprocessing**: Cleans and segments raw text into logical claims for pair-wise analysis.

## 🛠️ Project Structure

* `Contradict_detector.py`: The core Python implementation containing the `SemanticContradictionDetector` class.
* `requirements.txt`: List of Python dependencies required to run the project.
* `sample_data.py`: A structured test set for immediate evaluation.
* `dataset.txt`: Raw text data used for bulk analysis.

## 📋 Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/lavanblavan/contradiction-detector.git](https://github.com/lavanblavan/contradiction-detector.git)
    cd contradiction-detector
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For GPU support, ensure you have the appropriate [PyTorch CUDA](https://pytorch.org/get-started/locally/) version installed for your hardware.*

## 💻 Usage

The detector is designed as a modular class that can be integrated into larger data processing pipelines.

```bash
python Contradict_detector.py
