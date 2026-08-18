# PPI-Flow: Efficient Taxonomy Tagging for Educational Content

PPI-Flow is an end-to-end framework for automating and calibrating hierarchical taxonomy classification of educational content. The project addresses the challenges of dynamic educational taxonomies, label imbalance, and noisy weak supervision by integrating dense retrieval models, Large Language Models (LLMs), and statistical calibration techniques.

This repository contains the implementation of the concepts detailed in the research project "Efficient Taxonomy Tagging for Educational Content" by Harsh Rajput (IIIT Delhi).

## Features

- **Automated Taxonomy Tagging (TaxTag)**: An efficient taxonomy-aware dense retrieval model built on an E5-base-v2 backbone with Low-Rank Adaptation (LoRA). It aligns question embeddings with taxonomy label representations using an InfoNCE-based contrastive objective and hierarchy-aware smoothing.
- **LLM-Based Evaluation**: Utilizes LLMs (such as `openai/gpt-oss-120b`) as automated judges to score the semantic alignment between questions and predicted taxonomy labels across multiple hierarchical levels (Subject, Chapter, Topic).
- **Local Prediction-Powered Inference (L-PPI)**: A novel instance-level calibration method that corrects systematic biases in LLM-generated confidence scores. L-PPI computes locally calibrated scores and confidence intervals using a small human-annotated seed set in a prediction-aware embedding space.
- **Iterative Retraining**: A scalable pipeline that filters weakly supervised pseudo-labels based on calibrated L-PPI scores and confidence intervals to produce high-quality training datasets for iteratively improving the taxonomy classifier.

## Repository Structure

- `data/`: Datasets, raw input questions, and processed files for calibration and evaluation.
- `llm/`: Unified Language Model interface with routing between local instances (Ollama) and cloud APIs (Groq). Contains the LLM judge implementation and prompting strategies.
- `main/`: Core execution scripts and Jupyter notebooks demonstrating the end-to-end inference and calibration pipeline.
- `models/`: PyTorch-based neural architectures, including the contrastive `TaxTag` dense retrieval model.
- `ppi/`: Implementation of the Local Prediction-Powered Inference (L-PPI) algorithm for estimating local bias and confidence intervals.
- `taxonomy/`: Parsed hierarchical taxonomy structures derived from the NCERT syllabus (Subject -> Chapter -> Topic) and relevant utilities.
- `utils/`: Data loaders and comprehensive evaluation modules for measuring rectification performance (MAE, improvement rate, directional correctness).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/PPI-Flow.git
   cd PPI-Flow
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Setup external dependencies:
   - For local LLM evaluation, install and configure Ollama.
   - For cloud-based evaluation, ensure your Groq or OpenAI API keys are correctly set in the environment.

## Usage

To execute the core pipeline, run the main script:

```bash
python main/main.py
```

### End-to-End Workflow

1. **Prediction**: The `TaxTag` model generates top-k candidate taxonomy paths for unlabeled questions.
2. **Scoring**: The LLM judge evaluates the semantic alignment of these candidate paths and assigns raw validity probabilities.
3. **Rectification**: The `L-PPI` module calibrates the LLM scores using a human-labeled gold set, producing bias-corrected probabilities and rigorous confidence intervals.
4. **Filtering**: High-confidence question-tag pairs are isolated using confidence-based thresholding.
5. **Retraining**: The filtered, high-quality pseudo-gold dataset is used to iteratively fine-tune the `TaxTag` model.

## Documentation

For a detailed theoretical breakdown of the methodology, experimental setup, and quantitative results, please refer to the project report: `Efficient Taxonomy Tagging for Educational Content.pdf`.

## License

This project is open-source and available under the MIT License.
