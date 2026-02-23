# OCR/HTR Models Evaluation on IAM Database and Extended Datasets

## Project Overview

This project aims to evaluate and benchmark various Optical Character Recognition (OCR) and Handwriting Text Recognition (HTR) models on the IAM Handwriting Database and other complementary datasets. The objective is to systematically assess model performance, accuracy, speed, and suitability for different use cases.

## Objectives

1. **Model Evaluation**: Assess performance metrics of multiple state-of-the-art OCR/HTR models
2. **Comparative Analysis**: Compare models across various dimensions (accuracy, inference time, resource usage)
3. **Dataset Coverage**: Evaluate models on IAM database and extended datasets to ensure comprehensive assessment
4. **Benchmarking**: Establish baseline performance metrics for future improvements
5. **Documentation**: Provide detailed analysis and recommendations for model selection

## Datasets

### Primary Dataset
- **IAM Handwriting Database**
  - Description: Large-scale database of handwritten text
  - Coverage: Various writing styles and quality levels
  - Size: ~115,000 isolated handwritten words and ~50,000 text lines
  - Access: Requires registration at IAM website

### Secondary Datasets (To be determined)
- Additional publicly available handwriting or document datasets
- Potentially includes:
  - MNIST/EMNIST
  - RIMES dataset
  - CVL dataset
  - Custom domain-specific datasets (if applicable)

## Models to Evaluate

### OCR Models
- [ ] Tesseract
- [ ] Kraken
- [ ] DeepSeek OCR
- [ ] Chandra

### HTR Models
- [ ] PyLaia
- [ ] Qwen2-VL 8B (Vision-Language)

## Evaluation Metrics

### Accuracy Metrics
- Character Error Rate (CER): % of character-level errors
- Word Error Rate (WER): % of word-level errors
- Sequence Error Rate (SER): % of completely incorrect sequences
- Confidence scores (if applicable)

### Performance Metrics
- Inference time per image/sample
- GPU/CPU memory usage
- Throughput (samples per second)
- Model size (MB)

### Robustness Metrics
- Performance on degraded/low-quality images
- Performance on different writing styles
- Performance on different document types

## Methodology

### Pre-HPC Setup (Local/Interactive)
1. **Data Download**: Run `setup/download_data.sh` to fetch IAM and other datasets
2. **Model Preparation**: Run `setup/download_models.sh` to download/prepare all models
3. **Dependencies**: Install requirements via `setup/setup_environment.sh`
4. **Validation**: Test one model locally before HPC submission

### Data Preparation
1. Standardize image preprocessing across all models (in `evaluators/utils.py`)
2. Define train/validation/test splits
3. Document any augmentation techniques used
4. Ensure consistent evaluation protocol

### HPC Evaluation Process
1. Submit job via `hpc/submit_job.sh` to Adroit
2. SLURM job calls `hpc/run_evaluation.py` orchestrator
3. Orchestrator sequentially runs each model-specific evaluator
4. Each evaluator:
   - Loads pre-downloaded model from `models/` directory
   - Processes data from `data/` directory
   - Writes results to `results/` directory
   - Logs metrics and inference times
   - Documents any failures or edge cases

### Analysis & Reporting (Post-HPC)
1. Download results from Adroit
2. Compare results across models using Jupyter notebooks
3. Generate visualizations (charts, graphs)
4. Identify strengths and weaknesses of each model
5. Provide recommendations based on use case

## Project Structure

```
not_flawless/
├── SPEC.md (this file)
├── README.md
├── requirements.txt
├── setup/
│   ├── download_data.sh          # Download IAM and other datasets
│   ├── download_models.sh        # Download/prepare pre-trained models
│   └── setup_environment.sh      # Install dependencies
├── data/
│   ├── iam/                      # IAM database (downloaded)
│   ├── other_datasets/           # Additional datasets (downloaded)
│   └── processed/                # Preprocessed data (generated during job)
├── models/                       # Pre-downloaded models
│   ├── tesseract/
│   ├── paddleocr/
│   ├── easyocr/
│   └── [other_models]/
├── evaluators/                   # Model-specific evaluation scripts
│   ├── tesseract_eval.py
│   ├── paddleocr_eval.py
│   ├── easyocr_eval.py
│   ├── keras_ocr_eval.py
│   ├── trocr_eval.py
│   └── utils.py                  # Shared utility functions
├── hpc/
│   ├── submit_job.sh             # Main SLURM submission script
│   ├── run_evaluation.py          # Orchestrator for all evaluations
│   └── job_config.txt            # HPC parameters
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_evaluation.ipynb
│   └── 03_results_analysis.ipynb
├── results/                      # Output directory (generated)
│   ├── metrics/
│   ├── visualizations/
│   ├── logs/
│   └── reports/
└── .gitignore
```

## Timeline & Milestones

- [ ] **Phase 1**: Finalize model list, dataset requirements
- [ ] **Phase 2**: Create setup scripts for data/model downloads
- [ ] **Phase 3**: Implement model-specific evaluator scripts
- [ ] **Phase 4**: Create HPC orchestrator and SLURM submission script
- [ ] **Phase 5**: Local testing and validation
- [ ] **Phase 6**: Submit and run HPC evaluation job
- [ ] **Phase 7**: Analysis and report generation

## Success Criteria

- [ ] All selected models successfully evaluated
- [ ] Comprehensive metrics collected for each model
- [ ] Clear ranking/comparison of models
- [ ] Documented findings and recommendations
- [ ] Reproducible evaluation pipeline

## Risks & Mitigation

| Risk | Mitigation |
|------|-----------|
| Data access issues | Multiple dataset sources, pre-downloaded mirrors |
| Model dependency conflicts | Docker containers, virtual environments |
| Computational constraints | GPU access, model quantization if needed |
| Inconsistent evaluation | Standardized evaluation pipeline, automated testing |

## Future Work

- Fine-tune selected models on custom datasets
- Ensemble approaches
- Real-time inference optimization
- Integration with document processing pipelines
- Extended language support evaluation

## References

- IAM Database: https://fki.ics.unimaas.nl/databases/iam-handwriting-database/
- [Additional research papers and documentation TBD]

## Contact & Contributors

[To be populated]
