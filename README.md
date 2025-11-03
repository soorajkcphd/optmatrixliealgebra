# Optimization over Matrix Lie Algebras

This repository provides a Python implementation for optimization and analysis over **Matrix Lie Algebras**, including geometric learning and structure-preserving reinforcement learning experiments.

---

## Setup Instructions

### 1. Create the Conda Environment
```bash
conda create -n envoptliealgebra python=3.10
```

### 2. Activate the Environment
```bash
conda activate envoptliealgebra
```

### 3. Install Required Dependencies
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

---

## Run the Code
Once all dependencies are installed, you can run the main optimization script:
```bash
python OptimizeMatrixLieAlgebraTheory.py
```
*Expected runtime*:  ~45 minutes (GPU)

*Output*:
- Text report: `./focm_validation_final/report.txt`
- Plots: `./focm_validation_final/plots/`
- Results: `./focm_validation_final/results.pt`

---

## Configuration

Default settings (in `Config` class):
- `d=64`: Ambient dimension
- `n=8`: Number of transformations
- `m=100`: Number of data points
- `lambda_reg=1e-3`: Regularization parameter
- `pgd_iterations=3000`: PGD iterations
- `sgd_iterations=8000`: SGD iterations

Modify these in `main()` function as needed.
---

## Project Structure
```
.
├── OptimizeMatrixLieAlgebraTheory.py   # Main experiment script
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

---

## Notes
- Python version: **3.9**
- Recommended environment: **Conda**
- Ensure you have `pip` and `conda` updated to the latest versions.
- To update dependencies later:
  ```bash
  pip install -r requirements.txt --upgrade
  ```

---

## License
This project is released under the **MIT License**.  
Feel free to use and modify it for research or educational purposes.

---

## Citation
If you use this work in academic research, please cite:

Sooraj K.C, “Optimization over Matrix Lie Algebras,” 2025.  
GitHub repository: (https://github.com/soorajkcphd/optmatrixliealgebra)
