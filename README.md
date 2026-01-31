# MNIST Local-ML Pipeline

Overview
--------
MNIST Local-ML Pipeline is a compact, self-contained example demonstrating training and inference workflows for a small neural network on the MNIST handwritten digits dataset. The repository is organized for quick experimentation and learning: it shows dataset loading, pre-processing, model training, checkpointing, and inference.

This project is intended for developers and learners who want a lightweight, easy-to-follow PyTorch example that runs on a local Windows machine.

Repository Structure
--------------------
- `main.py`: Entrypoint — CLI-style runner that wires training, testing, and inference flows and exposes common flags for hyperparameters.
- `nn_training.py`: Training and evaluation logic — model definition (small MLP by default), training/validation loops, checkpointing, and metrics.
- `utils.py`: Shared utilities — dataset loaders for IDX files, normalization helpers, save/load helpers for model weights and normalization parameters.
- `neural_net_weights.pth`: Example saved model weights (PyTorch checkpoint).
- `norm_parameters.pth`: Saved normalization/scaling parameters used during preprocessing.
- `requirements.txt`: Python package dependencies for the project.
- `LICENSE`: Project license.
- `mnist_dataset/`: Local copy of MNIST IDX files (if present):
  - `train-images.idx3-ubyte`
  - `train-labels.idx1-ubyte`
  - `t10k-images.idx3-ubyte`
  - `t10k-labels.idx1-ubyte`

Key Features
------------
- Reproducible pipeline: read IDX files → normalize → train → save weights/normalization parameters.
- Checkpointing: save `neural_net_weights.pth` and `norm_parameters.pth` for later inference or fine-tuning.
- Minimal dependencies and clear code separation for easy modification and extension.

Getting Started (Windows)
-------------------------
These instructions assume Windows PowerShell and Python 3.10+ installed.

1. Clone the repository or copy files locally.

2. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```
4. Run server using uvicorn main:app or other command with "main" as main file.

Endpoints
---------
- `POST /predict`: accepts .PNG file type only. Returns JSON with predicted digit and confidence, e.g. `{"message":"predicted number:7,probability out of 10 numbers:0.98"}`.

5. Create or edit configuration flags (learning rate, epochs, batch size) in `nn_training.py` as needed.

6. If you want to compute training parameters yourself read and run `nn_training.py`.

Configuration & Environment
---------------------------
- `requirements.txt`: install the packages listed (e.g., `torch`, `numpy`).
- Hyperparameters and runtime options are exposed via `main.py` flags; you can also hardcode defaults in `nn_training.py`.
- If a CUDA-compatible GPU is available and PyTorch is installed with CUDA support, training will use the GPU automatically when detected.

Usage
-----
Typical flow:

1. Ensure MNIST IDX files are present in `mnist_dataset/` (or add a downloader to `utils.py`).
2. Train the model with `python main.py --train` to produce `neural_net_weights.pth` and `norm_parameters.pth`.
3. Start the server. 
4. Use `/predict` endpoint to load the photo and get the prediction.

Code Walkthrough
----------------
- `utils.py`: IDX file parsing, normalization, helper I/O for saving/loading numpy/PyTorch artifacts.
- `nn_training.py`: Model definition (small MLP), loss/optimizer setup, training loop, validation, checkpoint saving.
- `main.py`: CLI entrypoint that connects the components.

Data and Artifacts
------------------
- The `mnist_dataset/` folder contains the MNIST IDX files used for training and evaluation.
- Checkpoints and normalization parameters are stored as:
  - `neural_net_weights.pth` — model checkpoint
  - `norm_parameters.pth` — saved normalization parameters

Security & Privacy
------------------
- This repository uses the public MNIST dataset and does not collect user data by default.
- If you adapt the code to work with private datasets, ensure secure storage and handling of sensitive data.

Contributing
------------
Contributions are welcome. Suggested workflow:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Add or update code and include tests where applicable.
4. Open a pull request with a clear description of changes.

Suggested improvements:
- Add a small CNN model for higher accuracy.
- Add automated dataset downloading and verification in `utils.py`.
- Integrate training logging with TensorBoard or Weights & Biases.

License
-------
See the `LICENSE` file in the repository root for license terms.

Acknowledgements
----------------
This project is a compact educational example for local ML experimentation. Thanks to PyTorch and the maintainers of the MNIST dataset for providing widely used educational resources.

Contact
-------
If you have questions, feature requests, or bug reports, please open an issue in the repository.
