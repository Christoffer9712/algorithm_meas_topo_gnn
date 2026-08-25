Project: Overlay Flow Scheduler (Predictor + MILP)

This repository implements a supervised predictor for overlay path characteristics
(packet-loss and delay) and a MILP-based flow scheduler that uses those
predictions to assign flows to overlay paths over a receding horizon.

Overview

- Predictor (meas_predictor.py)
  - MeasurementEmbedder: attention-based embedding over recent measurements
    conditioned on a topology embedding h_pred.
  - Predictor: MLP that consumes H = [h_pred, g, horizon_m] and outputs
    (predicted_packet_loss, predicted_delay_ms).
  - The predictor now uses known future ephemeris: at inference the GAT-based
    topology embedding h_pred is computed for each future horizon step m using
    the known positions (the simulator's env is deep-copied and stepped to
    compute future snapshots). The predictor takes a scalar horizon m as a
    staleness indicator so the model can learn to rely more on topology for
    larger m.

- Scheduler (scheduler_milp.py + scheduler.py)
  - MILP (OR-Tools/CBC) that solves a receding-horizon assignment mu[f,p,m].
  - Objective includes normalized penalties for delay and packet-loss and a
    path-switching cost. Per-metric exceedances are clipped to 1.0 to avoid
    domination by very small requirements.

Repository layout (important files)

- main.py
  - Top-level entry. Edit RUN_MODE at the top to one of:
    'train_predictor' - generate dataset and train predictor
    'predictor_only'  - run inference across simulator snapshots (requires trained models)
    'full'            - run full predictor -> MILP test loop (requires trained models)
  - The full run computes future topology embeddings for horizon steps using
    the simulator's known ephemeris and uses the predictor outputs in the MILP.

- network.py
  - Topology generator (LayeredOrbitNetwork). Use this to customize scenario
    geometry, add multiple aircraft, or set overlay bandwidths.

- nodeQueues.py
  - Queueing model used by the simulator to produce ground-truth delays and losses.

- pathEncoder.py
  - Graph encoder (GATv2) used to compute per-overlay topology embeddings.
    It supports adding a virtual overlay node per overlay (edges to overlay
    nodes use special features) and returns per-overlay vectors h_k^p.

- meas_predictor.py
  - MeasurementEmbedder and Predictor (see Overview). Predictor input H now
    includes the horizon scalar.

- scheduler_milp.py
  - MILP formulation using OR-Tools CBC. See code comments for decision variable
    definitions and constraints.

- scheduler.py
  - Wrapper that exposes map_flows(...), cost(...) and integrates with the
    environment API.

- trainer/
  - trainer/generate_dataset.py: runs the simulator and encoder to generate
    supervised samples for multiple prediction horizons (M). Saves to
    data/predictor_dataset.pt.
  - trainer/train_predictor.py: trains MeasurementEmbedder + Predictor and
    saves models/predictor_models.pth.
  - trainer/dataset.py: Dataset wrapper used by the trainer.

Dependencies and installation

- Python 3.8+ (recommended 3.10+)
- PyTorch
- PyTorch Geometric (and its dependencies: torch-scatter, torch-sparse, etc.)
- OR-Tools (CBC backend)

Suggested installs (example for WSL/Ubuntu):

- Install PyTorch (CPU) — see https://pytorch.org for the correct command for your environment.
- Install PyTorch Geometric following its install guide (matching your torch version).
- Install OR-Tools (on the environment you will run the code in):
    python -m pip install ortools

Notes about Windows + WSL paths

- The repository may live on a WSL filesystem. When running Python from
  Windows-side Python, file paths that point into WSL must be provided as
  UNC paths (for example: //wsl.localhost/Ubuntu-22.04/home/chris/algorithm_test/data/...).
  To avoid path issues, prefer running the scripts from inside the same
  environment where the packages are installed (e.g., run inside WSL if your
  Python + packages are installed in WSL).

Quickstart: generate dataset, train predictor, run full test

1) Generate dataset and train (recommended to run inside the repo root):

   - Option A: using the main.py RUN_MODE
     Edit the top of main.py and set:
       RUN_MODE = 'train_predictor'
     Then run:
       python main.py
     This will generate data/predictor_dataset.pt and train models/predictor_models.pth.

   - Option B: run generator and trainer directly (examples used during development):
       python -c "import sys; sys.path.append(r'//wsl.localhost/Ubuntu-22.04/home/chris/algorithm_test'); from trainer.generate_dataset import generate; generate(T=400, Delta=10, seed=1)"
       python -c "import sys; sys.path.append(r'//wsl.localhost/Ubuntu-22.04/home/chris/algorithm_test'); from trainer.train_predictor import train; train(dataset_path='//wsl.localhost/Ubuntu-22.04/home/chris/algorithm_test/data/predictor_dataset.pt', epochs=10, batch_size=64)"

   Adjust T, Delta, M and epochs as desired.

2) Run predictor-only inference (useful for debugging):
   - Set RUN_MODE = 'predictor_only' in main.py and run python main.py
   - The script loads models/predictor_models.pth and prints per-overlay predictions.

3) Run full predictor->MILP loop:
   - Set RUN_MODE = 'full' in main.py and run python main.py
   - Requires models/predictor_models.pth to exist. The script will run a 50-step
     test loop and print per-step cost and assignments.

Where to tune parameters

- Delta (history length), M (prediction horizon) — set in trainer/generate_dataset.py
  and in main.py where the scheduler is invoked. Default Delta=10, M=5.
- Predictor architecture (hidden sizes) — meas_predictor.py
- Scheduler weights (c_l, c_d, c_s, gamma) and penalty clipping — scheduler_milp.py
- Overlay bandwidths — network.get_all_overlays() to add realistic B^p values

Troubleshooting

- FileNotFoundError for dataset:
  - Ensure you run generation before training, and run both generator and trainer
    in the same environment (WSL vs Windows Python). If your dataset path points
    into WSL, use the //wsl.localhost/... path when invoking Python from Windows.

- OR-Tools errors:
  - Make sure ortools is installed in the Python environment you are running
    the scripts from. On WSL, install there separately if needed.

- Checkpoint input-size mismatch:
  - The code includes a best-effort loader that copies compatible weight slices
    when the predictor input dimensionality changes between saves. For best
    results, retrain after making predictor input changes.

Development & extensions

- Horizon-aware dynamics: current predictor computes GAT topology embeddings
  at each future horizon step (ephemeris known) and consumes a scalar horizon
  m as a staleness indicator. This was implemented to let the predictor rely
  more on topology-derived features when measurements are stale.

- Future work suggestions:
  - Normalize predictor targets (delay in seconds or standardized targets)
    to improve training stability.
  - Replace environment deepcopy used to compute future embeddings with a
    deterministic future-snapshot API (fast and more robust outside the simulator).
  - Support flow splitting/duplication by converting binary mu variables to
    continuous fractions in the MILP.
  - Exact switching-cost modeling via auxiliary binary variables.

Contact

If anything is unclear, or you want a live demo or a notebook to visualize
predictions and schedules, ask and a minimal demo can be prepared.

Quick example commands (copy-paste)

- Run training inside WSL (recommended):

  cd /home/chris/algorithm_test
  python3 -u main.py    # with RUN_MODE='train_predictor' set in main.py

- Run training from Windows Python (when repo is on WSL):

  REM: adjust Python path and use UNC path to dataset/model files
  python -c "import sys; sys.path.append(r'//wsl.localhost/Ubuntu-22.04/home/chris/algorithm_test'); from trainer.generate_dataset import generate; generate(T=400,Delta=10,seed=1)"
  python -c "import sys; sys.path.append(r'//wsl.localhost/Ubuntu-22.04/home/chris/algorithm_test'); from trainer.train_predictor import train; train(dataset_path='//wsl.localhost/Ubuntu-22.04/home/chris/algorithm_test/data/predictor_dataset.pt', epochs=10, batch_size=64)"

- Run predictor-only inference (after training):

  Edit main.py and set RUN_MODE='predictor_only', then run:
  python main.py

- Run full predictor->MILP test (after training):

  Edit main.py and set RUN_MODE='full', then run:
  python main.py

Notebook: visualize_predictions.ipynb

- A minimal Jupyter notebook is provided at [visualize_predictions.ipynb](/home/chris/algorithm_test/visualize_predictions.ipynb).
  It loads the saved dataset and the trained models and plots predicted vs. true
  values for packet-loss and delay (ms). Run it with Jupyter from the repo root:

  jupyter notebook visualize_predictions.ipynb

Notes on environment and paths

- If running from Windows Python but the repository lives in WSL, make sure to
  reference dataset and model paths via the //wsl.localhost/... UNC path as in
  the examples above. Running everything inside the same environment (WSL or
  native) is simpler and recommended.

If you'd like, I can also add a small script that runs the notebook code as a
standalone .py script and saves the plots to disk (PNG)."}```