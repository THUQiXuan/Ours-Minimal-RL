

## Environment Setup
1. Create a new environment.
   ```bash
   conda create -n raftpp python==3.10
   conda activate raftpp
   ```
2. Install dependencies
   ```bash
   pip install pip --upgrade
   pip install uv
   python -m uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
   python -m uv pip install flash-attn --no-build-isolation
   git clone https://github.com/RLHFlow/Minimal-RL.git
   cd Minimal-RL/
   python -m uv pip install -e .
   python -m uv pip install vllm==0.6.3
   ```

## Experiments Running
1. Prepare the training and test datasets. (Already down)
    ```bash
    python scripts/data_preprocess/math_dataset.py
    python scripts/data_preprocess/numina_math.py
    ```
2. Start the training loop.
   ```bash
   bash scripts/run_grpo8.sh
   ```
