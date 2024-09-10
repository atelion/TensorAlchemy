<div align="center">

![TensorAlchemy - Splash image](../../TensorAlchemy-splash.png)

# **TESTNET GUIDE FOR VALIDATORS** <!-- omit in toc -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

This is documentation for starting and running a validator on the Tensor Alchemy testnet. Note that we are continuing to push and test updates so you may notice the miner and/or validator encounter issues.

## **Important note**: You will need **1025 test tao to validate**.

## VALIDATOR REQUIREMENTS
- ⚗️ NVIDIA 3070 GPU or better (4GB~ VRAM requirement)
- ⚗️ Ubuntu 20.04 or 22.04
- ⚗️ Python 3.8, 3.9 or 3.10
- ⚗️ CUDA 12.0 or higher

### STEP 1. CLONE REPOSITORY
```bash
git clone https://github.com/TensorAlchemy/TensorAlchemy.git ~/TensorAlchemy
```

### STEP 2. INSTALL PREREQUISITES
```bash
sudo apt-get update && sudo apt-get install python3.10-venv
```

### STEP 3. CREATE A VENV
```bash
python3.10 -m venv ~/venvs/TensorAlchemy && source ~/venvs/TensorAlchemy/bin/activate && pip install wheel && pip install --upgrade setuptools
```

Important note: The recommended python version for TensorAlchemy is 3.10 on Ubuntu 22.04.
You may need to change the name of the venv to TensorAlchemyValidator if testing both miner and validator on the same server.

### STEP 4. INSTALL REQUIREMENTS INTO VENV
```bash
cd ~/TensorAlchemy && pip install -r requirements.txt
```

### STEP 5. Export API key
```bash
export OPENAI_API_KEY=...
```

### STEP 6. LAUNCH VALIDATOR
Running without manual validator
```bash
python ~/TensorAlchemy/neurons/validator/main.py --wallet.name NAME --wallet.hotkey HOTKEY --netuid 25 --subtensor.network test --axon.port 8000 --logging.debug --logging.trace
```

### Running with manual validator
```bash
cd  ~/TensorAlchemy

python neurons/validator/main.py --wallet.name NAME --wallet.hotkey HOTKEY --netuid 25 --subtensor.network test --axon.port 8000 --logging.debug --logging.trace
```

Important note:  Your must run your start command (step 6) within the TensorAlchemy folder if you are running a manual validator
