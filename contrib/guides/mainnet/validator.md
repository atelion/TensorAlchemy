<div align="center">

![TensorAlchemy - Splash image](../../TensorAlchemy-splash.png)

# **MAINNET GUIDE FOR VALIDATORS** <!-- omit in toc -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>


This documentation is for starting and running a validator on the TensorAlchemy mainnet.


## VALIDATOR REQUIREMENTS
- ⚗️ NVIDIA 3070 GPU or better (4GB~ VRAM requirement)
- ⚗️ Ubuntu 20.04 or 22.04
- ⚗️ Python 3.8, 3.9 or 3.10
- ⚗️ CUDA 12.0 or higher


Below are several guides which include instructions to get validating with the Tensor Alchemy subnet. We recommend using a local subtensor for best results.

## GUIDE 1 (venv)

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

### STEP 4. INSTALL REQUIREMENTS INTO VENV
```bash
source ~/venvs/TensorAlchemy/bin/activate && cd ~/TensorAlchemy && pip install -r requirements.txt
```

### STEP 5. Export API key
```bash
export OPENAI_API_KEY=KEY
```

### STEP 6. LAUNCH VALIDATOR
```bash
python ~/TensorAlchemy/neurons/validator/main.py --wallet.name NAME --wallet.hotkey HOTKEY --netuid 26 --subtensor.network finney --axon.port 8101 --alchemy.device cuda:0 --logging.debug --logging.trace
```

### NOTES
- ⚗️ You may want to use a different name for the validator’s venv if you intend to also run a miner on the same machine.

## PREREQUISITES FOR GUIDE 2 & 3

### INSTALL REQUIRED PACKAGES

```bash
sudo apt-get update && sudo apt-get install curl -y && curl https://raw.githubusercontent.com/creationix/nvm/master/install.sh | bash && source ~/.bashrc && nvm install node && npm install -g npm && npm install pm2 -g && curl -sL https://raw.githubusercontent.com/Unitech/pm2/master/packager/setup.deb.sh | sudo -E bash - && echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf && sudo sysctl -p && pm2 install pm2-logrotate && pm2 set pm2-logrotate:max_size 50M && pm2 startup && pm2 save && pm2 ls
```

### SET UP START UP SCRIPT
The final output may tell you "To set up the Startup Script, copy paste the following command..."
Copy paste it and hit enter. If you don’t see this message, you can move on.

### REMOVE THE KEYMETRICS APT REPOSITORY
```bash
sudo rm /etc/apt/sources.list.d/Keymetrics_pm2.list
```


## GUIDE 2 (pm2 + venv)

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

### STEP 4. INSTALL REQUIREMENTS INTO VENV
```bash
source ~/venvs/TensorAlchemy/bin/activate && cd ~/TensorAlchemy && pip install -r requirements.txt
```

### STEP 5. Export API key
```bash
export OPENAI_API_KEY=KEY
```

### STEP 6. LAUNCH VALIDATOR
```bash
pm2 start ~/TensorAlchemy/neurons/validator/main.py --interpreter ~/venvs/TensorAlchemy/bin/python --restart-delay 30000 --name NAME -- --wallet.name NAME --wallet.hotkey HOTKEY --netuid 26 --subtensor.network finney --axon.port 8101 --alchemy.device cuda:0 --logging.debug --logging.trace
```

## GUIDE 3 (pm2 + conda)

### STEP 1. CLONE REPOSITORY
```bash
git clone https://github.com/TensorAlchemy/TensorAlchemy.git ~/TensorAlchemy
```

### STEP 2. INSTALL CONDA

#### Change to root user directory and download Anaconda
```bash
cd && wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
```

#### Change permissions on the Anaconda installer
```bash
chmod 777 ~/Anaconda3-2023.09-0-Linux-x86_64.sh
```

#### Install Anaconda
```bash
./Anaconda3-2023.09-0-Linux-x86_64.sh
```

#### Follow the prompts.

- Enter. 
- Accept agreement. 
- Enter. 
- Type yes.

#### Create conda environment
```bash
conda create -n py310 python=3.10
```

#### Activate conda env
```bash
conda activate py310
```

#### Prevent conda from activating automatically (optional):
```bash
conda config --set auto_activate_base false
```

### STEP 3. INSTALL REQUIREMENTS INTO VENV
```bash
cd ~/TensorAlchemy && pip install -r requirements.txt
```

### STEP 4. Export API key
```bash
export OPENAI_API_KEY=KEY
```

### STEP 5. LAUNCH VALIDATOR

```bash
pm2 start ~/TensorAlchemy/neurons/validator/main.py --interpreter ~/anaconda3/envs/py310/bin/python --restart-delay 30000 --name NAME -- --wallet.name NAME --wallet.hotkey HOTKEY --netuid 26 --subtensor.network finney --axon.port 8101 --alchemy.device cuda:0 --logging.debug --logging.trace
```
