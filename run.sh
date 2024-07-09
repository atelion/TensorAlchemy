#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 neurons/miners/StableMiner/main.py --wallet.name testcold --wallet.hotkey testhot --netuid 25 --subtensor.network test --axon.port 15368
# python3 neurons/miners/StableMiner/main.py --wallet.name JJcold --wallet.hotkey JJb --netuid 26 --subtensor.network finney --axon.port 22033
# python3 neurons/miners/StableMiner/main.py --wallet.name JJcold --wallet.hotkey JJwarm --netuid 26 --subtensor.network finney --axon.port 22037
# python3 neurons/miners/StableMiner/main.py --wallet.name JJcold --wallet.hotkey JJa --netuid 26 --subtensor.network finney --axon.port 22024
# CUDA_VISIBLE_DEVICES=3 python3 neurons/miners/StableMiner/main.py --wallet.name JJcold --wallet.hotkey JJhot --netuid 26 --subtensor.network finney --axon.port 22059

