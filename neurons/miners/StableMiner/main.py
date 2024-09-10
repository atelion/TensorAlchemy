import os
import pathlib
import sys
import warnings

# Suppress the eth_utils network warnings
# "does not have a valid ChainId."
# NOTE: It's not our bug, it's upstream
# TODO: Remove after updating bittensor
warnings.simplefilter("ignore")

# Use the older torch style for now
os.environ["USE_TORCH"] = "1"


def setup_paths():
    file_path: str = str(
        pathlib.Path(__file__).parent.parent.parent.parent.resolve(),
    )
    if file_path not in sys.path:
        sys.path.append(file_path)


def main():
    setup_paths()
    from neurons.miners.StableMiner.run_miner import run_miner

    run_miner()


if __name__ == "__main__":
    main()
