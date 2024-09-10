from loguru import logger


def broken_pipe_message() -> None:
    logger.error(
        """
===========================================================================
                        BittensorBrokenPipe Error
===========================================================================
A connection issue with Bittensor has been detected. This is likely due to
a problem with your Subtensor connection.

Action required:
1. Check your Subtensor connection and ensure it's running without error.
2. Verify your network connectivity.
3. Ensure your Subtensor node is up-to-date and running correctly.

Please try to update your subtensor to the latest version!

The validator will attempt to restart to resolve this issue.
If the problem persists, please consult the Bittensor documentation or
seek assistance from the community support channels.
===========================================================================
"""
    )


class BittensorBrokenPipe(Exception):
    def __init__(self):
        broken_pipe_message()
