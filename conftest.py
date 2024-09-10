import os
import sys

# Get the absolute path of the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))

# Add the project's root directory to the Python path
sys.path.insert(0, project_root)

# Use the older torch style for now
os.environ["USE_TORCH"] = "1"
os.environ["CI"] = "true"
