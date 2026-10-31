"""
Root wrapper for generate_model_focus_heatmap.py.
Forwards directly to scripts/generate_model_focus_heatmap.py
"""
import sys
from pathlib import Path

# Add scripts directory to path and execute main
scripts_dir = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from generate_model_focus_heatmap import main

if __name__ == "__main__":
    main()
