"""
Entry point for deployment compatibility.
This wrapper allows the app to run on various cloud platforms.
"""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app.py"],
        cwd=".",
    )
