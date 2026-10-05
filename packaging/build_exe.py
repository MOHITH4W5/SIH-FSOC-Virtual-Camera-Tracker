"""
build_exe.py - Build script to compile SIH26169 into a standalone Windows executable.
"""

import os
import sys
import subprocess
import shutil

def build():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    spec_path = os.path.join(script_dir, "sih_fsoc_tracker.spec")

    print(f"Building standalone executable from spec: {spec_path}")
    print(f"Project root: {project_root}")

    dist_dir = os.path.join(project_root, "dist")
    build_dir = os.path.join(project_root, "build")

    # Command: pyinstaller --clean --noconfirm packaging/sih_fsoc_tracker.spec
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        "--distpath", dist_dir,
        "--workpath", build_dir,
        spec_path
    ]

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root)
    if result.returncode == 0:
        exe_path = os.path.join(dist_dir, "SIH_FSOC_Tracker", "SIH_FSOC_Tracker.exe")
        print("\n" + "="*60)
        print(" BUILD SUCCESSFUL!")
        print(f" Standalone executable available at:\n  {exe_path}")
        print("="*60)
    else:
        print(f"\nBuild failed with return code {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    build()
