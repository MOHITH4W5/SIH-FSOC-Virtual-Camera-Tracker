# -*- mode: python ; coding: utf-8 -*-
# sih_fsoc_tracker.spec - PyInstaller Specification for SIH26169 FSOC Virtual Camera Tracker

import os
import sys

block_cipher = None

# Base directory
base_dir = os.path.abspath(os.path.join(SPECPATH, ".."))

from PyInstaller.utils.hooks import collect_data_files

# Datas to bundle
datas = [
    (os.path.join(base_dir, "models", "beacon_discriminator.pt"), "models"),
]
datas += collect_data_files("customtkinter")

hidden_imports = [
    "torch",
    "cv2",
    "numpy",
    "scipy",
    "scipy.spatial",
    "scipy.spatial.transform",
    "matplotlib",
    "matplotlib.backends.backend_tkagg",
    "PIL",
    "PIL.Image",
    "PIL.ImageTk",
    "tkinter",
    "tkinter.ttk",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "customtkinter",
    "darkdetect",
]

a = Analysis(
    [os.path.join(base_dir, "main.py")],
    pathex=[base_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "IPython", "jupyter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SIH_FSOC_Tracker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="SIH_FSOC_Tracker",
)
