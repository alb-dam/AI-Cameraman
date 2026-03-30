# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file per AI-Cameraman.

Genera un bundle --onedir con tutti i modelli AI inclusi (~3.5 GB).
Eseguire con:  pyinstaller ai_cameraman.spec
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# ── Path di progetto ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(SPECPATH)

# ── Data files ──────────────────────────────────────────────────────────
# Modelli AI e file di configurazione da includere nel bundle
datas = [
    (os.path.join(PROJECT_ROOT, 'assets', 'yoloe-26m-seg.pt'), 'assets'),
    (os.path.join(PROJECT_ROOT, 'assets', 'logo.png'), 'assets'),
    (os.path.join(PROJECT_ROOT, 'tmp', 'config.json'), 'tmp'),
]

# Raccogli data files dai pacchetti che ne hanno bisogno
datas += collect_data_files('ultralytics')

# ── Hidden imports ──────────────────────────────────────────────────────
# Moduli caricati dinamicamente che PyInstaller non rileva automaticamente
hiddenimports = []

# Ultralytics + YOLOE
hiddenimports += collect_submodules('ultralytics')

# PyTorch
hiddenimports += [
    'torch',
    'torch.utils',
    'torch.utils.data',
    'torchvision',
    'torchvision.models',
    'torchvision.transforms',
]

# PySide6
hiddenimports += collect_submodules('PySide6')

# NDI
hiddenimports += collect_submodules('cyndilib')

# OpenCV
hiddenimports += ['cv2']

# Platform-specific
if sys.platform == 'darwin':
    hiddenimports += ['AVFoundation']
elif sys.platform == 'win32':
    hiddenimports += collect_submodules('pygrabber')
    hiddenimports += ['onnx', 'onnxruntime']

# Moduli applicazione (pacchetti interni)
hiddenimports += [
    'app', 'app.controller', 'app.factory', 'app.logger',
    'app.pipeline', 'app.state',
    'core', 'core.director', 'core.geometry', 'core.interfaces',
    'core.models', 'core.performance', 'core.roi',
    'core.thread_manager', 'core.tracking', 'core.vision',
    'core.yolo_model',
    'config', 'config.settings',
    'gui', 'gui.main_window', 'gui.panels',
    'gui.panels.control_panel', 'gui.panels.preview_panel',
    'gui.panels.log_panel',
    'video', 'video.input', 'video.output',
    'video.ndi_output', 'video.overlay',
]

# ── Runtime hook per risolvere i path dei dati ──────────────────────────
# Crea un runtime hook che imposta la directory di lavoro
# al percorso del bundle così che i path relativi (assets/, config.json) funzionino
runtime_hook_content = '''
import os
import sys

if getattr(sys, 'frozen', False):
    # Eseguibile PyInstaller: i dati sono nella stessa directory dell'exe (onedir)
    os.chdir(os.path.dirname(sys.executable))
'''

runtime_hook_path = os.path.join(PROJECT_ROOT, '_runtime_hook.py')
with open(runtime_hook_path, 'w') as f:
    f.write(runtime_hook_content)

# ── Analysis ────────────────────────────────────────────────────────────
a = Analysis(
    [os.path.join(PROJECT_ROOT, 'main.py')],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[runtime_hook_path],
    excludes=[
        'pytest', 'pytest_cov', 'pytest_mock',
        'tkinter', '_tkinter',
        'matplotlib',
        'IPython', 'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ── PYZ ─────────────────────────────────────────────────────────────────
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ── EXE ─────────────────────────────────────────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # onedir: binaries vanno in COLLECT
    name='AI-Cameraman',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Disabilitato: UPX non è compatibile con file da GB
    console=False,  # Applicazione GUI, niente finestra terminale
    target_arch=None,
)

# ── COLLECT (onedir) ────────────────────────────────────────────────────
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='AI-Cameraman',
)

# ── macOS App Bundle (opzionale) ────────────────────────────────────────
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='AI-Cameraman.app',
        icon=os.path.join(PROJECT_ROOT, 'assets', 'logo.icns'),  # Aggiunto il logo .icns
        bundle_identifier='com.aicameraman.app',
        info_plist={
            'CFBundleName': 'AI-Cameraman',
            'CFBundleDisplayName': 'AI-Cameraman',
            'CFBundleVersion': '1.0.0',
            'CFBundleShortVersionString': '1.0.0',
            'NSHighResolutionCapable': True,
            'NSCameraUsageDescription': 'AI-Cameraman necessita della camera per acquisire il video.',
        },
    )
