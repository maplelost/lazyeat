# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['src-py\\main.py'],
    pathex=[],
    binaries=[],
        datas=[
        ('C:\\Users\\che\\miniconda3\\envs\\lazyeat\\Lib\\site-packages\\vosk\\*', 'vosk'),
        ('C:\\Users\\che\\miniconda3\\envs\\lazyeat\\Lib\\site-packages\\mediapipe\\*', 'mediapipe'),
        ('C:\\Users\\che\\miniconda3\\envs\\lazyeat\\Lib\\site-packages\\mediapipe\\modules\\*', 'mediapipe/modules'),
        ('C:\\Users\\che\\miniconda3\\envs\\lazyeat\\Lib\\site-packages\\mediapipe\\modules\\palm_detection\\*', 'mediapipe/modules/palm_detection'),
        ('C:\\Users\\che\\miniconda3\\envs\\lazyeat\\Lib\\site-packages\\mediapipe\\modules\\hand_landmark\\*', 'mediapipe/modules/hand_landmark'),
        ('C:\\Users\\che\\miniconda3\\envs\\lazyeat\\Lib\\site-packages\\mediapipe\\python\\*', 'mediapipe/python'),
        ('C:\\Users\\che\\miniconda3\\envs\\lazyeat\\Lib\\site-packages\\uvicorn\\*', 'uvicorn'),
        ('C:\\Users\\che\\miniconda3\\envs\\lazyeat\\Lib\\site-packages\\win10toast\\*', 'win10toast'),
    ],
    hiddenimports=[
        'vosk',
        'mediapipe',
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.protocols',
        'win10toast',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Lazyeat Backend-x86_64-pc-windows-msvc',
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
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='backend-py',
)
