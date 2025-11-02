"""Simple environment checker for the gesture player.

Run this after installing requirements to ensure the critical packages import.
"""
import importlib
mods = ['cv2', 'mediapipe', 'pygame', 'pycaw']

print('Checking modules:')
for m in mods:
    try:
        importlib.import_module(m)
        print(f'  {m}: OK')
    except Exception as e:
        print(f'  {m}: ERROR -> {e}')

print('\nPython version:', __import__('sys').version)
print('Location of project:', __import__('os').path.abspath(__file__))
