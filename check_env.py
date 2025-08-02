import sys
import os

print("--- Python Executable ---")
print(f"sys.executable: {sys.executable}")
print(f"os.environ['VIRTUAL_ENV']: {os.environ.get('VIRTUAL_ENV', 'Not set')}")

print("\n--- sys.path (Module Search Paths) ---")
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")

print("\n--- Attempting to Import scipy ---")
try:
    import scipy
    print(f"scipy imported successfully from: {scipy.__file__}")
except ImportError as e:
    print(f"Error importing scipy: {e}")

print("\n--- Attempting to Import sentence_transformers ---")
try:
    import sentence_transformers
    print(f"sentence_transformers imported successfully from: {sentence_transformers.__file__}")
except ImportError as e:
    print(f"Error importing sentence_transformers: {e}")