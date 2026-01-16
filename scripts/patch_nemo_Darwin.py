#!/usr/bin/env python3
"""
Patch NeMo for macOS/Darwin compatibility.

Run this after installing nemo_toolkit on macOS:
    python scripts/patch_nemo_Darwin.py
"""

import sys
from pathlib import Path


def find_nemo_exp_manager():
    """Find the NeMo exp_manager.py file."""
    try:
        import nemo.utils.exp_manager as em
        return Path(em.__file__)
    except ImportError:
        print("NeMo is not installed. Install with: pip install nemo_toolkit[asr]")
        sys.exit(1)


def check_nemo_compatibility():
    """Check if NeMo is properly installed and working on macOS."""
    print("🧪 Checking NeMo compatibility on macOS...")
    try:
        from nemo.collections.asr.models import NeuralDiarizer
        print("✅ NeuralDiarizer import successful!")
        return True
    except ImportError as e:
        print(f"⚠️  Import issue detected: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Unexpected error: {e}")
        return False


def main():
    """Main patch function for macOS."""
    print("🔧 Checking NeMo for macOS compatibility...")
    
    # macOS/Darwin typically doesn't have the same signal issues as Windows
    # but we verify that NeMo imports work correctly
    try:
        exp_manager_path = find_nemo_exp_manager()
        print(f"✅ Found NeMo at: {exp_manager_path}")
    except SystemExit:
        return
    
    # Verify NeMo works
    if check_nemo_compatibility():
        print("\n✅ NeMo is compatible with macOS!")
    else:
        print("\n⚠️  NeMo may have compatibility issues on macOS.")
        print("   Consider updating: pip install --upgrade nemo_toolkit")


if __name__ == "__main__":
    main()
