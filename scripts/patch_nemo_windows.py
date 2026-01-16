#!/usr/bin/env python3
"""
Patch NeMo for Windows compatibility.

NeMo uses signal.SIGKILL which is not available on Windows.
This script patches exp_manager.py to use SIGTERM instead.

Run this after installing nemo_toolkit on Windows:
    python scripts/patch_nemo_windows.py
"""

import platform
import re
import sys
from pathlib import Path


def find_nemo_exp_manager():
    """Find the NeMo exp_manager.py file."""
    try:
        import nemo.utils.exp_manager as em
        return Path(em.__file__)
    except AttributeError:
        # Import failed due to SIGKILL issue - find it manually
        import nemo
        nemo_path = Path(nemo.__file__).parent
        return nemo_path / "utils" / "exp_manager.py"
    except ImportError:
        print("NeMo is not installed. Install with: pip install nemo_toolkit[asr]")
        sys.exit(1)


def patch_exp_manager(filepath: Path) -> bool:
    """Patch exp_manager.py to use SIGTERM instead of SIGKILL on Windows."""
    content = filepath.read_text(encoding="utf-8")
    
    # Check if already patched
    if "SIGTERM" in content and "SIGKILL" not in content:
        print(f"✅ {filepath.name} is already patched for Windows")
        return False
    
    # Pattern to find the SIGKILL usage
    pattern = r"rank_termination_signal:\s*signal\.Signals\s*=\s*signal\.SIGKILL"
    replacement = "rank_termination_signal: signal.Signals = signal.SIGTERM if hasattr(signal, 'SIGTERM') else signal.SIGINT"
    
    new_content, count = re.subn(pattern, replacement, content)
    
    if count == 0:
        print(f"⚠️  Could not find SIGKILL pattern in {filepath.name}")
        return False
    
    filepath.write_text(new_content, encoding="utf-8")
    print(f"✅ Patched {filepath.name} ({count} replacement(s))")
    return True


def main():
    if platform.system() != "Windows":
        print("This patch is only needed on Windows. Skipping.")
        return
    
    print("🔧 Patching NeMo for Windows compatibility...")
    
    exp_manager_path = find_nemo_exp_manager()
    
    if not exp_manager_path.exists():
        print(f"❌ Could not find exp_manager.py at {exp_manager_path}")
        sys.exit(1)
    
    patched = patch_exp_manager(exp_manager_path)
    
    if patched:
        print("\n✅ NeMo patched successfully!")
        print("   You can now import NeMo on Windows.")
    
    # Verify the fix worked
    print("\n🧪 Testing import...")
    try:
        from nemo.collections.asr.models import NeuralDiarizer
        print("✅ NeuralDiarizer import successful!")
    except ImportError as e:
        print(f"⚠️  Import still failing: {e}")
        print("   You may need to check other compatibility issues.")


if __name__ == "__main__":
    main()
