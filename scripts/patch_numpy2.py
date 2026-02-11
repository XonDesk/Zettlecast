"""
NumPy 2.0 Compatibility Patch for NeMo

This script patches NumPy to restore 'np.sctypes' which was removed in NumPy 2.0.
NeMo and some other ML libraries still use this deprecated API.

Run this BEFORE importing nemo:
    import scripts.patch_numpy2  # Applies the patch
    import nemo.collections.asr as nemo_asr
"""

import numpy as np

def apply_numpy2_patch():
    """
    Restore np.sctypes for NumPy 2.0+ compatibility.
    
    np.sctypes was a dict mapping type categories to lists of dtypes.
    This patch recreates it for backward compatibility with old code.
    """
    if hasattr(np, 'sctypes'):
        # Already exists (NumPy < 2.0), nothing to do
        return False
    
    # Recreate sctypes dict for NumPy 2.0+
    # Based on the original NumPy 1.x implementation
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str, np.void],
    }
    
    # Also add longdouble if available
    if hasattr(np, 'longdouble'):
        np.sctypes['float'].append(np.longdouble)
    if hasattr(np, 'clongdouble'):
        np.sctypes['complex'].append(np.clongdouble)
    
    return True

# Auto-apply patch on import
_patched = apply_numpy2_patch()

if __name__ == "__main__":
    if _patched:
        print("✅ NumPy 2.0 compatibility patch applied (np.sctypes restored)")
    else:
        print("ℹ️  NumPy < 2.0 detected, no patch needed")
