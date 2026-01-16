#!/usr/bin/env python3
"""
Test script to check whisper model availability and GPU support
"""

def test_whisper_setup():
    try:
        from faster_whisper import WhisperModel
        print("✅ faster-whisper imported successfully")
        
        # Test model availability
        available_models = ["large-v3-turbo", "distil-large-v3", "large-v3", "medium", "small"]
        
        for model_name in available_models:
            try:
                print(f"\n🔍 Testing model: {model_name}")
                # Just test model initialization without loading
                model = WhisperModel(model_name, device="cpu", download_root="./test_models")
                print(f"✅ Model {model_name} is available")
                del model  # Clean up
                break  # Stop after first successful model
            except Exception as e:
                print(f"❌ Model {model_name} failed: {str(e)}")
                continue
        
        # Test GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                print(f"\n🎯 GPU available: {torch.cuda.get_device_name()}")
                print(f"   CUDA version: {torch.version.cuda}")
                print("✅ GPU acceleration possible")
                
                # Test GPU model loading
                try:
                    gpu_model = WhisperModel("small", device="cuda", compute_type="float16")
                    print("✅ GPU model loading works")
                    del gpu_model
                except Exception as e:
                    print(f"❌ GPU model loading failed: {str(e)}")
            else:
                print("\n❌ No GPU available - will use CPU")
        except ImportError:
            print("\n❌ PyTorch not available - cannot check GPU")
            
    except ImportError as e:
        print(f"❌ Failed to import faster-whisper: {str(e)}")
        print("Try: pip install faster-whisper")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    print("🎵 Testing Whisper Setup")
    print("=" * 40)
    test_whisper_setup()