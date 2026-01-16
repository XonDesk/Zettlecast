# NeMo Transcription Troubleshooting

## Common Issues and Fixes

### Issue 1: Deadlock Detected by _ModuleLock

**Error:**
```
deadlock detected by _ModuleLock('nemo.collections.asr.models') at 1423672333648
```

**Cause:**
NeMo models have thread-safety issues when loading multiple models in parallel. The parallel transcription + diarization approach can trigger deadlocks.

**Fix Applied:**
- Sequential processing: Transcription runs first, then diarization
- Explicit device mapping when loading Parakeet model
- Fallback to faster-whisper if NeMo fails to load

**Result:**
Processing is now sequential but still fast enough since each chunk is ~10 minutes.

### Issue 2: Missing Key 'device' in Diarization Config

**Error:**
```
Diarization failed for chunk 0: Missing key device
    full_key: device
    object_type=dict
```

**Cause:**
NeMo's diarization config requires an explicit `device` parameter but it wasn't being set.

**Fix Applied:**
Added `device: self.device` to the diarization config.

**Result:**
Diarization config now includes device specification.

### Issue 3: NeMo Model Loading Timeout

**If the NeMo model fails to load:**
The system now falls back to faster-whisper for transcription. You'll see a warning:
```
WARNING: Falling back to faster-whisper for chunk transcription
```

This allows processing to continue even if NeMo has issues.

## When to Use Which Pipeline

### Use Whisper (Default)
```bash
zettlecast podcast run
```
- Lightweight and reliable
- Works on CPU
- Good enough for most use cases
- No speaker diarization

### Use NeMo (When Needed)
```bash
zettlecast podcast run --use-nemo
```
Or set in `.env`:
```env
USE_NEMO=true
```

Only use NeMo if:
1. You need speaker diarization (multiple speakers)
2. You have a GPU and want faster processing
3. You have adequate disk space (~5GB)

## Debugging Tips

### Enable Debug Logging
If you need to troubleshoot further, check the logs:
```bash
# Look for detailed error messages
cat ~/._BRAIN_STORAGE/podcast_queue.json  # Check queue status
```

### Test NeMo Installation
```bash
python -c "from nemo.collections.asr.models import EncDecRNNTBPEModel; print('NeMo OK')"
```

### Test Whisper
```bash
python -c "from faster_whisper import WhisperModel; print('Whisper OK')"
```

### Monitor Processing
```bash
zettlecast podcast status  # Check queue progress
```

## Known Limitations

1. **Sequential Processing**: NeMo processes chunks sequentially (not in parallel) to avoid deadlocks
2. **Windows Compatibility**: NeMo on Windows may have additional issues; Whisper is more stable
3. **Model Loading**: First use of NeMo is slow (models download and cache)
4. **GPU Memory**: Large audio files on GPU may need to adjust `NEMO_CHUNK_DURATION_MINUTES`

## Recommendations

- **Production Use**: Stick with Whisper (default) for reliability
- **Testing NeMo**: Use a small audio file first (< 5 minutes)
- **Large Files**: If using NeMo with files > 2 hours, reduce chunk size or use Whisper
- **Windows Users**: Default to Whisper unless you specifically need speaker diarization
