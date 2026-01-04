"""
This script automatically extracts speech transcripts and word-level timing information
from WAV audio files using OpenAI’s Whisper and WhisperX models. It can process both
mono and stereo recordings, compare the results to existing transcripts, and prepare outputs
for downstream analysis or forced alignment (for example, with the Montreal Forced Aligner).

The explanation below is written for a mixed technical and non-technical audience.

-------------------------------------------------------------------------------
STEP-BY-STEP OVERVIEW
-------------------------------------------------------------------------------

1. Environment Setup
   - Limits thread usage and sets environment variables to prevent OpenMP errors on Apple Silicon (M1/M2).
   - Loads libraries like soundfile, whisper, and whisperx (if available).

2. File Classification
   - Mono files (named male_*.wav, female_*.wav) → processed with Whisper.
   - Stereo files (named List_*.wav) → processed with WhisperX (which includes precise forced alignment).

3. Audio Handling
   - Stereo files are split into separate left and right mono channel WAVs.
   - Each channel is transcribed individually, then merged back with speaker labels.
   - Mono “List_” files (if found) are handled gracefully as mono recordings.

4. Transcription Extraction
   - Whisper transcribes audio directly to text and provides word-level start and end times.
   - WhisperX refines this with forced alignment, improving timing accuracy.
   - Each word’s text, start, and end time are stored, along with an overall transcript.

5. Merging Stereo Channels
   - Combines left and right transcripts chronologically.
   - Adds labels (for example, LEFT_SPEAKER, RIGHT_SPEAKER) and identifies overlapping speech.
   - Produces a readable combined dialogue transcript.

6. Creating Predictors
   - Builds binary word onset arrays: 1 where a word begins, 0 elsewhere.
   - Optionally creates Eelbrain NDVar objects linking these onsets to time (for neuroimaging analysis).

7. Saving Results
   For each file, saves:
     - <filename>_whisper.txt → the full transcript text.
     - <filename>_timings.json → word-level timing data.
     - <filename>_word_onsets.pickle → optional Eelbrain-ready data.
   Additionally, a summary JSON (extraction_summary.json) is created for the entire run.

8. Comparison with Existing Transcripts
   - If a reference text file exists (same name, .txt), compares Whisper’s transcript with it.
   - Computes a similarity score (based on text overlap and word counts) and reports discrepancies.

9. Optional Output Preparation
   - Can generate clean “MFA-ready” transcript files for external forced alignment tools.

10. Error Handling and Resource Management
    - Catches missing dependencies, model loading errors, or unexpected audio formats.
    - Frees memory (gc.collect()) after processing each file to handle large datasets efficiently.

-------------------------------------------------------------------------------
EXAMPLE INPUTS AND OUTPUTS
-------------------------------------------------------------------------------

Input 1 — Mono WAV file:
    binaural-cocktail-bids/stimulus/male_speaker01.wav
    (Single-speaker audio recording)

Input 2 — Stereo WAV file:
    binaural-cocktail-bids/stimulus/List_A01.wav
    (Two-speaker stereo recording; left = speaker A, right = speaker B)

Input 3 — Reference transcript (optional):
    binaural-cocktail-bids/stimulus/male_speaker01.txt
Contents:
    "Hello everyone, welcome to the session."

-------------------------------------------------------------------------------

Output 1 — Transcript text file (.txt):
    binaural-cocktail-bids/derivatives/predictors/word_onsets/male_speaker01_whisper.txt

Example contents:
    "Hello everyone welcome to the session"

For stereo files:
    [LEFT_SPEAKER] hello how are you
    [RIGHT_SPEAKER] I’m fine thanks

-------------------------------------------------------------------------------

Output 2 — Word timing JSON file:
    binaural-cocktail-bids/derivatives/predictors/word_onsets/male_speaker01_timings.json

Example contents:
[
  {"word": "hello", "start": 0.42, "end": 0.76},
  {"word": "everyone", "start": 0.78, "end": 1.25},
  {"word": "welcome", "start": 1.27, "end": 1.68},
  {"word": "to", "start": 1.70, "end": 1.82},
  {"word": "the", "start": 1.84, "end": 1.92},
  {"word": "session", "start": 1.94, "end": 2.30}
]

-------------------------------------------------------------------------------

Output 3 — Word onset predictor (array or NDVar):
    A time-aligned binary array sampled at 100 Hz:
        [0, 0, 1, 0, 0, 1, 0, 0, ...]
    Each “1” corresponds to a word onset time.

-------------------------------------------------------------------------------

Output 4 — Summary report (JSON):
    extraction_summary.json

Example contents:
{
  "total_files": 5,
  "mono_files": 3,
  "stereo_files": 2,
  "successful": 5,
  "failed": 0,
  "comparisons": [
    {"file": "male_speaker01.wav", "type": "mono", "similarity": 0.95},
    {"file": "List_A01.wav", "type": "stereo", "left_words": 103, "right_words": 97, "overlapping": 5}
  ]
}

-------------------------------------------------------------------------------
In short:
This script takes speech recordings, extracts word-level transcriptions and timings
(using Whisper or WhisperX), merges stereo channels with speaker labels, compares them
to existing texts, and exports structured data ready for speech, linguistic, or
neuroimaging analysis.
"""


# ============================================================================
# CRITICAL: These environment variables MUST be set BEFORE any other imports
# to prevent OpenMP threading issues on macOS M1/M2
# ============================================================================
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import soundfile as sf
from pathlib import Path
import json
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from analysis.params import RESAMPLING_FREQUENCY
import gc

try:
    from eelbrain import NDVar, UTS
    EELBRAIN_AVAILABLE = True
except ImportError:
    EELBRAIN_AVAILABLE = False


def is_stereo_list_file(wav_path):
    """Check if file is a stereo List_ file"""
    return wav_path.name.startswith('List_')


def split_stereo_channels(wav_path, output_dir):
    """
    Split stereo wav into left and right channel mono files
    
    Parameters:
    -----------
    wav_path : Path
        Path to stereo wav file
    output_dir : Path
        Directory to save mono channel files
        
    Returns:
    --------
    tuple : (left_channel_path, right_channel_path)
    """
    print(f"  Splitting stereo channels...")
    
    # Read stereo file
    audio_data, sample_rate = sf.read(str(wav_path))
    
    if len(audio_data.shape) != 2 or audio_data.shape[1] != 2:
        raise ValueError(f"Expected stereo file, got shape: {audio_data.shape}")
    
    # Split channels
    left_channel = audio_data[:, 0]
    right_channel = audio_data[:, 1]
    
    # Save mono files
    left_path = output_dir / f"{wav_path.stem}_left.wav"
    right_path = output_dir / f"{wav_path.stem}_right.wav"
    
    sf.write(str(left_path), left_channel, sample_rate)
    sf.write(str(right_path), right_channel, sample_rate)
    
    print(f"    Left channel: {left_path.name}")
    print(f"    Right channel: {right_path.name}")
    
    return left_path, right_path


def extract_transcript_whisper(wav_path, model):
    """
    Extract transcript from a single wav file using Whisper
    
    Parameters:
    -----------
    wav_path : Path
        Path to wav file
    model : whisper.model
        Loaded Whisper model
    
    Returns:
    --------
    dict with transcript and word timings
    """
    print(f"  Transcribing: {wav_path.name}")
    
    try:
        # Use fp16=False to avoid FP16 warnings on CPU
        result = model.transcribe(
            str(wav_path),
            word_timestamps=True,
            language="en",
            verbose=False,
            fp16=False  # Disable FP16 on CPU
        )
        
        full_text = result['text'].strip()
        
        word_timings = []
        for segment in result['segments']:
            for word_info in segment.get('words', []):
                word_timings.append({
                    'word': word_info['word'].strip(),
                    'start': word_info['start'],
                    'end': word_info['end']
                })
        
        duration = result['segments'][-1]['end'] if result['segments'] else 0
        print(f"    ✓ {len(word_timings)} words, {duration:.2f}s")
        
        return {
            'success': True,
            'text': full_text,
            'word_timings': word_timings,
            'duration': duration
        }
    
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


def extract_transcript_whisperx(wav_path, device="cpu", compute_type="float32"):
    """
    Extract transcript using WhisperX with forced alignment
    (Works with mono audio; for stereo, split channels first.)
    """
    try:
        import whisperx
    except ImportError:
        print("    WhisperX not installed. Install with: pip install whisperx")
        return {'success': False, 'error': 'WhisperX not installed'}
    
    print(f"    Transcribing with WhisperX: {wav_path.name}")
    
    try:
        # Load audio
        audio = whisperx.load_audio(str(wav_path))
        
        # Transcribe with Whisper
        model = whisperx.load_model("base", device, compute_type=compute_type)
        result = model.transcribe(audio, batch_size=16)
        
        # Align whisper output
        model_a, metadata = whisperx.load_align_model(
            language_code="en", 
            device=device
        )
        result_aligned = whisperx.align(
            result["segments"], 
            model_a, 
            metadata, 
            audio, 
            device
        )
        
        # Extract word timings
        word_timings = []
        for segment in result_aligned["segments"]:
            for word_info in segment.get("words", []):
                word_timings.append({
                    'word': word_info['word'].strip(),
                    'start': word_info['start'],
                    'end': word_info['end'],
                    'score': word_info.get('score', 1.0)
                })
        
        full_text = " ".join([w['word'] for w in word_timings])
        duration = word_timings[-1]['end'] if word_timings else 0
        
        print(f"      ✓ {len(word_timings)} words, {duration:.2f}s")
        
        # Clean up
        del model, model_a
        gc.collect()
        if device == "cuda":
            import torch  # <-- ensure torch is imported before use
            torch.cuda.empty_cache()
        
        return {
            'success': True,
            'text': full_text,
            'word_timings': word_timings,
            'duration': duration
        }
        
    except Exception as e:
        print(f"      ✗ Error: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def merge_stereo_transcripts(left_result, right_result, speaker_labels=None):
    """
    Merge transcripts from left and right channels
    
    Parameters:
    -----------
    left_result : dict
        Result from left channel transcription
    right_result : dict
        Result from right channel transcription
    speaker_labels : tuple
        (left_speaker_name, right_speaker_name), e.g., ("male", "female")
        
    Returns:
    --------
    dict with merged transcript and word timings
    """
    if not left_result['success'] or not right_result['success']:
        return {'success': False, 'error': 'One or both channels failed'}
    
    if speaker_labels is None:
        speaker_labels = ("left", "right")
    
    # Add speaker labels to word timings
    left_timings = left_result['word_timings'].copy()
    for word in left_timings:
        word['speaker'] = speaker_labels[0]
        word['channel'] = 'left'
    
    right_timings = right_result['word_timings'].copy()
    for word in right_timings:
        word['speaker'] = speaker_labels[1]
        word['channel'] = 'right'
    
    # Merge and sort by start time
    all_timings = left_timings + right_timings
    all_timings.sort(key=lambda x: x['start'])
    
    # Identify overlapping segments
    for i, word in enumerate(all_timings):
        word['overlapping'] = False
        if i > 0:
            prev_word = all_timings[i-1]
            if (prev_word['channel'] != word['channel'] and 
                prev_word['end'] > word['start']):
                word['overlapping'] = True
                prev_word['overlapping'] = True
        
        if i < len(all_timings) - 1:
            next_word = all_timings[i+1]
            if (next_word['channel'] != word['channel'] and 
                word['end'] > next_word['start']):
                word['overlapping'] = True
    
    # Create combined transcript with speaker labels
    transcript_parts = []
    current_speaker = None
    current_text = []
    
    for word in all_timings:
        if word['speaker'] != current_speaker:
            if current_text:
                transcript_parts.append(f"[{current_speaker.upper()}] {' '.join(current_text)}")
            current_speaker = word['speaker']
            current_text = [word['word']]
        else:
            current_text.append(word['word'])
    
    if current_text:
        transcript_parts.append(f"[{current_speaker.upper()}] {' '.join(current_text)}")
    
    full_text = "\n".join(transcript_parts)
    
    duration = max(
        left_result.get('duration', 0),
        right_result.get('duration', 0)
    )
    
    return {
        'success': True,
        'text': full_text,
        'word_timings': all_timings,
        'duration': duration,
        'left_word_count': len(left_timings),
        'right_word_count': len(right_timings),
        'overlap_count': sum(1 for w in all_timings if w.get('overlapping', False))
    }


def process_list_mono_file(wav_file, output_path, model_size="base"):
    """
    Process a 'List_' file that turns out to be mono:
    Prefer WhisperX for alignment; fall back to Whisper if WhisperX is unavailable.
    """
    print(f"  Detected mono 'List_' file – processing as mono")
    # Try WhisperX first
    xr = extract_transcript_whisperx(wav_file)
    if not xr['success']:
        print("    WhisperX unavailable or failed – falling back to Whisper")
        try:
            import whisper
            model = whisper.load_model(model_size)
            xr = extract_transcript_whisper(wav_file, model)
        except Exception as e:
            print(f"    ✗ Whisper fallback failed: {e}")
            return {'success': False, 'error': f"Mono List_ processing failed: {e}"}

    if not xr.get('success'):
        return xr

    # Save standard outputs
    save_transcript_outputs(xr, wav_file, output_path)
    xr.update({'mode': 'list_mono'})
    return xr


def process_stereo_file(wav_file, output_path, temp_dir, model_size="base"):
    """
    Process a single 'List_' file.
    If it's really stereo -> split + WhisperX per channel + merge.
    If it's actually mono -> process as mono and still emit the same outputs.
    """
    print(f"  Processing stereo file: {wav_file.name}")

    # Quick probe: if the 'List_' file is actually mono, handle as mono immediately
    try:
        info = sf.info(str(wav_file))
        if getattr(info, 'channels', 2) == 1:
            return process_list_mono_file(wav_file, output_path, model_size=model_size)
    except Exception:
        # If probing fails, continue with the old path; split_stereo_channels will catch issues.
        pass

    # --- Original stereo path ---
    # Split stereo into channels
    try:
        left_path, right_path = split_stereo_channels(wav_file, temp_dir)
    except Exception as e:
        # Fallback: if split fails because it's actually mono, process as mono
        msg = str(e)
        if "Expected stereo file" in msg or "shape:" in msg:
            print(f"    Detected mono data during split – falling back to mono processing")
            return process_list_mono_file(wav_file, output_path, model_size=model_size)
        print(f"    ✗ Error splitting channels: {e}")
        return {'success': False, 'error': str(e)}
    
    # Process left channel with WhisperX
    print(f"    Processing LEFT channel...")
    left_result = extract_transcript_whisperx(left_path)
    
    # Process right channel with WhisperX
    print(f"    Processing RIGHT channel...")
    right_result = extract_transcript_whisperx(right_path)
    
    # Clean up temp files
    try:
        left_path.unlink()
        right_path.unlink()
    except Exception:
        pass
    
    if not left_result['success'] or not right_result['success']:
        return {
            'success': False,
            'error': 'Channel processing failed',
            'left_result': left_result,
            'right_result': right_result
        }
    
    # Determine speaker labels from filename (keep generic)
    speaker_labels = ("left_speaker", "right_speaker")
    
    # Merge transcripts
    merged_result = merge_stereo_transcripts(left_result, right_result, speaker_labels)
    
    if not merged_result['success']:
        return merged_result
    
    # Save merged outputs
    save_transcript_outputs(merged_result, wav_file, output_path)
    print(f"    Left: {merged_result['left_word_count']} words | Right: {merged_result['right_word_count']} words | Overlap: {merged_result['overlap_count']}")
    
    return merged_result


def create_word_onsets_array(word_timings, total_duration, sample_rate=100):
    """Create word onset predictor array from word timings (robust for very short files)."""
    import math
    n_samples = max(1, int(math.ceil(total_duration * sample_rate)))
    time_axis = np.linspace(0, total_duration, n_samples, endpoint=False)
    word_onsets = np.zeros(n_samples)

    for word_info in word_timings:
        word_time = word_info['start']
        idx = np.argmin(np.abs(time_axis - word_time))
        word_onsets[idx] = 1

    return word_onsets, time_axis


def create_eelbrain_ndvar(word_onsets, sample_rate, name):
    """Create eelbrain NDVar from word onset array"""
    try:
        from eelbrain import NDVar, UTS
        time_dim = UTS(0, 1.0/sample_rate, len(word_onsets))
        ndvar = NDVar(word_onsets, time_dim, name=name)
        return ndvar
    except ImportError:
        return None

def save_transcript_outputs(result, wav_file, output_path):
    """Save <stem>_whisper.txt, <stem>_timings.json, and <stem>_word_onsets.pickle (if eelbrain available)."""
    # Save transcript
    whisper_txt_path = output_path / f"{wav_file.stem}_whisper.txt"
    with open(whisper_txt_path, 'w', encoding='utf-8') as f:
        f.write(result['text'])

    # Save timings
    timings_json_path = output_path / f"{wav_file.stem}_timings.json"
    with open(timings_json_path, 'w', encoding='utf-8') as f:
        json.dump(result['word_timings'], f, indent=2)

    # Create word onsets (+ optional pickle)
    word_onsets, _ = create_word_onsets_array(
        result['word_timings'],
        result['duration'],
        sample_rate=RESAMPLING_FREQUENCY
    )
    ndvar = create_eelbrain_ndvar(word_onsets, RESAMPLING_FREQUENCY, wav_file.name)

    if ndvar is not None:
        words = [w['word'] for w in result['word_timings']]
        df_single = pd.DataFrame({
            'filename': [wav_file.name],
            'word_onsets': [ndvar],
            'words': [words]
        })
        pickle_path = output_path / f"{wav_file.stem}_word_onsets.pickle"
        df_single.to_pickle(pickle_path)
        print(f"    ✓ Saved transcript, timings, and pickle")
    else:
        print(f"    ✓ Saved transcript and timings")

def compare_transcripts(original_text, whisper_text):
    """Compare original transcript with Whisper output"""
    orig_normalized = ' '.join(original_text.lower().split())
    whisper_normalized = ' '.join(whisper_text.lower().split())
    
    similarity = SequenceMatcher(None, orig_normalized, whisper_normalized).ratio()
    
    orig_words = orig_normalized.split()
    whisper_words = whisper_normalized.split()
    
    return {
        'similarity': similarity,
        'original_word_count': len(orig_words),
        'whisper_word_count': len(whisper_words),
        'word_count_diff': abs(len(orig_words) - len(whisper_words))
    }

def process_all_files(input_dir, base_output_dir="binaural-cocktail-bids/derivatives/predictors", model_size="base"):
    """
    Process all wav files in directory
    Handles both mono files (male_/female_) with Whisper and stereo files (List_) with WhisperX
    """
    input_path = Path(input_dir)
    base_path = Path(base_output_dir)
    
    output_path = base_path / "word_onsets"
    output_path.mkdir(exist_ok=True)
    
    # Create temp directory for channel splits
    temp_dir = output_path / "temp_channels"
    temp_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("TRANSCRIPT EXTRACTION (Whisper + WhisperX)")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_path}")
    print(f"Whisper model size: {model_size}")
    
    # Separate files by type
    mono_files = [f for f in input_path.glob("*.wav") 
                  if f.name.startswith(('male_', 'female_'))]
    stereo_files = [f for f in input_path.glob("*.wav") 
                    if f.name.startswith('List_')]
    
    print(f"\nFound {len(mono_files)} mono files (Whisper)")
    print(f"Found {len(stereo_files)} stereo files (WhisperX)")
    
    results = {}
    comparison_report = []
    
    # Process mono files with Whisper
    if mono_files:
        print("\n" + "=" * 80)
        print("PROCESSING MONO FILES WITH WHISPER")
        print("=" * 80)
        
        print(f"\nLoading Whisper model '{model_size}'...")
        try:
            import whisper
            model = whisper.load_model(model_size)
            print("Model loaded successfully\n")
        except Exception as e:
            print(f"✗ Error loading Whisper model: {e}")
            print("\nPlease run: pip uninstall whisper && pip install openai-whisper")
            return {}, []
        
        for i, wav_file in enumerate(sorted(mono_files), 1):
            print(f"[{i}/{len(mono_files)}] {wav_file.name}")
            
            try:
                whisper_result = extract_transcript_whisper(wav_file, model)
                
                if not whisper_result['success']:
                    results[wav_file.stem] = whisper_result
                    continue
                
                # Save transcript
                whisper_txt_path = output_path / f"{wav_file.stem}_whisper.txt"
                with open(whisper_txt_path, 'w', encoding='utf-8') as f:
                    f.write(whisper_result['text'])
                
                # Save timings
                timings_json_path = output_path / f"{wav_file.stem}_timings.json"
                with open(timings_json_path, 'w', encoding='utf-8') as f:
                    json.dump(whisper_result['word_timings'], f, indent=2)
                
                # Create word onsets
                word_onsets, time_axis = create_word_onsets_array(
                    whisper_result['word_timings'],
                    whisper_result['duration'],
                    sample_rate=RESAMPLING_FREQUENCY
                )
                
                ndvar = create_eelbrain_ndvar(word_onsets, RESAMPLING_FREQUENCY, wav_file.name)
                
                if ndvar is not None:
                    words = [w['word'] for w in whisper_result['word_timings']]
                    df_single = pd.DataFrame({
                        'filename': [wav_file.name],
                        'word_onsets': [ndvar],
                        'words': [words]
                    })
                    pickle_path = output_path / f"{wav_file.stem}_word_onsets.pickle"
                    df_single.to_pickle(pickle_path)
                
                # Compare with original if exists
                original_txt_path = wav_file.with_suffix('.txt')
                if original_txt_path.exists():
                    with open(original_txt_path, 'r', encoding='utf-8') as f:
                        original_text = f.read().strip()
                    
                    comparison = compare_transcripts(original_text, whisper_result['text'])
                    print(f"    Similarity: {comparison['similarity']:.1%}")
                    
                    comparison_report.append({
                        'file': wav_file.name,
                        'type': 'mono',
                        'similarity': comparison['similarity'],
                        'original_words': comparison['original_word_count'],
                        'whisper_words': comparison['whisper_word_count']
                    })
                
                results[wav_file.stem] = whisper_result
                
                # Clean up memory after each file
                gc.collect()
                
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                results[wav_file.stem] = {'success': False, 'error': str(e)}
                continue
    
    # Process stereo files with WhisperX
    if stereo_files:
        print("\n" + "=" * 80)
        print("PROCESSING STEREO FILES WITH WHISPERX")
        print("=" * 80 + "\n")
        
        for i, wav_file in enumerate(sorted(stereo_files), 1):
            print(f"[{i}/{len(stereo_files)}] {wav_file.name}")
            
            try:
                stereo_result = process_stereo_file(wav_file, output_path, temp_dir, model_size=model_size)
                
                if stereo_result['success']:
                    if stereo_result.get('mode') == 'list_mono':
                        comparison_report.append({
                            'file': wav_file.name,
                            'type': 'list_mono',
                            'words': len(stereo_result.get('word_timings', []))
                        })
                    else:
                        comparison_report.append({
                            'file': wav_file.name,
                            'type': 'stereo',
                            'left_words': stereo_result['left_word_count'],
                            'right_words': stereo_result['right_word_count'],
                            'overlapping': stereo_result['overlap_count']
                        })

                
                results[wav_file.stem] = stereo_result
                
                # Clean up memory
                gc.collect()
                
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                results[wav_file.stem] = {'success': False, 'error': str(e)}
                continue
    
    # Clean up temp directory
    try:
        temp_dir.rmdir()
    except:
        pass
    
    # Save summary
    summary_path = output_path / "extraction_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_files': len(mono_files) + len(stereo_files),
            'mono_files': len(mono_files),
            'stereo_files': len(stereo_files),
            'successful': sum(1 for r in results.values() if r.get('success', False)),
            'failed': sum(1 for r in results.values() if not r.get('success', False)),
            'comparisons': comparison_report
        }, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files: {len(mono_files) + len(stereo_files)}")
    print(f"  Mono files (Whisper): {len(mono_files)}")
    print(f"  Stereo files (WhisperX): {len(stereo_files)}")
    print(f"Successful: {sum(1 for r in results.values() if r.get('success', False))}")
    print(f"Failed: {sum(1 for r in results.values() if not r.get('success', False))}")
    
    if comparison_report:
        mono_comparisons = [c for c in comparison_report if c.get('type') == 'mono']
        if mono_comparisons:
            low_similarity = [c for c in mono_comparisons if c['similarity'] < 0.8]
            if low_similarity:
                print(f"\n{len(low_similarity)} mono file(s) with low similarity (< 80%):")
                for comp in low_similarity:
                    print(f"  {comp['file']}: {comp['similarity']:.1%}")
    
    print(f"\nAll results saved to: {output_path}")
    
    return results, comparison_report


def create_mfa_ready_transcripts(word_onsets_dir):
    """Create clean transcripts from Whisper output for MFA alignment"""
    word_onsets_path = Path(word_onsets_dir)
    output_path = word_onsets_path / "mfa_ready_transcripts"
    output_path.mkdir(exist_ok=True)
    
    print("\nCreating MFA-ready transcripts")
    print(f"Output directory: {output_path}")
    
    whisper_files = list(word_onsets_path.glob("*_whisper.txt"))
    
    for whisper_file in whisper_files:
        original_name = whisper_file.stem.replace('_whisper', '') + '.txt'
        output_file = output_path / original_name
        
        with open(whisper_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  Created: {original_name}")
    
    print(f"Created {len(whisper_files)} MFA-ready transcripts")


if __name__ == "__main__":
    import sys
    
    print("Checking dependencies...")
    
    try:
        import soundfile
        print("✓ soundfile installed")
    except ImportError:
        print("✗ soundfile not installed. Install with: pip install soundfile")
        sys.exit(1)
    
    try:
        import whisperx
        print("✓ WhisperX detected - will be used for stereo files")
    except ImportError:
        print("⚠ WhisperX not installed - true stereo 'List_' files will be skipped; mono 'List_' will be processed with Whisper")
    
    print("\n")
    
    # Run processing
    results, comparisons = process_all_files(
        input_dir="binaural-cocktail-bids/stimulus",
        base_output_dir="binaural-cocktail-bids/derivatives/predictors",
        model_size="base"
    )
    
    # Offer to create MFA-ready transcripts
    if results:
        response = input("\nCreate MFA-ready transcripts from Whisper output? (y/n): ")
        
        if response.lower() == 'y':
            create_mfa_ready_transcripts(
                word_onsets_dir="binaural-cocktail-bids/derivatives/predictors/word_onsets"
            )
    
    print("\nProcessing complete.")


