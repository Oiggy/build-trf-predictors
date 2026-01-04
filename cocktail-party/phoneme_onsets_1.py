"""
This script automatically extracts **phoneme onsets** (the precise timing of individual speech sounds)
from spoken audio recordings (WAV files) using **OpenAI’s Whisper** transcriptions and the **phonemizer**
library. It is designed for use in neuroscience, linguistics, and speech research but explained here
for an interdisciplinary audience.

STEP-BY-STEP OVERVIEW:

1. **Setup and Configuration**
   - Ensures required libraries are installed and environment paths are set.
   - Loads Whisper (for speech-to-text) and Phonemizer (for text-to-phoneme conversion).

2. **Phonemization (Text → Phonemes)**
   - Converts the full spoken text of each recording into a sequence of phonemes
     (e.g., “cat” → [“k”, “æ”, “t”]).
   - Uses an optimized single-pass method to avoid redundant processing.

3. **Phoneme Alignment (Timing Each Phoneme)**
   - Uses existing **word timing files** (from Whisper output) to estimate when each
     phoneme starts and ends.
   - If a word lasts 0.3 seconds and has 3 phonemes, each phoneme is spaced evenly
     across that time window.

4. **Onset Predictor Creation**
   - Builds a **binary time-series array**: a “1” marks the start of a phoneme, and
     “0” elsewhere.
   - Optionally creates an **Eelbrain NDVar** object linking onsets to a continuous
     time axis for use in EEG/MEG data analysis.

5. **Batch Processing**
   - Iterates through all timing files in a folder.
   - For each one, produces phoneme-level timing and onset data.
   - Saves the results as:
       - JSON files (readable phoneme timing info)
       - Optional pickle files (for Eelbrain analysis)
       - A summary report with overall success/failure stats.

6. **Error Handling and Cleanup**
   - Handles missing data or missing dependencies gracefully.
   - Frees memory between files to improve performance.

-------------------------------------------------------------------------------
EXAMPLE INPUTS AND OUTPUTS
-------------------------------------------------------------------------------

**Input 1 — WAV file (audio):**
    binaural-cocktail-bids/stimulus/example.wav
    (Speech recording of a short spoken sentence)

**Input 2 — Word timing file (JSON):**
    binaural-cocktail-bids/derivatives/predictors/word_onsets/example_timings.json

Example contents:
[
  {"word": "hello", "start": 0.45, "end": 0.78},
  {"word": "world", "start": 0.80, "end": 1.20}
]

**Input 3 — (Optional) Transcript text file:**
    binaural-cocktail-bids/derivatives/predictors/word_onsets/example_whisper.txt
Contents:
    hello world

-------------------------------------------------------------------------------

**Output 1 — Phoneme timing file (JSON):**
    binaural-cocktail-bids/derivatives/predictors/phoneme_onsets/example_phoneme_timings.json

Example contents:
[
  {"phoneme": "h", "start": 0.45, "end": 0.53, "word": "hello"},
  {"phoneme": "ɛ", "start": 0.53, "end": 0.61, "word": "hello"},
  {"phoneme": "l", "start": 0.61, "end": 0.69, "word": "hello"},
  {"phoneme": "oʊ", "start": 0.69, "end": 0.78, "word": "hello"},
  {"phoneme": "w", "start": 0.80, "end": 0.90, "word": "world"},
  {"phoneme": "ɝ", "start": 0.90, "end": 1.00, "word": "world"},
  {"phoneme": "l", "start": 1.00, "end": 1.10, "word": "world"},
  {"phoneme": "d", "start": 1.10, "end": 1.20, "word": "world"}
]

**Output 2 — Phoneme onset array (NumPy or NDVar):**
    A binary array sampled at the given frequency (e.g., 100 Hz):
        [0, 0, 1, 0, 0, 1, 0, 0, ...]
    Each “1” marks a phoneme start time along the recording.

**Output 3 — Summary file (JSON):**
    phoneme_extraction_summary.json
Contains counts of total processed files, successful extractions, and errors.

-------------------------------------------------------------------------------
In short:
The script takes **audio recordings + word-level timings**, aligns them to
**phoneme-level time points**, and produces ready-to-analyze predictors for
**speech and brain signal studies**.
"""




import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set espeak-ng library path explicitly
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = '/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.1.dylib'

import whisper
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analysis.params import RESAMPLING_FREQUENCY
import gc  # For garbage collection

try:
    from eelbrain import NDVar, UTS
    EELBRAIN_AVAILABLE = True
except ImportError:
    EELBRAIN_AVAILABLE = False


def phonemize_text(text):
    """
    Convert text to phonemes using a phonemizer
    
    Parameters:
    -----------
    text : str
        Text to convert to phonemes
    
    Returns:
    --------
    list of phonemes
    """
    try:
        from phonemizer import phonemize
        from phonemizer.separator import Separator
        
        sep = Separator(phone=' ', word='|', syllable='')
        
        phonemes = phonemize(
            text,
            language='en-us',
            backend='espeak',
            separator=sep,
            strip=True,
            preserve_punctuation=False,
            with_stress=False
        )
        
        phoneme_list = [p for p in phonemes.split() if p != '|']
        
        return phoneme_list
    
    except Exception as e:
        print(f"Phonemization error: {e}")
        return None


def align_phonemes_to_words_optimized(word_timings, full_text):
    """
    OPTIMIZED: Phonemize text once, then distribute across words
    
    Parameters:
    -----------
    word_timings : list
        List of dicts with 'word', 'start', 'end' keys
    full_text : str
        Full transcript text
    
    Returns:
    --------
    list of dicts with 'phoneme', 'start', 'end' keys
    """
    from phonemizer import phonemize
    from phonemizer.separator import Separator
    
    # Phonemize full text once
    sep = Separator(phone=' ', word='|', syllable='')
    
    try:
        phonemes_with_boundaries = phonemize(
            full_text,
            language='en-us',
            backend='espeak',
            separator=sep,
            strip=True,
            preserve_punctuation=False,
            with_stress=False
        )
        
        # Split by word boundaries
        word_phoneme_groups = phonemes_with_boundaries.split('|')
        
        phoneme_timings = []
        
        for i, word_info in enumerate(word_timings):
            if i >= len(word_phoneme_groups):
                break
                
            word_phonemes = word_phoneme_groups[i].strip().split()
            
            if not word_phonemes:
                continue
            
            word_start = word_info['start']
            word_end = word_info['end']
            word_duration = word_end - word_start
            
            n_phonemes = len(word_phonemes)
            phoneme_duration = word_duration / n_phonemes
            
            for j, phoneme in enumerate(word_phonemes):
                phoneme_start = word_start + (j * phoneme_duration)
                phoneme_end = phoneme_start + phoneme_duration
                
                phoneme_timings.append({
                    'phoneme': phoneme,
                    'start': phoneme_start,
                    'end': phoneme_end,
                    'word': word_info['word']
                })
        
        return phoneme_timings
        
    except Exception as e:
        print(f"  Error in phoneme alignment: {e}")
        return []


def create_phoneme_onsets_array(phoneme_timings, total_duration, sample_rate=100):
    """
    Create phoneme onset predictor array from phoneme timings
    
    Parameters:
    -----------
    phoneme_timings : list
        List of dicts with 'phoneme', 'start', 'end' keys
    total_duration : float
        Total duration in seconds
    sample_rate : int
        Sampling rate in Hz
    
    Returns:
    --------
    phoneme_onsets : numpy array
        Binary array with 1s at phoneme onsets
    time_axis : numpy array
        Time axis in seconds
    """
    n_samples = int(total_duration * sample_rate)
    time_axis = np.linspace(0, total_duration, n_samples)
    
    phoneme_onsets = np.zeros(n_samples)
    
    for phoneme_info in phoneme_timings:
        phoneme_time = phoneme_info['start']
        idx = np.argmin(np.abs(time_axis - phoneme_time))
        phoneme_onsets[idx] = 1
    
    return phoneme_onsets, time_axis


def create_eelbrain_ndvar(phoneme_onsets, sample_rate, name):
    """
    Create eelbrain NDVar from phoneme onset array
    
    Parameters:
    -----------
    phoneme_onsets : numpy array
        Binary array with 1s at phoneme onsets
    sample_rate : int
        Sampling rate in Hz
    name : str
        Name for the NDVar
    
    Returns:
    --------
    ndvar : eelbrain NDVar or None
    """
    try:
        from eelbrain import NDVar, UTS
        
        time_dim = UTS(0, 1.0/sample_rate, len(phoneme_onsets))
        ndvar = NDVar(phoneme_onsets, time_dim, name=name)
        
        return ndvar
    except ImportError:
        return None


def process_all_files(input_dir, base_output_dir="binaural-cocktail-bids/derivatives/predictors"):
    """
    Process all wav files and extract phoneme onsets
    
    Parameters:
    -----------
    input_dir : str
        Directory with wav files
    base_output_dir : str
        Base output directory
    """
    input_path = Path(input_dir)
    base_path = Path(base_output_dir)
    
    word_onsets_dir = base_path / "word_onsets"
    output_path = base_path / "phoneme_onsets"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PHONEME ONSET EXTRACTION (OPTIMIZED)")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Word onsets directory: {word_onsets_dir}")
    print(f"Output directory: {output_path}")
    
    timing_files = list(word_onsets_dir.glob("*_timings.json"))
    
    if not timing_files:
        print(f"\nNo timing files found in {word_onsets_dir}")
        print("Please run word onset extraction first!")
        return
    
    print(f"\nFound {len(timing_files)} timing files to process\n")
    
    results = {}
    
    for i, timing_file in enumerate(sorted(timing_files), 1):
        wav_name = timing_file.stem.replace('_timings', '') + '.wav'
        print(f"[{i}/{len(timing_files)}] {wav_name}")
        
        try:
            with open(timing_file, 'r', encoding='utf-8') as f:
                word_timings = json.load(f)
            
            if not word_timings:
                print(f"  No word timings found")
                continue
            
            transcript_file = timing_file.parent / f"{timing_file.stem.replace('_timings', '_whisper')}.txt"
            if transcript_file.exists():
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    full_text = f.read().strip()
            else:
                full_text = ' '.join([w['word'] for w in word_timings])
            
            # Use optimized alignment
            phoneme_timings = align_phonemes_to_words_optimized(word_timings, full_text)
            
            if not phoneme_timings:
                print(f"  Failed to extract phoneme timings")
                results[wav_name] = {'success': False, 'error': 'phoneme extraction failed'}
                continue
            
            print(f"  Extracted {len(phoneme_timings)} phoneme timings")
            
            total_duration = word_timings[-1]['end']
            
            # Save phoneme timings
            phoneme_json_path = output_path / f"{timing_file.stem.replace('_timings', '_phoneme_timings')}.json"
            with open(phoneme_json_path, 'w', encoding='utf-8') as f:
                json.dump(phoneme_timings, f, indent=2)
            
            # Create phoneme onset array
            phoneme_onsets, time_axis = create_phoneme_onsets_array(
                phoneme_timings,
                total_duration,
                sample_rate=RESAMPLING_FREQUENCY
            )
            
            # Create eelbrain NDVar
            ndvar = create_eelbrain_ndvar(phoneme_onsets, RESAMPLING_FREQUENCY, wav_name)
            
            if ndvar is not None:
                phonemes = [p['phoneme'] for p in phoneme_timings]
                
                df_single = pd.DataFrame({
                    'filename': [wav_name],
                    'phoneme_onsets': [ndvar],
                    'phonemes': [phonemes]
                })
                
                pickle_path = output_path / f"{timing_file.stem.replace('_timings', '_phoneme_onsets')}.pickle"
                df_single.to_pickle(pickle_path)
                
                print(f"  Saved phoneme timings and onsets DataFrame")
                
                results[wav_name] = {
                    'success': True,
                    'n_phonemes': len(phonemes),
                    'duration': total_duration
                }
            else:
                print(f"  Saved phoneme timings only (eelbrain not available)")
                
                results[wav_name] = {
                    'success': False,
                    'error': 'eelbrain not available'
                }
            
            # Force garbage collection after each file
            gc.collect()
            
        except Exception as e:
            print(f"  ERROR processing {wav_name}: {e}")
            results[wav_name] = {'success': False, 'error': str(e)}
            gc.collect()
            continue
    
    # Save summary
    summary_path = output_path / "phoneme_extraction_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_files': len(timing_files),
            'successful': sum(1 for r in results.values() if r.get('success', False)),
            'failed': sum(1 for r in results.values() if not r.get('success', False)),
            'results': results
        }, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files: {len(timing_files)}")
    print(f"Successful: {sum(1 for r in results.values() if r.get('success', False))}")
    print(f"Failed: {sum(1 for r in results.values() if not r.get('success', False))}")
    print(f"\nAll results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    import sys
    
    try:
        import phonemizer
    except ImportError:
        print("phonemizer not installed!")
        print("\nInstall with: pip install phonemizer")
        sys.exit(1)
    
    results = process_all_files(
        input_dir="binaural-cocktail-bids/stimulus",
        base_output_dir="binaural-cocktail-bids/derivatives/predictors"
    )
    
    print("\nProcessing complete.")