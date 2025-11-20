# Acoustic Feature Summary for Parkinson's Detection
## Sustained Phonation Task (10-second "AHHH" sounds)

**Total Features: 57**
- Metadata: 3 (filename, healthcode, record_id)
- Acoustic Features: 54

---

## Feature Categories

### **Category 1: MFCCs - Vocal Tract Characteristics (26 features)**

**Purpose:** Captures the spectral envelope and vocal tract resonance patterns

| Feature | What it Measures | Why Relevant for PD |
|---------|-----------------|---------------------|
| `mfcc_1-13_mean` | Average spectral shape (13 coefficients) | Altered vocal tract resonance due to reduced muscle control |
| `mfcc_1-13_std` | How much spectral shape varies over time | Instability in vocal tract positioning, tremor effects |

**Simple Explanation:** Like a fingerprint of your voice's resonance. Even saying "AHHH", PD patients have different vocal tract vibrations due to reduced articulation and muscle control.

**Interpretation with Examples:**
- **`mfcc_1_mean = -150.2`**: Average energy content (typically negative values, ranges -200 to -50)
  - Lower values = quieter overall
  - Similar between healthy and PD (both saying "AHHH")
  
- **`mfcc_2_mean = 85.3`**: Spectral slope/balance (typically 50-150)
  - Higher = brighter voice
  - PD often lower (darker, breathier voice)
  
- **`mfcc_5_std = 12.4`**: Variation in mid-frequency resonance
  - Healthy: 5-10 (stable)
  - PD: 12-20+ (unstable vocal tract control, tremor)

**Computation:** 
- Waveform → STFT → Mel-filterbank → Log → DCT → 13 MFCCs per time frame
- Statistical summary: mean and std across all time frames

---

### **Category 2: Spectral Features - Frequency Characteristics (8 features)**

**Purpose:** Captures voice quality, breathiness, and frequency distribution

| Feature | What it Measures | Normal Range | PD Impact |
|---------|-----------------|--------------|-----------|
| `spectral_centroid_mean` | Average "brightness" (Hz) | 2000-3000 Hz | Lower (hoarse/breathy voice) |
| `spectral_centroid_std` | Brightness variation | Low | Higher (unstable voice) |
| `spectral_bandwidth_mean` | Frequency spread (Hz) | Wide | Narrower (reduced range) |
| `spectral_bandwidth_std` | Spread variation | Low | Higher (instability) |
| `spectral_rolloff_mean` | High-frequency cutoff (Hz) | ~3500 Hz | Variable (breathiness) |
| `spectral_rolloff_std` | Rolloff variation | Low | Higher (unsteady) |
| `zcr_mean` | Average noisiness | Low | Higher (breathy voice) |
| `zcr_std` | Noisiness variation | Low | Higher (inconsistent) |

**Simple Explanations:**
- **Spectral Centroid** = Center of gravity of frequencies (bright vs dull voice)
  - Example: `2450 Hz` = bright voice, `1800 Hz` = darker/breathier voice
  
- **Spectral Bandwidth** = How spread out the frequencies are
  - Example: `1800 Hz` = wide range (full voice), `1200 Hz` = narrow (weak voice)
  
- **Spectral Rolloff** = Frequency below which 85% of energy is contained
  - Example: `3500 Hz` = good high frequencies, `2800 Hz` = lacking high frequencies (hoarse)
  
- **Zero-Crossing Rate** = How noisy (vs pure tonal) the voice is
  - Example: `0.15` = smooth/tonal, `0.35` = noisy/breathy

**Interpretation with Examples:**

**`spectral_centroid_mean = 2450 Hz`** (Healthy)
- Bright, clear voice
- Good harmonic content

**`spectral_centroid_mean = 1820 Hz`** (PD)
- Darker, breathier voice
- Loss of high-frequency harmonics

**`spectral_centroid_std = 145 Hz`** (Healthy)
- Very stable brightness

**`spectral_centroid_std = 280 Hz`** (PD)
- Unstable, fluctuating voice quality

**`zcr_mean = 0.18`** (Healthy)
- Smooth, tonal voice

**`zcr_mean = 0.35`** (PD)
- Breathy, noisy voice (air leakage)

**Computation:**
- Computed frame-by-frame from STFT
- Statistical summary: mean and std across time

---

### **Category 3: Pitch Features - Vocal Cord Vibration (4 features)**

**Purpose:** Measures pitch stability - should be very steady for sustained "AHHH"

| Feature | What it Measures | Expected for "AHHH" | PD Pattern |
|---------|-----------------|---------------------|------------|
| `pitch_mean` | Average pitch (Hz) | 100-250 Hz (gender-dependent) | Similar baseline |
| `pitch_std` | Pitch instability (Hz) | <5 Hz (very stable) | **>10 Hz (tremor)** ⚠️ |
| `pitch_range` | Max - min pitch (Hz) | <20 Hz | **>50 Hz (very unstable)** ⚠️ |
| `pitch_variation_coef` | Relative variation (std/mean) | <0.02 | **>0.05 (high tremor)** ⚠️ |

**Simple Explanation:** 
How steady you can hold your pitch. Healthy voice = rock steady. PD voice = wobbles up and down (vocal tremor).

**Interpretation with Examples:**

**`pitch_mean = 145 Hz`** (Male) or **`210 Hz`** (Female)
- Just the average pitch - similar for healthy and PD
- Men: 85-180 Hz, Women: 165-255 Hz

**`pitch_std = 3.2 Hz`** (Healthy)
- Very stable pitch
- Wobbles only ±3 Hz around average
- **Interpretation: Rock-steady voice control**

**`pitch_std = 12.8 Hz`** (PD with tremor)
- Unstable pitch, wobbles ±13 Hz
- **Interpretation: Voice shakes noticeably, can't hold steady**

**`pitch_range = 15 Hz`** (Healthy)
- From lowest to highest pitch only varies 15 Hz
- Very controlled

**`pitch_range = 65 Hz`** (PD)
- Pitch swings wildly across 65 Hz range
- **Interpretation: Can't maintain steady pitch, tremor causes wide swings**

**`pitch_variation_coef = 0.018`** (Healthy)
- Relative variation: std/mean = 3.2/145 = 0.022
- Less than 2% variation

**`pitch_variation_coef = 0.088`** (PD)
- std/mean = 12.8/145 = 0.088
- Almost 9% variation
- **Interpretation: Nearly 10% wobble - severe instability**

**Computation:**
- Parselmouth (Praat) autocorrelation method for F0 extraction
- Extracts pitch for each time frame
- Statistical summary of voiced frames only

**Clinical Significance:**
- `pitch_std` is a **key tremor indicator**
- Pitch should be extremely stable during sustained phonation
- Any significant variation indicates poor vocal control

---

### **Category 4: Voice Quality - Micro-Perturbations (6 features)**

**Purpose:** Gold standard clinical markers for voice disorders

| Feature | What it Measures | Healthy Threshold | PD Values |
|---------|-----------------|-------------------|-----------|
| `jitter_local` | Pitch cycle-to-cycle variation (%) | <1% | **>1.5% (pathological)** ⚠️ |
| `jitter_rap` | 3-cycle pitch variation (%) | <0.5% | **>1% (tremor)** ⚠️ |
| `shimmer_local` | Amplitude cycle variation (%) | <3% | **>5% (weak cords)** ⚠️ |
| `shimmer_apq3` | 3-cycle amplitude variation (%) | <3% | **>6% (instability)** ⚠️ |
| `hnr_mean` | Voice clarity (dB) | >20 dB (clear) | **<15 dB (hoarse)** ⚠️ |
| `hnr_std` | HNR consistency (dB) | <3 dB | **>5 dB (variable quality)** ⚠️ |

**Simple Explanations:**
- **Jitter** = Pitch wobbling on a tiny scale (cycle-to-cycle, milliseconds)
  - Example: `0.65%` = very small wobbles (healthy), `2.1%` = noticeable instability (PD)
  
- **Shimmer** = Loudness wobbling cycle-by-cycle
  - Example: `2.3%` = smooth amplitude (healthy), `6.8%` = jerky loudness (PD)
  
- **HNR (Harmonic-to-Noise Ratio)** = Clean voice (high) vs noisy/breathy voice (low)
  - Example: `22 dB` = clear voice, `12 dB` = hoarse/breathy voice

**Interpretation with Examples:**

**`jitter_local = 0.58%`** (Healthy)
- Each pitch cycle varies only 0.58% from previous
- Very smooth vocal cord vibration
- **Interpretation: Vocal cords vibrating regularly, like a well-tuned instrument**

**`jitter_local = 2.34%`** (PD - Pathological)
- Pitch cycles vary 2.34% cycle-to-cycle
- **Interpretation: Vocal cords can't vibrate regularly - tremor/weakness causing micro-wobbles**

**`shimmer_local = 2.1%`** (Healthy)
- Loudness varies only 2.1% cycle-to-cycle
- Smooth amplitude control

**`shimmer_local = 7.2%`** (PD - Pathological)
- Loudness jumps 7% between adjacent cycles
- **Interpretation: Can't maintain steady vocal cord closure - weak/trembling muscles**

**`hnr_mean = 23.5 dB`** (Healthy)
- Voice is 23.5 dB clearer than background noise
- Harmonics dominate over breathiness
- **Interpretation: Clean, strong voice - vocal cords closing completely**

**`hnr_mean = 11.8 dB`** (PD - Pathological)
- Only 11.8 dB separation between voice and noise
- **Interpretation: Breathy, hoarse voice - vocal cords not closing fully (air leakage)**

**`hnr_std = 2.1 dB`** (Healthy)
- HNR stays consistent

**`hnr_std = 5.8 dB`** (PD)
- Voice quality fluctuates wildly
- **Interpretation: Sometimes clear, sometimes breathy - inconsistent vocal cord control**

**Computation:**
- Parselmouth (Praat) point process analysis
- Identifies glottal pulses (vocal cord closures)
- Measures period-to-period variations
- HNR: separates harmonic (periodic) from noise (aperiodic) components

**Clinical Significance:**
- **Most sensitive markers for voice disorders**
- Used in clinical practice for voice assessment
- Directly measure vocal cord dysfunction
- **Jitter >1%, Shimmer >3%, HNR <15 dB** = pathological voice

---

### **Category 5: Energy Dynamics - Loudness Patterns (13 features)**

**Purpose:** Captures tremor, choppiness, interruptions, and overall stability

#### **5A: Overall Energy (4 features)**

| Feature | What it Measures | Healthy "AHHH" | PD "AHHH" |
|---------|-----------------|----------------|-----------|
| `energy_variation` | Loudness fluctuation (std/mean) | <0.2 (very stable) | **>0.4 (unstable)** ⚠️ |
| `dynamic_range` | Loudness range (max/min ratio) | 3-5 | Variable (weak or erratic) |
| `energy_entropy` | Complexity of loudness | Low (predictable) | Higher (chaotic) |
| `low_energy_frame_ratio` | % of very quiet frames | <0.15 | **>0.3 (weak voice)** ⚠️ |

**Simple Explanation:**
- **Energy variation** = Overall stability of loudness (should be steady)
- **Dynamic range** = Difference between loudest and quietest parts
- **Energy entropy** = Predictability vs randomness of loudness
- **Low-energy frames** = Percentage of time voice is very weak

**Interpretation with Examples:**

**`energy_variation = 0.15`** (Healthy)
- Coefficient of variation: std/mean = 0.15
- Loudness varies only 15% around average
- **Interpretation: Steady breath support, consistent loudness**

**`energy_variation = 0.48`** (PD)
- Loudness fluctuates 48% around average
- **Interpretation: Can't maintain steady airflow - tremor or poor breath control**

**`dynamic_range = 4.2`** (Healthy)
- Loudest moment is 4.2× the quietest
- Reasonable variation

**`dynamic_range = 12.5`** (PD)
- Loudest is 12.5× quietest - huge swings
- **Interpretation: Either very weak moments or erratic bursts - unstable**

**`energy_entropy = 5.2`** (Healthy)
- Low entropy = predictable loudness pattern

**`energy_entropy = 7.8`** (PD)
- High entropy = chaotic loudness
- **Interpretation: Unpredictable energy - loss of fine motor control**

**`low_energy_frame_ratio = 0.08`** (Healthy)
- Only 8% of time very quiet
- Strong sustained phonation

**`low_energy_frame_ratio = 0.35`** (PD)
- 35% of time very weak
- **Interpretation: Weak voice (hypophonia) or frequent energy drops**

#### **5B: Choppiness/Roughness - "Hachuré" (4 features)**

| Feature | What it Measures | Healthy | PD |
|---------|-----------------|---------|-----|
| `energy_instability` | Std of loudness changes | Low | **High (jumpy)** ⚠️ |
| `energy_roughness` | Average magnitude of changes | Low | **High (choppy)** ⚠️ |
| `abrupt_changes_ratio` | % of sudden jumps | <0.05 | **>0.15 (very choppy)** ⚠️ |
| `voice_breaks_ratio` | % of very weak moments | <0.1 | **>0.3 (breaking voice)** ⚠️ |

**Simple Explanation:**
- Voice should transition smoothly in loudness
- PD patients have jerky, unsmooth transitions ("hachuré" = choppy)
- Measures how much the voice jumps around vs glides smoothly

**Interpretation with Examples:**

**`energy_instability = 0.008`** (Healthy)
- Std of frame-to-frame changes = 0.008
- Smooth transitions between frames
- **Interpretation: Voice glides smoothly like a violin note**

**`energy_instability = 0.024`** (PD)
- Std of changes = 0.024 (3× higher)
- **Interpretation: Voice jumps erratically - like a stuttering engine**

**`energy_roughness = 0.012`** (Healthy)
- Average magnitude of changes = 0.012
- Gentle loudness variations

**`energy_roughness = 0.035`** (PD)
- Average change = 0.035
- **Interpretation: Choppy voice, can't make smooth transitions**

**`abrupt_changes_ratio = 0.03`** (Healthy)
- Only 3% of transitions are sudden jumps
- Mostly smooth

**`abrupt_changes_ratio = 0.18`** (PD)
- 18% of transitions are abrupt (>2 std)
- **Interpretation: Frequent sudden jerks in loudness - tremor/dyscontrol**

**`voice_breaks_ratio = 0.05`** (Healthy)
- 5% of time energy drops below 50% of mean
- Occasional dips

**`voice_breaks_ratio = 0.32`** (PD)
- 32% of time voice very weak
- **Interpretation: Voice keeps cutting out - "hachuré" pattern**

**Computation:**
- Frame-to-frame energy differences
- Count sudden changes (>2 std deviations)
- Detect energy drops below 50% of mean

#### **5C: Tremor Speed (2 features)**

| Feature | What it Measures | Healthy | PD |
|---------|-----------------|---------|-----|
| `tremor_frequency` | Shaking speed (Hz) | 0 (no tremor) | **4-6 Hz (Parkinsonian)** ⚠️ |
| `energy_oscillation_rate` | Fluctuation rate | Low | **High (rapid wobbles)** ⚠️ |

**Simple Explanation:**
- **Tremor frequency** = How many times per second the voice shakes
- **4-6 Hz is the Parkinsonian tremor signature**
- Different from essential tremor (8-12 Hz) or normal variation

**Interpretation with Examples:**

**`tremor_frequency = 0 Hz`** (Healthy)
- No periodic oscillation detected
- Voice doesn't shake rhythmically
- **Interpretation: Steady voice, no tremor**

**`tremor_frequency = 2.1 Hz`** (Irregular)
- Very slow oscillation, not pathological
- Could be natural variation or breathing

**`tremor_frequency = 4.8 Hz`** (PD - Diagnostic!)
- Classic Parkinsonian tremor range (4-6 Hz)
- Voice shakes ~5 times per second
- **Interpretation: PARKINSONIAN TREMOR - highly diagnostic for PD**

**`tremor_frequency = 9.5 Hz`** (Essential Tremor)
- Faster tremor (8-12 Hz range)
- **Interpretation: Essential tremor, NOT Parkinsonian**

**`energy_oscillation_rate = 0.12`** (Healthy)
- Energy crosses mean 12% of the time
- Low fluctuation

**`energy_oscillation_rate = 0.34`** (PD)
- Energy crosses mean 34% of the time
- **Interpretation: Rapid fluctuations, unstable loudness control**

**Computation:**
- Autocorrelation of energy signal to find dominant periodicity
- Peak detection in autocorrelation
- Convert lag to frequency (Hz)

**Clinical Significance:**
- **4-6 Hz tremor is diagnostic for Parkinson's**
- Distinguishes PD from other movement disorders
- Captures the speed of vocal tremor

#### **5D: Voice Interruptions (2 features)**

| Feature | What it Measures | Healthy | PD |
|---------|-----------------|---------|-----|
| `num_interruptions` | Count of stop/start cycles | 0 | **2-10+ (can't sustain)** ⚠️ |
| `avg_interruption_duration` | Average break length (seconds) | 0 | **0.2-1s (voice fails)** ⚠️ |

**Simple Explanation:**
- Voice should be continuous for 10 seconds
- PD patients struggle to sustain - voice stops and restarts
- Measures both **how many times** it happens and **how long** each break lasts

**Interpretation with Examples:**

**`num_interruptions = 0`** (Healthy)
- Voice sustained continuously
- **Interpretation: Good breath support and vocal cord control**

**`num_interruptions = 1`** (Borderline)
- One brief stop/restart
- Could be normal (swallowing, breath)

**`num_interruptions = 6`** (PD)
- Voice stopped and restarted 6 times in 10 seconds
- **Interpretation: Can't sustain phonation - vocal fatigue, poor breath control**

**`num_interruptions = 15`** (Severe PD)
- Voice breaks 15 times
- **Interpretation: Severe inability to maintain steady vocalization**

**`avg_interruption_duration = 0 seconds`** (Healthy)
- No interruptions

**`avg_interruption_duration = 0.3 seconds`** (Mild PD)
- Each break lasts ~0.3 seconds on average
- **Interpretation: Brief micro-interruptions, voice recovery quick**

**`avg_interruption_duration = 0.85 seconds`** (Severe PD)
- Each interruption lasts nearly 1 second
- **Interpretation: Long pauses where voice completely fails**

**Real-world example:**
- Healthy: "AAAAaaaaaaaaaaahhhhhhh" (10 seconds continuous)
- PD: "AAAaa...aaa..AAhh...aa...hh" (multiple breaks)

**Computation:**
- Define silence as energy < 30% of mean
- Count voice → silence → voice transitions
- Measure duration of silent segments

**Clinical Significance:**
- Indicates vocal fatigue and poor breath support
- Reflects inability to sustain phonation
- Different from general weakness (measures actual interruptions)

#### **5E: Combined Hachuré Index (1 feature)**

| Feature | What it Measures | Range | Interpretation |
|---------|-----------------|-------|----------------|
| `hachure_index` | Overall choppy-trembling score | 0-1 | <0.2=smooth, 0.5+=severe ⚠️ |

**Simple Explanation:**
- **Single composite score** combining:
  - 40% tremor frequency (is it shaking?)
  - 30% energy roughness (is it choppy?)
  - 30% abrupt changes (sudden jumps?)
- High score = voice is both trembling AND choppy (classic PD)

**Interpretation with Examples:**

**`hachure_index = 0.08`** (Healthy)
- Very low score
- Voice is smooth and steady
- **Interpretation: No tremor, no choppiness - healthy voice**

**`hachure_index = 0.25`** (Mild PD)
- Low-moderate score
- Some instability detected
- **Interpretation: Slight tremor OR mild choppiness**

**`hachure_index = 0.52`** (Moderate PD)
- High score
- Clear pathology
- **Interpretation: Voice is both trembling AND choppy - classic "hachuré" pattern**

**`hachure_index = 0.78`** (Severe PD)
- Very high score
- Severe dysfunction
- **Interpretation: Pronounced tremor + extreme choppiness - severe dyscontrol**

**Breakdown example (Moderate PD, hachure_index = 0.52):**
- Tremor component: `tremor_frequency = 5.2 Hz` → normalized to 0.52 → weighted 0.40 × 0.52 = **0.208**
- Roughness component: `energy_roughness = 0.035` → ×100 = 3.5, capped at 1.0 → weighted 0.30 × 1.0 = **0.300**
- Abrupt component: `abrupt_changes_ratio = 0.04` → weighted 0.30 × 0.04 = **0.012**
- **Total: 0.208 + 0.300 + 0.012 = 0.520**

**Computation:**
```
If tremor detected:
  hachure_index = 0.4 × (tremor_freq/10) + 0.3 × (roughness×100) + 0.3 × abrupt_ratio
Else:
  hachure_index = 0.5 × (roughness×100) + 0.5 × abrupt_ratio
```

---

## Feature Importance by Clinical Relevance

### **🔴 PRIMARY PD MARKERS (Most Diagnostic)**
1. **`jitter_local`**, **`shimmer_local`** - Gold standard voice disorder markers used in clinical practice
2. **`hnr_mean`** - Voice quality (hoarseness/breathiness), <15 dB = pathological
3. **`tremor_frequency`** - 4-6 Hz = Parkinsonian tremor signature (diagnostic)
4. **`pitch_std`** - Pitch instability, should be <5 Hz for healthy sustained phonation
5. **`hachure_index`** - Combined choppy-trembling score, >0.5 = severe

### **🟡 SECONDARY MARKERS (Strong Supporting Evidence)**
6. **`num_interruptions`**, **`avg_interruption_duration`** - Vocal fatigue, inability to sustain
7. **`energy_roughness`**, **`abrupt_changes_ratio`** - Choppiness, lack of smooth control
8. **`spectral_centroid_mean`** - Voice brightness, lower in hoarse/breathy PD voice
9. **`low_energy_frame_ratio`**, **`voice_breaks_ratio`** - Weak voice, hypophonia

### **🟢 CONTEXTUAL FEATURES (Useful but Less Specific)**
10. **MFCCs** - Overall vocal characteristics, general spectral patterns
11. **Spectral bandwidth/rolloff** - Frequency distribution, voice quality nuances
12. **Energy variation/entropy** - General stability measures

---

## Data Processing Pipeline

```
1. Input: Normalized .npy waveform (10 seconds, 44100 Hz)
   ↓
2. Frame-wise Analysis (sliding windows ~23ms, 50% overlap)
   ↓
3. Feature Extraction per Frame:
   - MFCCs: STFT → Mel-filterbank → Log → DCT
   - Spectral: STFT → Compute centroid, bandwidth, rolloff, ZCR
   - Pitch: Autocorrelation → F0 tracking
   - Energy: RMS computation
   ↓
4. Voice Quality Analysis:
   - Point process → Identify glottal pulses
   - Jitter/Shimmer: Period variations
   - HNR: Harmonic vs noise separation
   ↓
5. Tremor & Interruption Detection:
   - Energy autocorrelation → Tremor frequency
   - Silence detection → Count interruptions
   ↓
6. Statistical Summarization (per feature):
   - Mean across all frames
   - Std (standard deviation) across frames
   ↓
7. Output: Single row with 57 features (+ 3 metadata)
```

---

## Feature Extraction Libraries

- **librosa**: MFCCs, spectral features, energy (RMS), ZCR
- **parselmouth** (Praat): Pitch (F0), jitter, shimmer, HNR
- **numpy**: Statistical computations, autocorrelation, signal processing
- **pandas**: DataFrame creation and CSV export

---

## Output Format

**CSV file:** `acoustic_features.csv`

**Structure:**
- **Rows**: One per audio recording
- **Columns**: 
  - 3 metadata: `filename`, `healthcode`, `record_id`
  - 54 acoustic features

**Example:**
```
filename,healthcode,record_id,mfcc_1_mean,mfcc_1_std,...,hachure_index
abc123_def456_audio.npy,abc123,def456,-150.2,12.3,...,0.35
```

---

## Clinical Interpretation Guide

### **Healthy Voice Pattern:**
- Jitter <1%, Shimmer <3%, HNR >20 dB
- Pitch std <5 Hz (very stable)
- No tremor (tremor_frequency ≈ 0)
- No interruptions (num_interruptions = 0)
- Smooth energy (hachure_index <0.2)

### **Mild PD Pattern:**
- Jitter 1-2%, Shimmer 3-5%, HNR 15-20 dB
- Pitch std 5-10 Hz (some instability)
- Low tremor 3-5 Hz
- Few interruptions (1-3)
- Moderate choppiness (hachure_index 0.2-0.4)

### **Moderate-Severe PD Pattern:**
- Jitter >2%, Shimmer >6%, HNR <15 dB
- Pitch std >10 Hz (very unstable)
- Clear tremor 4-6 Hz (Parkinsonian signature)
- Multiple interruptions (5-10+)
- Severe choppiness (hachure_index >0.5)

---

## Notes on Normalization

All features work with **normalized waveforms** ([-1, 1] range) because they measure:
- **Patterns** (not absolute values)
- **Relative relationships** (ratios, variations)
- **Temporal dynamics** (how things change over time)
- **Frequency content** (spectral patterns)

**What we lose:** Absolute loudness (hypophonia detection)

**What we keep:** Everything else! All frequency-based, stability, tremor, and quality features are preserved.

This is **appropriate for mPower dataset** where recordings are made with different devices at home (uncontrolled microphone distance/sensitivity).

---

## Usage

```bash
# Extract features from all .npy files
python extract_acoustic_features.py

# Input: /mloscratch/users/gnahas/data/waveform_norm/*.npy
# Output: /mloscratch/users/gnahas/data/features/acoustic_features.csv
```

---

## References

**Clinical Voice Analysis:**
- Jitter, Shimmer, HNR: Standard Praat voice quality measures
- Tremor frequency 4-6 Hz: Established Parkinsonian tremor range

**Signal Processing:**
- MFCCs: Standard speech/audio feature representation
- Spectral features: Common audio descriptors (librosa)

**PD Voice Characteristics:**
- Hypophonia, monotone, hoarseness, breathiness
- Vocal tremor, reduced pitch variation
- Difficulty sustaining phonation
