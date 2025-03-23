import pretty_midi
import numpy as np
import mir_eval

def load_melody_track(midi_path):
    """Load the 'MELODY' track from a MIDI file."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    for instrument in pm.instruments:
        if instrument.name == "MELODY":
            return instrument.notes
    raise ValueError(f"No 'MELODY' track found in {midi_path}")

def compute_note_level_f1(gt_notes, pred_notes, onset_tolerance=0.05, offset_ratio=0.5):
    """
    Compute note-level precision, recall and F1 using mir_eval.
    Each note is represented as [onset, offset, pitch].
    """
    # Convert note objects to numpy arrays (exact pitch)
    gt = np.array([[note.start, note.end, note.pitch] for note in gt_notes])
    pred = np.array([[note.start, note.end, note.pitch] for note in pred_notes])
    
    precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_score(
        gt, pred, onset_tolerance=onset_tolerance, offset_ratio=offset_ratio)
    return precision, recall, f1

def compute_note_level_f1_ignore_octave(gt_notes, pred_notes, onset_tolerance=0.05, offset_ratio=0.5):
    """
    Compute note-level precision, recall and F1 ignoring octave errors.
    Pitches are compared modulo 12 so that octave differences are not penalized.
    """
    # Convert note objects to numpy arrays (pitch modulo 12)
    gt = np.array([[note.start, note.end, note.pitch % 12] for note in gt_notes])
    pred = np.array([[note.start, note.end, note.pitch % 12] for note in pred_notes])
    
    precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_score(
        gt, pred, onset_tolerance=onset_tolerance, offset_ratio=offset_ratio)
    return precision, recall, f1

def notes_to_pitch_sequence(notes, duration, frame_rate=50, ignore_octave=False):
    """
    Create a frame-level pitch sequence from note events.
    If no note is active during a frame, the pitch is set to -1 (representing silence).
    When ignore_octave is True, note pitches are reduced modulo 12.
    """
    num_frames = int(np.ceil(duration * frame_rate))
    # Use -1 to represent silence (avoiding conflict with valid modulo values 0-11)
    pitch_seq = np.full(num_frames, -1, dtype=int)
    
    for note in notes:
        start_frame = int(np.floor(note.start * frame_rate))
        end_frame = int(np.ceil(note.end * frame_rate))
        pitch = note.pitch % 12 if ignore_octave else note.pitch
        pitch_seq[start_frame:end_frame] = pitch
    return pitch_seq

def compute_frame_accuracy(gt_notes, pred_notes, frame_rate=50, ignore_octave=False):
    """
    Compute frame-wise accuracy between groundtruth and predicted notes.
    If ignore_octave is True, the comparison is done modulo 12.
    The duration is taken as the maximum end time across both note sets.
    """
    if len(gt_notes) == 0 or len(pred_notes) == 0:
        raise ValueError("One of the note sets is empty.")
    
    duration = max(max(note.end for note in gt_notes), max(note.end for note in pred_notes))
    gt_seq = notes_to_pitch_sequence(gt_notes, duration, frame_rate, ignore_octave)
    pred_seq = notes_to_pitch_sequence(pred_notes, duration, frame_rate, ignore_octave)
    accuracy = np.mean(gt_seq == pred_seq)
    return accuracy

def process_file(file_list_path):
    """
    Process a file containing pairs of file paths.
    Each pair: first line = groundtruth MIDI, second line = generated MIDI.
    Returns a list of dictionaries with computed metrics for each pair.
    """
    results = []
    with open(file_list_path, 'r') as f:
        # Read non-empty lines
        lines = [line.strip() for line in f if line.strip()]
    
    if len(lines) % 2 != 0:
        raise ValueError("The file should contain an even number of non-empty lines (groundtruth/generated pairs).")
    
    for i in range(0, len(lines), 2):
        gt_path = lines[i]
        gen_path = lines[i+1]
        try:
            gt_notes = load_melody_track(gt_path)
        except ValueError as e:
            print(e)
            continue
        
        # For generated MIDI, combine notes from all instruments.
        gen_pm = pretty_midi.PrettyMIDI(gen_path)
        pred_notes = []
        for instrument in gen_pm.instruments:
            pred_notes.extend(instrument.notes)
        # Sort notes by onset time
        pred_notes = sorted(pred_notes, key=lambda note: note.start)
        
        # Compute note-level metrics (exact pitch)
        precision, recall, note_f1 = compute_note_level_f1(gt_notes, pred_notes)
        # Compute note-level metrics ignoring octave errors (pitch mod 12)
        precision_ignore_octave, recall_ignore_octave, note_f1_ignore_octave = compute_note_level_f1_ignore_octave(gt_notes, pred_notes)
        
        # Compute frame-wise accuracy (exact pitch)
        frame_acc = compute_frame_accuracy(gt_notes, pred_notes)
        # Compute frame-wise accuracy ignoring octave errors (pitch mod 12)
        frame_acc_ignore_octave = compute_frame_accuracy(gt_notes, pred_notes, ignore_octave=True)
        
        results.append({
            'groundtruth': gt_path,
            'generated': gen_path,
            'note_precision': precision,
            'note_recall': recall,
            'note_f1': note_f1,
            'note_precision_ignore_octave': precision_ignore_octave,
            'note_recall_ignore_octave': recall_ignore_octave,
            'note_f1_ignore_octave': note_f1_ignore_octave,
            'frame_accuracy': frame_acc,
            'frame_accuracy_ignore_octave': frame_acc_ignore_octave
        })
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python evaluate_melody.py <file_path_pairs.txt>")
        sys.exit(1)
    
    file_list = sys.argv[1]
    metrics = process_file(file_list)
    for res in metrics:
        print(f"Groundtruth: {res['groundtruth']}")
        print(f"Generated:   {res['generated']}")
        print(f"Note-level Precision:                {res['note_precision']:.3f}")
        print(f"Note-level Recall:                   {res['note_recall']:.3f}")
        print(f"Note-level F1:                       {res['note_f1']:.3f}")
        print(f"Note-level Precision (ignore octave):{res['note_precision_ignore_octave']:.3f}")
        print(f"Note-level Recall (ignore octave):   {res['note_recall_ignore_octave']:.3f}")
        print(f"Note-level F1 (ignore octave):         {res['note_f1_ignore_octave']:.3f}")
        print(f"Frame-wise Accuracy:                 {res['frame_accuracy']:.3f}")
        print(f"Frame-wise Accuracy (ignore octave): {res['frame_accuracy_ignore_octave']:.3f}")
        print("-" * 40)
