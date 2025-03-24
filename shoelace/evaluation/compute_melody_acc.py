import pretty_midi
import numpy as np
import mir_eval
import json
import editdistance
from tqdm import tqdm
import jiwer

def load_melody_track(midi_path):
    """Load the 'MELODY' track from a MIDI file."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    for instrument in pm.instruments:
        if instrument.name == "MELODY" or instrument.program > 0:
            return instrument.notes
    raise ValueError(f"No 'MELODY' track found in {midi_path}")

def convert_notes(notes, ignore_octave=False):
    """
    Convert a list of note objects to a 2D NumPy array of shape (n,3).
    Each row is [onset, offset, pitch]. If ignore_octave is True, the pitch is
    reduced modulo 12 (with an offset to avoid zero).
    """
    if ignore_octave:
        arr = np.array([[n.start, n.end, n.pitch % 12 + 1] for n in notes], dtype=float)
    else:
        arr = np.array([[n.start, n.end, n.pitch] for n in notes], dtype=float)
    return np.atleast_2d(arr)

def compute_note_level_f1(gt_notes, pred_notes,
                          onset_tolerance=0.5,
                          pitch_tolerance=50.0,
                          offset_ratio=0.2,
                          offset_min_tolerance=0.5,
                          strict=False,
                          beta=1.0,
                          ignore_octave=False):
    """
    Compute note-level precision, recall, F-measure and Average Overlap Ratio.
    If ignore_octave is True, note pitches are reduced modulo 12.
    """
    gt_arr = convert_notes(gt_notes, ignore_octave=ignore_octave)
    pred_arr = convert_notes(pred_notes, ignore_octave=ignore_octave)
    gt_intervals, gt_pitches = gt_arr[:, :2], gt_arr[:, 2]
    pred_intervals, pred_pitches = pred_arr[:, :2], pred_arr[:, 2]
    mir_eval.transcription.validate(gt_intervals, gt_pitches, pred_intervals, pred_pitches)
    precision, recall, f_measure, avg_overlap_ratio = mir_eval.transcription.precision_recall_f1_overlap(
        gt_intervals, gt_pitches, pred_intervals, pred_pitches,
        onset_tolerance=onset_tolerance,
        offset_min_tolerance=offset_min_tolerance,
    )
    return precision, recall, f_measure, avg_overlap_ratio

def notes_to_pitch_sequence(notes, duration, frame_rate=25, ignore_octave=False):
    """
    Create a frame-level pitch sequence from note events.
    Frames with no active note are set to 0.
    When ignore_octave is True, note pitches are reduced modulo 12.
    """
    num_frames = int(np.ceil(duration * frame_rate))
    seq = np.full(num_frames, 0, dtype=int)
    for n in notes:
        start = int(np.floor(n.start * frame_rate))
        end = int(np.ceil(n.end * frame_rate))
        seq[start:end] = n.pitch % 12 if ignore_octave else n.pitch
    return seq

def compute_frame_accuracy(gt_notes, pred_notes, frame_rate=25, ignore_octave=False):
    """
    Compute frame-wise accuracy between groundtruth and predicted notes.
    """
    if not gt_notes or not pred_notes:
        raise ValueError("One of the note sets is empty.")
    duration = max(max(n.end for n in gt_notes), max(n.end for n in pred_notes))
    gt_seq = notes_to_pitch_sequence(gt_notes, duration, frame_rate, ignore_octave)
    pred_seq = notes_to_pitch_sequence(pred_notes, duration, frame_rate, ignore_octave)
    return np.mean(gt_seq == pred_seq)

def compute_pitch_error_rate(reference, hypothesis):
    """
    Compute the Word Error Rate (WER) for numerical sequences.

    Args:
        reference (list of int): The ground truth sequence.
        hypothesis (list of int): The predicted sequence.

    Returns:
        float: The WER score.
    """
    ref_len = len(reference)

    # Edge cases
    if ref_len == 0:
        return float(len(hypothesis))  # If reference is empty, WER is the length of hypothesis

    # Initialize edit distance matrix
    d = np.zeros((len(reference) + 1, len(hypothesis) + 1), dtype=int)

    for i in range(len(reference) + 1):
        d[i][0] = i  # Deletions
    for j in range(len(hypothesis) + 1):
        d[0][j] = j  # Insertions

    # Compute Levenshtein distance
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                cost = 0  # No error if numbers match
            else:
                cost = 1  # Substitution error

            d[i][j] = min(
                d[i - 1][j] + 1,      # Deletion
                d[i][j - 1] + 1,      # Insertion
                d[i - 1][j - 1] + cost  # Substitution
            )

    wer_score = d[len(reference)][len(hypothesis)] / ref_len
    return wer_score

def process_file(file_list_path, tol):
    """
    Process a file containing pairs of file paths.
    Each line should have two paths separated by a tab.
    Returns a list of dictionaries with computed metrics for each pair.
    A progress bar is displayed via tqdm.
    """
    results = []
    with open(file_list_path, 'r') as f:
        lines = [line.strip().split("\t") for line in f if line.strip()]

    for gt_path, gen_path in tqdm(lines, desc="Processing file pairs"):
        #print(gt_path, gen_path)
        try:
            gt_notes = load_melody_track(gt_path)
        except ValueError as e:
            print(e)
            continue
        pred_notes = sorted(load_melody_track(gen_path), key=lambda n: n.start)

        # Compute note-level metrics (exact pitch).
        precision, recall, note_f1, avg_overlap = compute_note_level_f1(
            gt_notes, pred_notes, onset_tolerance=tol, offset_min_tolerance=tol, ignore_octave=False)
        # Compute note-level metrics (ignoring octave).
        (precision_ignore, recall_ignore, note_f1_ignore,
         avg_overlap_ignore) = compute_note_level_f1(
            gt_notes, pred_notes, onset_tolerance=tol, offset_min_tolerance=tol, ignore_octave=True)

        # Compute frame-level accuracy.
        frame_acc = compute_frame_accuracy(gt_notes, pred_notes, ignore_octave=False)
        frame_acc_ignore = compute_frame_accuracy(gt_notes, pred_notes, ignore_octave=True)

        # Compute Pitch Error Rate on the pitch sequences.
        pitch_ref = [n.pitch for n in gt_notes]
        pitch_hyp = [n.pitch for n in pred_notes]
        pitch_er = compute_pitch_error_rate(pitch_ref, pitch_hyp)
        pitch_ref_ignore = [n.pitch % 12 for n in gt_notes]
        pitch_hyp_ignore = [n.pitch % 12 for n in pred_notes]
        pitch_er_ignore = compute_pitch_error_rate(pitch_ref_ignore, pitch_hyp_ignore)

        results.append({
            'groundtruth': gt_path,
            'generated': gen_path,
            'note_precision': precision,
            'note_recall': recall,
            'note_f1': note_f1,
            'avg_overlap_ratio': avg_overlap,
            'note_precision_ignore_octave': precision_ignore,
            'note_recall_ignore_octave': recall_ignore,
            'note_f1_ignore_octave': note_f1_ignore,
            'avg_overlap_ratio_ignore': avg_overlap_ignore,
            'frame_accuracy': frame_acc,
            'frame_accuracy_ignore_octave': frame_acc_ignore,
            'pitch_error_rate': pitch_er,
            'pitch_error_rate_ignore': pitch_er_ignore
        })
    return results

def compute_aggregate_metrics(results):
    """Compute average metrics across all songs."""
    if not results:
        return {}
    aggregate = {
        'note_precision': np.mean([r['note_precision'] for r in results]),
        'note_recall': np.mean([r['note_recall'] for r in results]),
        'note_f1': np.mean([r['note_f1'] for r in results]),
        'avg_overlap_ratio': np.mean([r['avg_overlap_ratio'] for r in results]),
        'note_precision_ignore_octave': np.mean([r['note_precision_ignore_octave'] for r in results]),
        'note_recall_ignore_octave': np.mean([r['note_recall_ignore_octave'] for r in results]),
        'note_f1_ignore_octave': np.mean([r['note_f1_ignore_octave'] for r in results]),
        'avg_overlap_ratio_ignore': np.mean([r['avg_overlap_ratio_ignore'] for r in results]),
        'frame_accuracy': np.mean([r['frame_accuracy'] for r in results]),
        'frame_accuracy_ignore_octave': np.mean([r['frame_accuracy_ignore_octave'] for r in results]),
        'pitch_error_rate': np.mean([r['pitch_error_rate'] for r in results]),
        'pitch_error_rate_ignore': np.mean([r['pitch_error_rate_ignore'] for r in results])
    }
    return aggregate

if __name__ == "__main__":
    import sys
    file_list = sys.argv[1]
    tol = float(sys.argv[2])
    metrics = process_file(file_list, tol)

    print(f"Total file pairs processed: {len(metrics)}")
    aggregate = compute_aggregate_metrics(metrics)
    print("Aggregated Average Scores:")
    #print(f"Note-level Precision:                {aggregate['note_precision']:.3f}")
    #print(f"Note-level Recall:                   {aggregate['note_recall']:.3f}")
    print(f"Note-level F1:                       {aggregate['note_f1']:.3f}")
    #print(f"Average Overlap Ratio:               {aggregate['avg_overlap_ratio']:.3f}")
    #print(f"Note-level Precision (ignore octave):{aggregate['note_precision_ignore_octave']:.3f}")
    #print(f"Note-level Recall (ignore octave):   {aggregate['note_recall_ignore_octave']:.3f}")
    print(f"Note-level F1 (ignore octave):         {aggregate['note_f1_ignore_octave']:.3f}")
    #print(f"Average Overlap Ratio (ignore):      {aggregate['avg_overlap_ratio_ignore']:.3f}")
    print(f"Frame-wise Accuracy:                 {aggregate['frame_accuracy']:.3f}")
    print(f"Frame-wise Accuracy (ignore octave): {aggregate['frame_accuracy_ignore_octave']:.3f}")
    print(f"Pitch Error Rate:                    {aggregate['pitch_error_rate']:.3f}")
    print(f"Pitch Error Rate (ignore octave):    {aggregate['pitch_error_rate_ignore']:.3f}")

    output = {"aggregate": aggregate, "per_song_results": metrics}
    result_file = file_list + ".result.json"
    with open(result_file, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Detailed results saved to {result_file}")