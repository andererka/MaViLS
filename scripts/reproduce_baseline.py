"""Reproduce the MaViLS baseline for every lecture and log the scores to results/baseline.csv.

For each lecture the script
  1. checks the inputs: transcript sentences vs. ground-truth rows, highest ground-truth slide vs.
     PDF pages, and picks the downloaded video whose duration matches the transcript,
  2. estimates peak RAM and skips lectures that would not fit (run those on a bigger machine),
  3. runs mavils/matching_algorithm.py with --merge_method all in a subprocess at low CPU priority,
  4. scores every result file against the ground truth (same metric as
     evaluation/evaluate_recall_precision.py) and writes one row per method to results/baseline.csv,
     next to the authors' F1 from evaluation/results_F1.xlsx.

Lectures whose result files already exist are only re-scored, unless --force is given.

Examples (with the project's Python environment, from any folder):
    python scripts/reproduce_baseline.py --list
    python scripts/reproduce_baseline.py
    python scripts/reproduce_baseline.py --lectures image_processing climate_and_cities
    python scripts/reproduce_baseline.py --no_memory_check      # e.g. on the supercomputer

Missing videos come from the Kaggle dataset pad19tue/lecture-videos, e.g.
    kaggle datasets download pad19tue/lecture-videos -f video/<name>.mp4 -p data/video   (then unzip)
"""
import argparse
import ctypes
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import fitz
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))
from helpers.prepare_audioscript import generate_output_dict_by_sentence  # noqa: E402

DATA = REPO / 'data'
RUNS_DIR = REPO / 'results' / 'baseline_runs'
CSV_PATH = REPO / 'results' / 'baseline.csv'
AUTHORS_F1 = REPO / 'evaluation' / 'results_F1.xlsx'
# matching_algorithm.py rewrites this tracked file on every run; the script restores it afterwards
TRACKED_SIDE_EFFECT = DATA / 'unlabeled_ground_truth' / 'output_file.xlsx'

# lecture -> input file names (without extension). 'videos' lists candidates; the downloaded one whose
# duration matches the transcript is used. 'authors' is the row name in evaluation/results_F1.xlsx.
# Pairings were checked on 2026-09-16: sentence count == ground-truth rows and identical timestamps for all 20.
LECTURES = {
    'ML_for_health': dict(srt='ML_for_health_MIT', gt='ML_for_health_MIT', pdf='ML_for_health_care_MIT',
                          videos=['ML_for_health_high_res', 'clinical_care_high_res'], authors='ML for health'),
    'cities_and_decarbonization': dict(srt='cities_and_decarbonization', gt='cities_and_decarbonization',
                                       pdf='cities&decarbonization', videos=['cities_and_decarbonization_standard_res'],
                                       authors='Decarbonization'),
    'climate_and_cities': dict(srt='climate_and_cities', gt='climate_and_cities', pdf='cities&climate',
                               videos=['cities_and_climate_high_res'], authors='Climate & Cities'),
    'climate_science_policy': dict(srt='climate_science_policy_MIT', gt='climate_science_policy_MIT2',
                                   pdf='climate_science_policy_MIT',
                                   videos=['clinical_care_high_res', 'ML_for_health_high_res'], authors='Climate policies'),
    'cognitive_robotics': dict(srt='cognitive_robotics_MIT', gt='cognitive_robotics_MIT', pdf='cognitive_robotics_MIT',
                               videos=['cognitive_robotics_high_res'], authors='Cognitive robotics'),
    'computer_vision_2_2': dict(srt='computer_vision_2_2_high_res', gt='computer_vision_2_2',
                                pdf='computer_vision_2_2_geiger', videos=['computer_vision_2_2_high_res'],
                                authors='Computer Vision'),
    'creating_breakthrough_products': dict(srt='creating_breakthrough_products_MIT',
                                           gt='creating_breakthrough_products_MIT', pdf='creating_breakthrough_products',
                                           videos=['creating_breakthrough_products_MIT'], authors='Productdesign'),
    'cryptocurrency': dict(srt='cryptocurrency', gt='cryptocurrency_MIT', pdf='cryptocurrency',
                           videos=['cryptocurrency_high_res'], authors='Cryptocurrency'),
    'deep_learning': dict(srt='deeplearning_goodfellow', gt='deeplearning', pdf='deeplearning_goodfellow',
                          videos=['deep_learning_high_res'], authors='Deep learning'),
    'image_processing': dict(srt='image_processing_high_res', gt='image_processing', pdf='image_processing',
                             videos=['image_processing_high_res'], authors='Image processing'),
    'numerics': dict(srt='numerics_hennig', gt='numerics', pdf='numerics', videos=['numerics_high_res'],
                     authors='Numerics'),
    'phonetics': dict(srt='phonetics', gt='phonetics', pdf='phonetics', videos=['phonetics_high_res'],
                      authors='Phonetics'),
    'physics': dict(srt='physics_intro_02', gt='physics', pdf='physics_intro_02', videos=['physics_high_res', 'physics'],
                    authors='Physics'),
    'psychology': dict(srt='psychology_MIT', gt='psychology', pdf='psychology', videos=['psychology_high_res'],
                       authors='Psychology'),
    # RL_deepmind.pdf has only 42 pages while the ground truth goes up to slide 49
    'reinforcement_learning': dict(srt='reinforcement_learning_silver', gt='reinforcement_learning',
                                   pdf='reinforcement_learning', videos=['reinforcement_learning_high_res'],
                                   authors='Reinforcement'),
    'sensory_systems': dict(srt='visual_system_MIT', gt='sensory_systems', pdf='sensory_systems',
                            videos=['sensory_systems_high_res'], authors='Sensory systems'),
    'short_range': dict(srt='short_range_mit', gt='short_range', pdf='short_range_mit', videos=['short_range_MIT'],
                        authors='Short range'),
    'solar_resource': dict(srt='solar_resource', gt='solar_resource', pdf='solar_resource',
                           videos=['solar_resource_high_res'], authors='Solar resource'),
    'team_dynamics': dict(srt='team_dynamics_game_design_MIT', gt='team_dynamics_game_design_MIT',
                          pdf='team_dynamics_game_design_mit', videos=['team_dynamics_high_res'], authors='Team dynamics'),
    'theory_of_computation': dict(srt='theory_of_computation_MIT', gt='theory_of_computation',
                                  pdf='theory_of_computation', videos=['theory_of_computation_high_res'],
                                  authors='Theory of Computation'),
}

# method -> (result file suffix written by matching_algorithm.py, column in evaluation/results_F1.xlsx)
METHODS = {
    'ocr': ('ocr_{jp_file}', 'OCR {jp}'),
    'audio': ('audiomatching_{jp_file}', 'Audio {jp}'),
    'image': ('image_matching_{jp_file}', 'Images {jp}'),
    'mean': ('mean_matching_all_{jp_file}', 'All {jp} mean'),
    'max': ('max_matching_all_{jp_file}', 'All {jp} max'),
    'weighted_sum': ('weighted_sum_matching_all_{jp_file}', 'All {jp} weighted sum'),
}


def available_memory_mb():
    """returns the RAM that can be used without swapping in MB, or None if it cannot be determined"""
    if sys.platform == 'win32':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullAvailPhys / 2**20
        return None
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def inspect_lecture(name, spec):
    """collects the input paths, sanity checks and a peak-RAM estimate for one lecture

    Args:
        name (str): lecture key of LECTURES
        spec (dict): entry of LECTURES

    Returns:
        dict: paths, frame/page counts, chosen video, estimated RAM (MB) and a list of problems
    """
    info = {'lecture': name, 'problems': [], 'notes': '',
            'srt': DATA / 'audioscripts' / f"{spec['srt']}.srt",
            'gt_path': DATA / 'ground_truth_files' / f"ground_truth_{spec['gt']}.xlsx",
            'pdf': DATA / 'lectures' / f"{spec['pdf']}.pdf"}

    with tempfile.TemporaryDirectory() as tmp:
        sentences = generate_output_dict_by_sentence(str(info['srt']), output_file=os.path.join(tmp, 'sentences.xlsx'))
    gt = pd.read_excel(info['gt_path'])['Slidenumber']
    info['gt'] = gt
    info['frames'] = len(sentences)
    transcript_minutes = max(sentences) / 60
    if len(gt) != len(sentences):
        info['problems'].append(f'{len(sentences)} transcript sentences but {len(gt)} ground-truth rows')

    doc = fitz.open(info['pdf'])
    info['pages'] = len(doc)
    # slides are rendered at 2x zoom and every video frame is resized to that size
    page_mb = int(doc[0].rect.width * 2) * int(doc[0].rect.height * 2) * 3 / 2**20
    doc.close()
    if gt.max() > info['pages']:
        info['problems'].append(f'ground truth goes up to slide {gt.max()} but the PDF has {info["pages"]} pages')

    candidates = []
    for video_name in spec['videos']:
        path = DATA / 'video' / f'{video_name}.mp4'
        if not path.exists():
            continue
        cap = cv2.VideoCapture(str(path))
        fps, frame_count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_mb = cap.get(cv2.CAP_PROP_FRAME_WIDTH) * cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 3 / 2**20
        cap.release()
        if fps > 0:
            candidates.append((abs(frame_count / fps / 60 - transcript_minutes), path, frame_mb))
    tolerance = max(2.0, 0.05 * transcript_minutes)
    matching = sorted((c for c in candidates if c[0] <= tolerance), key=lambda c: c[0])

    info['video'] = None
    frame_mb = 1920 * 1080 * 3 / 2**20  # assume full HD while the video is not downloaded
    if matching:
        _, info['video'], frame_mb = matching[0]
        if len(matching) > 1:
            info['notes'] = 'several videos match the transcript length, used the closest'
    elif candidates:
        info['problems'].append(f'no downloaded video matches the transcript length ({transcript_minutes:.1f} min)')
    else:
        info['problems'].append('video not downloaded: ' + ' or '.join(f'{v}.mp4' for v in spec['videos']))

    # frames + frames resized to the slide size + image processor tensors (3x224x224 float32),
    # a few copies of every rendered slide, and ~1.5 GB for the two models
    info['ram_mb'] = info['frames'] * (frame_mb + page_mb + 0.6) + info['pages'] * 4 * page_mb + 1500
    return info


def run_matching(info, jump_penalty, file_base):
    """runs matching_algorithm.py for one lecture in a low-priority subprocess

    Returns:
        (int, float): exit code and runtime in seconds
    """
    file_base.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, 'matching_algorithm.py', '--merge_method', 'all', '--jump_penalty', str(jump_penalty),
               '--audio_script', str(info['srt']), '--file_path', str(info['pdf']),
               '--video_path', str(info['video']), '--file_name', str(file_base)]
    if sys.platform == 'win32':
        priority = {'creationflags': subprocess.BELOW_NORMAL_PRIORITY_CLASS}
    else:
        priority = {'preexec_fn': lambda: os.nice(10)}

    backup = TRACKED_SIDE_EFFECT.read_bytes() if TRACKED_SIDE_EFFECT.exists() else None
    start = time.time()
    try:
        with open(file_base.parent / 'run.log', 'w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=REPO / 'mavils', stdout=log, stderr=subprocess.STDOUT, **priority)
    finally:
        if backup is not None:
            TRACKED_SIDE_EFFECT.write_bytes(backup)
    return result.returncode, time.time() - start


def score(gt, result_path):
    """precision, recall and F1 like evaluation/evaluate_recall_precision.py: frames without a slide (-1) are ignored"""
    result = pd.read_excel(result_path)['Value']
    if len(result) != len(gt):
        raise ValueError(f'{result_path.name} has {len(result)} rows, ground truth has {len(gt)}')
    mask = gt != -1
    metric_args = dict(labels=gt.unique(), average='micro')
    return (precision_score(gt[mask], result[mask], **metric_args),
            recall_score(gt[mask], result[mask], **metric_args),
            f1_score(gt[mask], result[mask], **metric_args))


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ''


def update_csv(rows):
    """writes rows to results/baseline.csv, replacing earlier rows of the same lecture, method and jump penalty"""
    new = pd.DataFrame(rows)
    if CSV_PATH.exists():
        old = pd.read_csv(CSV_PATH)
        key = ['lecture', 'method', 'jump_penalty']
        replaced = old.set_index(key).index.isin(new.set_index(key).index)
        new = pd.concat([old[~replaced], new], ignore_index=True)
    new.sort_values(['lecture', 'method']).to_csv(CSV_PATH, index=False, float_format='%.4f')


def main():
    parser = argparse.ArgumentParser(description='Runs the MaViLS baseline on all lectures and logs the scores.')
    parser.add_argument('--lectures', nargs='+', choices=sorted(LECTURES), default=sorted(LECTURES),
                        help='lectures to process (default: all)')
    parser.add_argument('--jump_penalty', type=float, default=0.1)
    parser.add_argument('--list', action='store_true', help='only show inputs, RAM estimates and status')
    parser.add_argument('--force', action='store_true', help='run again even if result files exist')
    parser.add_argument('--no_memory_check', action='store_true', help='run even if the RAM estimate is too high')
    parser.add_argument('--memory_fraction', type=float, default=0.8,
                        help='run only if the RAM estimate is below this fraction of the available RAM')
    args = parser.parse_args()

    jp_file = str(args.jump_penalty).replace('.', 'comma')  # naming used by matching_algorithm.py
    authors = pd.read_excel(AUTHORS_F1).set_index('Lecture name')
    commit = git_commit()

    if args.list:
        print(f'{"lecture":32s} {"frames":>6s} {"pages":>5s} {"RAM GB":>6s}  {"done":4s}  video / problems')

    for name in args.lectures:
        spec = LECTURES[name]
        info = inspect_lecture(name, spec)
        file_base = RUNS_DIR / name / name
        result_files = {method: Path(f'{file_base}_{suffix.format(jp_file=jp_file)}.xlsx')
                        for method, (suffix, _) in METHODS.items()}
        done = all(path.exists() for path in result_files.values())

        if args.list:
            status = info['video'].name if info['video'] else ''
            status += ('  ' if status else '') + '; '.join(info['problems'])
            print(f'{name:32s} {info["frames"]:6d} {info["pages"]:5d} {info["ram_mb"] / 1024:6.1f}  '
                  f'{"yes" if done else "no":4s}  {status}')
            continue

        if not done or args.force:
            if info['problems']:
                print(f'[{name}] skipped: ' + '; '.join(info['problems']), flush=True)
                continue
            available = available_memory_mb()
            if not args.no_memory_check and available is not None and info['ram_mb'] > args.memory_fraction * available:
                print(f'[{name}] skipped: needs ~{info["ram_mb"] / 1024:.1f} GB RAM, '
                      f'{available / 1024:.1f} GB available (use a bigger machine or --no_memory_check)', flush=True)
                continue
            print(f'[{name}] running {info["frames"]} frames, ~{info["ram_mb"] / 1024:.1f} GB RAM, '
                  f'started {datetime.now():%H:%M}', flush=True)
            exit_code, seconds = run_matching(info, args.jump_penalty, file_base)
            if exit_code != 0:
                print(f'[{name}] FAILED with exit code {exit_code}, see {file_base.parent / "run.log"}', flush=True)
                continue
            (file_base.parent / 'runtime_seconds.txt').write_text(f'{seconds:.0f}\n')

        runtime_file = file_base.parent / 'runtime_seconds.txt'
        runtime_minutes = float(runtime_file.read_text()) / 60 if runtime_file.exists() else np.nan
        gt = info['gt']
        rows = []
        for method, (_, column) in METHODS.items():
            precision, recall, f1 = score(gt, result_files[method])
            column = column.format(jp=args.jump_penalty)
            authors_f1 = authors.at[spec['authors'], column] if column in authors.columns else np.nan
            rows.append({
                'lecture': name, 'method': method, 'jump_penalty': args.jump_penalty,
                'precision': precision, 'recall': recall, 'f1': f1,
                'authors_f1': authors_f1, 'f1_minus_authors': f1 - authors_f1,
                'frames': info['frames'], 'pdf_pages': info['pages'],
                # descriptors of the ground truth as in evaluation/analyse_videos.py
                'no_slide_ratio': np.count_nonzero(gt == -1) / np.count_nonzero(gt != -1),
                'jumpiness': np.count_nonzero(np.diff(gt)) / len(gt.unique()),
                'video': info['video'].name if info['video'] else '', 'runtime_minutes': runtime_minutes,
                'scored_at': f'{datetime.now():%Y-%m-%d %H:%M}', 'git_commit': commit, 'notes': info['notes'],
            })
        update_csv(rows)
        print(f'[{name}] F1 ' + '  '.join(f"{r['method']}={r['f1']:.3f}" for r in rows), flush=True)

    if not args.list and CSV_PATH.exists():
        table = pd.read_csv(CSV_PATH)
        table = table[table['jump_penalty'] == args.jump_penalty].pivot(index='lecture', columns='method', values='f1')
        table['best'] = table.max(axis=1)
        print(f'\nF1 per lecture (jump penalty {args.jump_penalty}), lowest best score first:')
        print(table.sort_values('best').round(3).to_string())


if __name__ == '__main__':
    main()
