"""Error triage for low-scoring lectures: what kind of mistakes does the baseline make?

For one method's result file per lecture, every wrong frame (ground truth != -1) is labelled as
  near_switch  the ground-truth slide changes right before or after it (timing of a genuine slide switch),
  neighbour    the predicted slide is adjacent to the true slide,
  far          any other slide.
The script also reports the longest runs of consecutive wrong frames, how often prediction and ground truth
change slides, and how much embedded text the PDF has (scanned slides depend on Tesseract alone).

It writes all wrong frames to results/baseline_runs/<lecture>/triage/<method>_errors.csv and, for manual
inspection, one side-by-side image (video frame | predicted slide | true slide) for each of the longest runs.

Example:
    python scripts/triage_errors.py --lectures image_processing climate_and_cities --method max --samples 6
"""
import argparse
import sys
from pathlib import Path

import cv2
import fitz
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from reproduce_baseline import LECTURES, METHODS, RUNS_DIR, inspect_lecture  # noqa: E402


def wrong_runs(wrong):
    """returns (start, end) index pairs (end inclusive) of consecutive wrong frames, longest first"""
    runs, start = [], None
    for i, is_wrong in enumerate(list(wrong) + [False]):
        if is_wrong and start is None:
            start = i
        elif not is_wrong and start is not None:
            runs.append((start, i - 1))
            start = None
    return sorted(runs, key=lambda r: r[1] - r[0], reverse=True)


def categorize(gt, pred):
    """labels every frame as correct, ignored (no slide in ground truth), near_switch, neighbour or far"""
    near_switch = np.zeros(len(gt), dtype=bool)
    for c in np.flatnonzero(np.diff(gt) != 0):  # the ground truth switches between frame c and c + 1
        near_switch[c:c + 2] = True
    labels = np.where(near_switch, 'near_switch', np.where(np.abs(pred - gt) == 1, 'neighbour', 'far'))
    labels = np.where(pred == gt, 'correct', labels)
    return np.where(gt == -1, 'ignored', labels)


def to_bgr(pixmap, height):
    img = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    img = cv2.cvtColor(img, {1: cv2.COLOR_GRAY2BGR, 3: cv2.COLOR_RGB2BGR, 4: cv2.COLOR_RGBA2BGR}[pixmap.n])
    return cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height))


def labelled(img, text):
    band = np.full((36, img.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(band, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return np.vstack([band, img])


def save_side_by_side(path, video, doc, seconds, predicted, true, height=360):
    """writes video frame | predicted slide | true slide next to each other"""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    frame = cv2.resize(frame, (int(frame.shape[1] * height / frame.shape[0]), height))
    minutes, secs = divmod(int(seconds), 60)
    panels = [labelled(frame, f'video {minutes}:{secs:02d}'),
              labelled(to_bgr(doc[predicted - 1].get_pixmap(), height), f'predicted slide {predicted}'),
              labelled(to_bgr(doc[true - 1].get_pixmap(), height), f'true slide {true}')]
    cv2.imwrite(str(path), np.hstack(panels))
    return True


def main():
    parser = argparse.ArgumentParser(description='Categorizes the baseline errors of lectures.')
    parser.add_argument('--lectures', nargs='+', choices=sorted(LECTURES), required=True)
    parser.add_argument('--method', choices=sorted(METHODS), default='max')
    parser.add_argument('--jump_penalty', type=float, default=0.1)
    parser.add_argument('--samples', type=int, default=6, help='number of longest wrong runs to save as images')
    args = parser.parse_args()
    jp_file = str(args.jump_penalty).replace('.', 'comma')

    for name in args.lectures:
        result_path = RUNS_DIR / name / f'{name}_{METHODS[args.method][0].format(jp_file=jp_file)}.xlsx'
        if not result_path.exists():
            print(f'[{name}] no result file {result_path.name}; run scripts/reproduce_baseline.py first')
            continue
        info = inspect_lecture(name, LECTURES[name])
        result = pd.read_excel(result_path)
        gt, pred = info['gt'].to_numpy(), result['Value'].to_numpy()
        labels = categorize(gt, pred)
        scored = labels != 'ignored'
        wrong = scored & (labels != 'correct')

        doc = fitz.open(info['pdf'])
        text_per_page = np.mean([len(page.get_text().strip()) for page in doc])
        counts = {c: int(np.sum(labels == c)) for c in ('near_switch', 'neighbour', 'far')}
        runs = wrong_runs(wrong)

        print(f'\n[{name}] method={args.method}  accuracy={1 - wrong.sum() / scored.sum():.3f}  '
              f'wrong {int(wrong.sum())} of {int(scored.sum())} scored frames '
              f'({int((~scored).sum())} no-slide frames ignored)')
        print('  errors: ' + ', '.join(f'{c} {n} ({n / max(wrong.sum(), 1):.0%})' for c, n in counts.items()))
        print(f'  slide changes: ground truth {int(np.count_nonzero(np.diff(gt)))}, '
              f'prediction {int(np.count_nonzero(np.diff(pred)))}')
        print(f'  embedded PDF text: {text_per_page:.0f} characters per page'
              + ('  (scanned slides: text comes from Tesseract only)' if text_per_page < 20 else ''))
        for start, end in runs[:args.samples]:
            t0, t1 = result['Key'].iloc[start], result['Key'].iloc[end]
            true_slides = sorted(set(gt[start:end + 1]) - {-1})
            predicted = sorted(set(pred[start:end + 1]))
            print(f'  wrong run of {end - start + 1:3d} frames  {t0 / 60:5.1f}-{t1 / 60:5.1f} min  '
                  f'true {true_slides}  predicted {predicted}')

        triage_dir = RUNS_DIR / name / 'triage'
        triage_dir.mkdir(exist_ok=True)
        errors = result.assign(true_slide=gt, category=labels)[wrong]
        errors.rename(columns={'Key': 'seconds', 'Value': 'predicted_slide'}).to_csv(
            triage_dir / f'{args.method}_errors.csv', index_label='frame')

        if info['video'] is None:
            print('  video not available, no images written')
            continue
        for rank, (start, end) in enumerate(runs[:args.samples], 1):
            i = (start + end) // 2
            image = triage_dir / f'{args.method}_run{rank}_frame{i}_pred{pred[i]}_true{gt[i]}.png'
            save_side_by_side(image, info['video'], doc, result['Key'].iloc[i], int(pred[i]), int(gt[i]))
        print(f'  images: {triage_dir}')
        doc.close()


if __name__ == '__main__':
    main()
