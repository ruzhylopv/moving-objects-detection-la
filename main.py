import cv2 as cv
import numpy as np

# ============================================================
# INPUT FILE PATHS
# ============================================================
GROUND_TRUTH_VIDEO = "322_gt.mp4"
RESULT_VIDEO = "321_result.mp4"
OUTPUT_VIDEO = "prayer_overlay.mp4"

THRESHOLD = 127
SLOW_FACTOR = 3   # result video becomes 3x slower

# ============================================================
# READ VIDEO AS GRAYSCALE FRAMES
# ============================================================
def read_video_gray(path):
    cap = cv.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {path}")

    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame.ndim == 3:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            gray = frame

        frames.append(gray)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames in {path}")

    return np.array(frames), fps, width, height, frame_count


gt_frames, gt_fps, gt_w, gt_h, gt_count = read_video_gray(GROUND_TRUTH_VIDEO)
res_frames, res_fps, res_w, res_h, res_count = read_video_gray(RESULT_VIDEO)


res_frames = np.repeat(res_frames, SLOW_FACTOR, axis=0)
res_count = len(res_frames)

# ============================================================
# ALIGN TO GT
# ============================================================
n = gt_count
h = min(gt_h, res_h)
w = min(gt_w, res_w)

gt_frames = gt_frames[:n, :h, :w]

if res_count < n:
    padded = np.zeros((n, h, w), dtype=np.uint8)
    padded[:res_count] = res_frames[:, :h, :w]
    res_frames = padded
else:
    res_frames = res_frames[:n, :h, :w]

fps = gt_fps if gt_fps > 0 else 30.0

fourcc = cv.VideoWriter_fourcc(*"mp4v")
writer = cv.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

TP = 0
FP = 0
FN = 0 
TN = 0

for i in range(n):
    gt = gt_frames[i] > THRESHOLD
    rs = res_frames[i] > THRESHOLD

    overlap = gt & rs
    gt_only = gt & (~rs)
    rs_only = (~gt) & rs
    bg = (~gt) & (~rs)

    TP += np.count_nonzero(overlap)
    FN += np.count_nonzero(gt_only)
    FP += np.count_nonzero(rs_only)
    TN += np.count_nonzero(bg)

    frame = np.zeros((h, w, 3), dtype=np.uint8)

    frame[gt_only] = (255, 255, 255) #gt

    frame[rs_only] = (0, 255, 255) #result

    frame[overlap] = (0, 255, 0) 

    writer.write(frame)

writer.release()


precision = TP / (TP + FP) if TP + FP else 0.0

recall = TP / (TP + FN) if TP + FN else 0.0

f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

iou = TP / (TP + FP + FN) if TP + FP + FN else 0.0


print("================================")
print("COMPARING RESULTS:")
print("Precision:", round(precision, 6))
print("Recall:", round(recall, 6))
print("F1 Score:", round(f1, 6))
print("IoU:", round(iou, 6))
print("Saved video:", OUTPUT_VIDEO)
print("================================")