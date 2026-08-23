#!/usr/bin/env python3
"""
M2 - bước 1/2.

Nhiệm vụ : TSV (output M1) -> lhotse RecordingSet + SupervisionSet.
Vào      : transcripts_<tag>/{train,dev,test}.tsv
           cột: utt_id, speaker, audio_path, text
Ra       : data/manifests_<tag>/{split}_recordings.jsonl.gz
                               /{split}_supervisions.jsonl.gz

Khái niệm:
  Recording   = metadata FILE audio (id, path, sampling_rate, num_samples,
                duration, num_channels). Không chứa mẫu âm thanh.
  Supervision = metadata NHÃN (id, recording_id, start, duration, speaker,
                text). Ở dự án này một wav = một câu = một Supervision,
                start=0 và phủ trọn file.
"""

import argparse
import csv
from pathlib import Path

from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions

TARGET_SR = 16000  # Zipformer + Fbank đều giả định 16 kHz; đổi ở đây phải đổi cả compute_fbank.
REPO_ROOT = Path(__file__).resolve().parent.parent  # ~/project/vnasr (local/ -> repo)

def read_tsv(path: Path):
    """Đọc TSV có header -> list[dict]. newline='' để module csv tự xử lý xuống dòng."""
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def build(rows, audio_root: Path):
    """
    Nhiệm vụ: list[dict] TSV -> (RecordingSet, SupervisionSet).
    Vào     : rows = các dòng TSV; audio_root = gốc để nối audio_path tương đối.
    Ra      : hai manifest lhotse, đã chặn mọi bất thường bằng exception.
    """
    recordings, supervisions, seen = [], [], set()

    for r in rows:
        uid = r["utt_id"]

        # BẪY 1: utt_id trùng. RecordingSet là dict theo id -> bản sau GHI ĐÈ bản
        # trước, số câu tụt xuống mà không có exception nào. Chặn ngay tại đây.
        if uid in seen:
            raise ValueError(f"utt_id trùng: {uid}")
        seen.add(uid)

        path = (audio_root / r["audio_path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{uid}: không thấy {path}")

        # Recording.from_file(path, recording_id=...)
        #   - chỉ đọc HEADER (soundfile), không nạp mẫu -> rất nhanh.
        #   - duration = num_samples / sampling_rate, lấy từ header.
        #   - recording_id: bỏ trống thì lhotse lấy TÊN FILE làm id. Ta ép bằng
        #     utt_id để id xuyên suốt TSV -> manifest -> cut.
        # BẪY 2: header hỏng -> duration sai -> supervision vượt biên recording
        #     -> fix_manifests cắt ngắn âm thầm. Vì vậy assert sr/kênh ở đây.
        rec = Recording.from_file(path, recording_id=uid)
        if rec.sampling_rate != TARGET_SR:
            raise ValueError(f"{uid}: sampling_rate={rec.sampling_rate}, cần {TARGET_SR}")
        if rec.num_channels != 1:
            raise ValueError(f"{uid}: num_channels={rec.num_channels}, cần mono")
        if rec.duration <= 0:
            raise ValueError(f"{uid}: duration={rec.duration}")

        text = r["text"].strip()
        if not text:
            raise ValueError(f"{uid}: text rỗng")

        recordings.append(rec)
        supervisions.append(
            SupervisionSegment(
                id=uid,                 # id của NHÃN
                recording_id=uid,       # khoá ngoại trỏ về Recording; sai = nhãn mồ côi
                start=0.0,              # tính từ đầu file (giây)
                duration=rec.duration,  # phủ trọn file: mỗi wav đúng một câu
                channel=0,              # mono -> kênh 0
                language="Vietnamese",
                speaker=r["speaker"],   # để thống kê/phân tích theo người nói
                text=text,              # chuỗi thô; M3 mới tokenize bằng BPE
            )
        )

    return (
        RecordingSet.from_recordings(recordings),
        SupervisionSet.from_segments(supervisions),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript-dir", type=Path, required=True,
                    help="thư mục chứa train/dev/test.tsv")
    ap.add_argument("--audio-root", type=Path, default=REPO_ROOT,
                    help="gốc để nối với cột audio_path nếu path tương đối")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "dev", "test"]:
        rows = read_tsv(args.transcript_dir / f"{split}.tsv")
        recordings, supervisions = build(rows, args.audio_root)

        # fix_manifests(recordings, supervisions) -> (recordings, supervisions)
        #   - bỏ supervision mồ côi (recording_id không tồn tại)
        #   - kẹp supervision vượt biên về đúng [0, duration]
        # icefall để nó chạy im lặng. Ở đây: nếu nó PHẢI sửa gì tức là M1 sai,
        # nên cho fail thay vì nuốt lỗi.
        
        n_before = len(supervisions)
        recordings, supervisions = fix_manifests(recordings, supervisions)
        if len(supervisions) != n_before:
            raise RuntimeError(
                f"{split}: fix_manifests bỏ {n_before - len(supervisions)} supervision"
            )

        # Kiểm tra sâu: id trùng, duration âm, sampling_rate không đồng nhất, text lạ.
        validate_recordings_and_supervisions(recordings, supervisions)

        recordings.to_file(args.out_dir / f"{split}_recordings.jsonl.gz")
        supervisions.to_file(args.out_dir / f"{split}_supervisions.jsonl.gz")

        total = sum(r.duration for r in recordings)
        print(f"{split}: {len(recordings)} rec, {total/3600:.3f} h, "
              f"min={min(r.duration for r in recordings):.2f}s "
              f"max={max(r.duration for r in recordings):.2f}s")


if __name__ == "__main__":
    main()