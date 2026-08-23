# local/prepare_splits.py — dataset/<Speaker>/<session>/ -> transcripts_matched_u20/{train,dev,test}.tsv
# Fidelity-first: train==dev==test (học vẹt). Ghép audio<->text HOÀN TOÀN theo thứ tự sort,
# transcript không có id -> sort sai = lệch nhãn toàn bộ mà WER không hề kêu.
import argparse, csv, re, wave, contextlib
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent  # ~/project/vnasr (local/ -> repo)
print(f"repo root: {REPO_ROOT}")

def dur(p):
    """Độ dài audio tính bằng giây.
    Vào : p (Path) — file .wav PCM.
    Ra  : float giây. (Nếu dùng .m4a/.flac thì đổi sang torchaudio.info.)"""
    with contextlib.closing(wave.open(str(p))) as w:
        return w.getnframes() / w.getframerate()

def recorder_key(name):
    """Khoá sort đúng cho tên đánh số kiểu 'Recording (N)'.
    Vào : name (str) — tên file.
    Ra  : tuple (số_trong_ngoặc, name). File thiếu '(number)' coi là bản 1.
    Vì sao: sort theo CHUỖI làm '(10)' nhảy trước '(2)'. Sort theo SỐ mới đúng thứ tự thu."""
    m = re.search(r"\((\d+)\)", name)
    return (int(m.group(1)) if m else 1, name)

def find_units(data):
    """Tìm mọi 'unit' — folder lá chứa script.txt (tự phủ cả clean/ và ngan/).
    Vào : data (str) — gốc dataset/.
    Ra  : list[Path] các folder có script.txt, đã sort ổn định."""
    return sorted(p.parent for p in Path(data).rglob("script.txt"))

def load_unit(unit_dir, data_root, mode):
    """Ghép wav<->dòng script trong MỘT unit, theo vị trí sau khi sort.
    Vào : unit_dir (Path) — folder lá; data_root (Path) — gốc dataset/;
          mode (str) — 'naive' (sort chuỗi, để lộ bug) hoặc 'recorder' (sort số, đúng).
    Ra  : (speaker, tag, pairs)
          speaker=str (vd 'Dung'); tag=str id-hoá đường dẫn (vd 'Dung_ngan');
          pairs=list[(Path_wav, text)].
    Chốt chặn: assert số wav == số dòng script; lệch là dừng ngay, không ghép mù."""
    wavs = list(unit_dir.glob("*.wav"))
    key = (lambda p: p.name) if mode == "naive" else (lambda p: recorder_key(p.name))
    wavs = sorted(wavs, key=key)
    texts = [l.strip() for l in (unit_dir / "script.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    rel = unit_dir.relative_to(data_root)
    assert len(wavs) == len(texts), f"{rel}: {len(wavs)} wav vs {len(texts)} dòng"
    speaker = rel.parts[0]
    tag = "_".join(rel.parts)
    return speaker, tag, list(zip(wavs, texts))

def main():
    """Duyệt mọi unit -> gom thành rows (utt_id, speaker, audio_path, text) -> ghi 3 file TSV
    giống hệt nhau (matched split), đồng thời in bảng corr(dur, word_count) theo unit.
    corr là self-check alignment: ghép đúng thì DƯƠNG rõ; lệch thì tụt về 0."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="transcripts_matched_u20")
    ap.add_argument("--sort", choices=["naive", "recorder"], default="recorder")
    ap.add_argument("--path-root", type=Path, default=REPO_ROOT,
                help="gốc để hạ audio_path về tương đối; phải trùng --audio-root ở M2")
    args = ap.parse_args()

    rows, report = [], []
    for unit in find_units(args.data):
        speaker, tag, pairs = load_unit(unit, Path(args.data), args.sort)
        d = np.array([dur(w) for w, _ in pairs])
        wc = np.array([len(t.split()) for _, t in pairs])
        corr = float(np.corrcoef(d, wc)[0, 1]) if len(pairs) > 2 else float("nan")

        # Gom một dòng report cho unit này:
        #   tag        : tên unit (vd 'Dung_ngan') — id-hoá từ đường dẫn
        #   len(pairs) : n — số cặp (wav, text) trong unit
        #   corr       : tương quan giữa duration và word_count.
        #                Ghép đúng -> DƯƠNG rõ; lệch nhãn -> tụt về 0.
        #   d.std()    : độ lệch chuẩn của duration (giây) — thời lượng dao động rộng hay hẹp
        #   wc.std()   : độ lệch chuẩn của word_count (chữ) — số chữ dao động rộng hay hẹp.
        #                CẢNH BÁO: std_wc ~ 0 thì corr MẤT Ý NGHĨA (mẫu số ~ 0 -> ra nhiễu),
        #                lúc đó corr thấp KHÔNG phải bằng chứng lệch nhãn. Phải kiểm tra thủ công.
        report.append((tag, len(pairs), round(corr, 3), round(float(d.std()), 2), round(float(wc.std()), 2)))
        path_root = args.path_root.resolve()
        for i, (w, t) in enumerate(pairs):
            # BẪY: rglob trả path tuyệt đối. Ghi thẳng vào TSV -> lộ path máy cá nhân
            # và chết trên Kaggle. Hạ về tương đối so với path_root; relative_to()
            # ném ValueError nếu wav nằm ngoài root -> fail, không ghi lặng lẽ.
            rel = w.resolve().relative_to(path_root)
            rows.append((f"{tag}_{i:04d}", speaker, rel.as_posix(), t))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    FIELDS = ["utt_id", "speaker", "audio_path", "text"]
    for split in ["train", "dev", "test"]:
        with open(out / f"{split}.tsv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(FIELDS)
            w.writerows(rows)

    print(f"sort={args.sort}  tổng {len(rows)} dòng")
    print(f"{'unit':14} {'n':>4} {'corr':>7} {'std_dur':>8} {'std_wc':>7}")
    for tag, n, c, sd, sw in report:
        print(f"{tag:14} {n:>4} {c:>7} {sd:>8} {sw:>7}")

if __name__ == "__main__":
    main()