#!/usr/bin/env python3
"""
M0.5 - chuẩn hoá audio. PHẢI chạy trước M1 (prepare_split.py).

Nhiệm vụ : đưa mọi bản thu về 16 kHz mono PCM_16, giữ nguyên cây thư mục,
           và copy kèm mọi file text (script.txt / transcript.txt / ...).
Vào      : data_raw/<Speaker>/<session>/*.wav + *.txt   (44.1k hoặc 48k, mono/stereo)
Ra       : dataset/<Speaker>/<session>/*.wav + *.txt     (16 kHz mono)

Vì sao đích là 16 kHz chứ không phải 48 kHz: Fbank cấu hình sampling_rate=16000
và high_freq=-400 (cắt tại 7600 Hz). Model không bao giờ nhìn thấy nội dung trên
8 kHz, nên giữ 48k chỉ tốn đĩa và thêm một lần nội suy vô ích.

Vì sao ghi ra `dataset/` chứ không sửa tại chỗ: M1/M2 đọc `dataset/` như cũ, không
phải sửa dòng code nào. `data_raw/` giữ nguyên làm bằng chứng gốc, CHỈ ĐỌC.
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

TARGET_SR = 16000
PEAK_CEIL = 0.999   # trần an toàn trước khi lượng tử hoá về PCM_16


def convert_wav(src: Path, dst: Path):
    """
    Vào: đường dẫn wav nguồn và đích.
    Ra : (sr_nguồn, số_kênh_nguồn, đỉnh_sau_resample, có_phải_hạ_biên_độ_không)
    """
    # always_2d: xử lý mono/stereo bằng cùng một nhánh code.
    # dtype="float32": soundfile đã chia sẵn về thang [-1, 1] - đúng thang lhotse cần.
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    n_ch = x.shape[1]
    x = x.mean(axis=1)  # stereo -> mono

    if sr != TARGET_SR:
        # 44100 -> 16000 là tỉ lệ không nguyên (147:160): bắt buộc dùng resampler
        # có lọc chống chồng phổ. quality="VHQ" là mức cao nhất của soxr.
        x = soxr.resample(x, sr, TARGET_SR, quality="VHQ")

    # BẪY: resample gây OVERSHOOT. Đo thực tế: đỉnh 0,990 ở 44.1k -> 1,184 sau khi
    # hạ về 16k (ringing của bộ lọc). Ghi thẳng ra PCM_16 sẽ CLIP - hỏng vĩnh viễn,
    # không sửa được bằng cách giảm âm lượng sau. Phải đo đỉnh rồi hạ nếu vượt trần.
    peak = float(np.abs(x).max())
    lowered = peak > PEAK_CEIL
    if lowered:
        x = x * (PEAK_CEIL / peak)

    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, x, TARGET_SR, subtype="PCM_16")
    return sr, n_ch, peak, lowered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data_raw"), help="thư mục gốc, chỉ đọc")
    ap.add_argument("--dst", type=Path, default=Path("dataset"), help="thư mục đã chuẩn hoá")
    ap.add_argument("--force", action="store_true",
                    help="cho phép ghi đè khi --dst đã có dữ liệu")
    args = ap.parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"Không thấy {args.src}. Đã đổi tên dataset/ -> data_raw/ chưa?")

    # BẪY: chạy lần hai sau khi đã đổi cấu trúc -> file cũ và mới trộn lẫn trong
    # cùng thư mục, M1 đếm được nhiều wav hơn số dòng script và assert nổ ở chỗ
    # không liên quan. Chặn ngay, bắt người dùng xoá tay hoặc nói rõ --force.
    if args.dst.exists() and any(args.dst.rglob("*")) and not args.force:
        raise SystemExit(f"{args.dst} đã có dữ liệu. Xoá nó hoặc chạy lại với --force.")

    src_stats, n_lowered, n_wav, n_txt = Counter(), 0, 0, 0
    worst = (0.0, None)

    for path in sorted(args.src.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(args.src)   # giữ nguyên cây <Speaker>/<session>/
        out = args.dst / rel

        if path.suffix.lower() == ".wav":
            sr, ch, peak, lowered = convert_wav(path, out)
            src_stats[(sr, ch)] += 1
            n_wav += 1
            if peak > worst[0]:
                worst = (peak, rel)
            if lowered:
                n_lowered += 1
                print(f"!! overshoot {peak:.3f} -> đã hạ: {rel}")
        elif path.suffix.lower() == ".txt":
            # Tên script không thống nhất (script.txt, transcript.txt, asr_script.txt).
            # Copy MỌI .txt để M1 tự tìm, thay vì đoán tên.
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)   # copy2 giữ mtime -> dễ đối chiếu về sau
            n_txt += 1
        else:
            print(f"   bỏ qua (không phải wav/txt): {rel}")

    print()
    for (sr, ch), n in sorted(src_stats.items()):
        print(f"nguồn {sr} Hz, {ch} kênh: {n} file")
    print(f"-> {n_wav} wav @ 16 kHz mono, {n_txt} file text đã copy")
    print(f"đỉnh cao nhất sau resample: {worst[0]:.3f} ({worst[1]})")
    print(f"số file phải hạ biên độ: {n_lowered}")

    # Self-check: mỗi thư mục con phải giữ nguyên số wav so với nguồn.
    for d in sorted(p for p in args.src.rglob("*") if p.is_dir()):
        rel = d.relative_to(args.src)
        a = len(list(d.glob("*.wav")))
        b = len(list((args.dst / rel).glob("*.wav")))
        if a != b:
            raise SystemExit(f"{rel}: nguồn {a} wav nhưng đích {b} wav")
    print("self-check: số wav mỗi thư mục khớp nguồn - OK")


if __name__ == "__main__":
    main()