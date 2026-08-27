"""
M2 - bước 2/2.

Nhiệm vụ : manifests -> CutSet có feature Fbank 80 chiều đã lưu ra đĩa.
Vào      : data/manifests_<tag>/{split}_{recordings,supervisions}.jsonl.gz
Ra       : fbank_<tag>/{split}_cuts.jsonl.gz  (cut + con trỏ tới feature)
           fbank_<tag>/{split}_feats.lca      (mảng feature nén lilcom)

Cut = Recording + Supervision + con trỏ feature. Từ M4 trở đi model CHỈ đọc Cut,
nên mọi sai lệch trước đó bị đóng băng vào file này.

Phụ thuộc: cần `pip install lilcom` (lhotse KHÔNG kéo theo gói này).
"""

import argparse
from pathlib import Path

import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter, load_manifest

# Mỗi worker của compute_and_store_features là một process riêng. Nếu mỗi process
# lại mở nhiều thread BLAS thì chúng tranh CPU với nhau và chậm hơn 1 job.
torch.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num-mel-bins", type=int, default=80,
                    help="số bộ lọc mel = số chiều feature; Zipformer cấu hình cho 80")
    ap.add_argument("--num-jobs", type=int, default=2,
                    help="số process trích đặc trưng; máy 16GB/4 core nên để 2")
    ap.add_argument("--min-duration", type=float, default=0.3,
                    help="câu ngắn hơn -> ít frame hơn số token -> pruned RNN-T lỗi")
    ap.add_argument("--max-duration", type=float, default=20.0,
                    help="câu dài hơn -> một câu chiếm trọn batch, dễ OOM")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Fbank(FbankConfig(...)) là extractor kaldi-style thuần torch của lhotse
    # (lhotse.features.kaldi.extractors.Fbank), KHÔNG phải
    # torchaudio.compliance.kaldi.fbank. Mặc định quan trọng:
    #   frame_length=0.025s, frame_shift=0.01s  -> ~100 frame/giây
    #   snip_edges=False   -> num_frames = round(duration / 0.01)
    #   window=povey, preemph=0.97, remove_dc_offset=True, dither=0.0
    #   low_freq=20.0, high_freq=-400.0  (âm = nyquist-400 = 7600 Hz)
    #   use_energy=False   -> 80 chiều đều là log-mel, không có kênh năng lượng
    # BẪY 3: `num_mel_bins` chỉ là alias, lhotse gán vào `num_filters` và để
    # `num_mel_bins=None`. In `num_filters` mới là con số thật.
    extractor = Fbank(FbankConfig(sampling_rate=16000, num_mel_bins=args.num_mel_bins))
    cfg = extractor.config
    print(f"Fbank cfg: filters={cfg.num_filters} frame_len={cfg.frame_length}s "
          f"shift={cfg.frame_shift}s snip_edges={cfg.snip_edges} "
          f"low={cfg.low_freq} high={cfg.high_freq} dither={cfg.dither}")

    dropped_total = 0
    for split in ["train", "dev", "test"]:
        # load_manifest: đọc eager (1055 câu, không đáng lazy). Lưu ý
        # CutSet.from_manifests mặc định lazy=False nên dù có nạp lazy thì nó
        # cũng materialize lại -> dùng load_manifest cho thẳng thắn.
        recordings = load_manifest(args.manifest_dir / f"{split}_recordings.jsonl.gz")
        supervisions = load_manifest(args.manifest_dir / f"{split}_supervisions.jsonl.gz")

        # from_manifests ghép theo recording_id và sinh MonoCut phủ trọn recording.
        # BẪY 4: cut.id KHÔNG bằng utt_id - lhotse thêm hậu tố kênh:
        #   utt_id "Dung_clean_0001" -> cut.id "Dung_clean_0001-0"
        # supervision.id thì giữ nguyên. Muốn map ngược về TSV phải dùng
        # cut.supervisions[0].id, không dùng cut.id.
        cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)

        n0 = len(cuts)
        keep = lambda c: args.min_duration <= c.duration <= args.max_duration
        # BẪY 5: filter XOÁ utterance. Nếu M3 train BPE từ TSV gốc trong khi cut
        # đã bị lọc bớt thì corpus text != corpus audio. Phải in ID cụ thể ra.
        dropped = [c.supervisions[0].id for c in cuts if not keep(c)]
        cuts = cuts.filter(keep).to_eager()  # to_eager: cần len()/split() ở dưới
        if dropped:
            dropped_total += len(dropped)
            print(f"!! {split}: loại {len(dropped)}/{n0} cut ngoài "
                  f"[{args.min_duration},{args.max_duration}]s -> {dropped[:5]}")

        # compute_and_store_features:
        #   extractor    - đối tượng Fbank ở trên
        #   storage_path - tiền tố file lưu; LilcomChunkyWriter tạo <path>.lca
        #   num_jobs     - lhotse .split(num_jobs) rồi chạy ProcessPoolExecutor
        #   storage_type - LilcomChunkyWriter: nén CÓ MẤT MÁT (~0,016 sai số
        #                  tuyệt đối trên log-mel). Đây là chuẩn của icefall;
        #                  nhớ con số này khi so sánh feature ở bước parity.
        # Hàm trả về CutSet MỚI (đã gắn con trỏ feature) - phải gán lại.
        cuts = cuts.compute_and_store_features(
            extractor=extractor,
            storage_path=args.out_dir / f"{split}_feats",
            num_jobs=args.num_jobs,
            storage_type=LilcomChunkyWriter,
        )
        cuts.to_file(args.out_dir / f"{split}_cuts.jsonl.gz")

        # Self-check: snip_edges=False => num_frames ~ round(duration/frame_shift).
        # Lệch quá 1 frame nghĩa là config Fbank không phải cái ta nghĩ.
        for c in cuts:
            expect = round(c.duration / cfg.frame_shift)
            if abs(c.num_frames - expect) > 1:
                raise RuntimeError(f"{c.id}: num_frames={c.num_frames}, kỳ vọng ~{expect}")
            if c.num_features != args.num_mel_bins:
                raise RuntimeError(f"{c.id}: num_features={c.num_features}")

        c = next(iter(cuts))  # KHÔNG dùng cuts[0]: chỉ eager CutSet mới index được
        print(f"{split}: {len(cuts)} cut | ví dụ cut_id={c.id} "
              f"sup_id={c.supervisions[0].id} dur={c.duration:.2f}s "
              f"frames={c.num_frames} dim={c.num_features}")

    if dropped_total:
        print(f"!! TỔNG {dropped_total} cut bị loại - M3 phải loại đúng các id này "
              f"khỏi text corpus trước khi train BPE")


if __name__ == "__main__":
    main()