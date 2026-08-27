#!/usr/bin/env python3
"""
M2 - self-check trọng tâm: XÁC ĐỊNH quy ước biên độ mà lhotse Fbank dùng.

Nhiệm vụ : lấy 1 cut đã có feature, tính lại fbank từ waveform ở hai thang
           ([-1,1] và int16 = x32768), xem thang nào khớp feature đã lưu.
Vào      : fbank_<tag>/dev_cuts.jsonl.gz
Ra       : in offset và kết luận; SystemExit nếu không khớp thang nào.

Vì sao cần: log-mel của tín hiệu nhân k sẽ lệch đúng 2*ln(k) trên MỌI hệ số
(log của phổ công suất ~ |X|^2). Với k=32768 thì offset = 20,794. Lúc train,
lhotse tự chuẩn hoá nên train/feature luôn nhất quán; nhưng đường mic (pyaudio
trả int16 thô) mà quên chia 32768 thì encoder nhận input lệch 20,8 -> triệu
chứng ở M8 là "chạy file wav thì tốt, chạy mic ra rác".

Nguyên tắc: DÙNG CHÍNH extractor của lhotse cho cả hai thang. So với
torchaudio.compliance.kaldi.fbank sẽ SAI - hai bên khác high_freq (lhotse cắt
tại 7600 Hz, torchaudio tới nyquist) nên chênh lệch không phải hằng số.
"""

import argparse

import numpy as np
import torch
from lhotse import CutSet, Fbank, FbankConfig

# LilcomChunkyWriter nén CÓ MẤT MÁT: sai số tuyệt đối đo được ~0,016 trên log-mel.
# Ngưỡng 1e-3 sẽ báo động giả. 0,05 đủ chặt để vẫn phân biệt được offset 20,79.
LILCOM_TOL = 0.05
OFFSET_I16 = 2 * np.log(32768)  # 20.7944

# Lhotse kẹp năng lượng mel dưới sàn float32-eps trước khi lấy log:
#   log_mel = ln(max(E, eps)),  ln(1.1921e-07) = -15.9424
# Bin nào chạm sàn ở thang [-1,1] thì phép nhân 32768 chỉ nhấc nó lên khỏi sàn
# chứ không cộng đủ 2*ln(32768) -> quan hệ "cộng hằng số" chỉ đúng trên các bin
# KHÔNG chạm sàn. Khung im lặng và bin tần số cao trong file thật hay chạm sàn,
# nên phải lọc chúng ra trước khi so, nếu không mean gap sẽ tụt xuống dưới 20,79.
LOG_FLOOR = float(np.log(np.finfo(np.float32).eps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", default="data/fbank_matched/dev_cuts.jsonl.gz")
    ap.add_argument("--num-mel-bins", type=int, default=80)
    args = ap.parse_args()

    cut = next(iter(CutSet.from_file(args.cuts)))

    # load_features(): đọc feature ĐÃ LƯU từ .lca -> ndarray (T, 80).
    ref = torch.from_numpy(cut.load_features())
    # load_audio(): đọc waveform qua lhotse -> ndarray (channels, samples), float [-1,1].
    # Dùng cái này thay torchaudio.load: torchaudio >= 2.9 bỏ backend cũ và đòi
    # torchcodec, còn lhotse thì luôn trả đúng waveform mà extractor đã thấy.
    wav = torch.from_numpy(cut.load_audio()).squeeze(0)
    print(f"cut={cut.id} feats={tuple(ref.shape)} "
          f"wav n={wav.numel()} range=[{wav.min():.3f},{wav.max():.3f}]")

    extractor = Fbank(FbankConfig(sampling_rate=cut.sampling_rate,
                                  num_mel_bins=args.num_mel_bins))
    # extract(samples, sampling_rate) nhận Tensor 1-D hoặc ndarray, trả (T, 80).
    f_unit = torch.as_tensor(np.asarray(extractor.extract(wav, cut.sampling_rate)))
    f_i16 = torch.as_tensor(np.asarray(extractor.extract(wav * 32768.0, cut.sampling_rate)))

    d_unit = (ref - f_unit).mean().item()
    d_i16 = (ref - f_i16).mean().item()
    print(f"ref - fbank([-1,1]) = {d_unit:+.4f}  (max|d| = {(ref - f_unit).abs().max():.4f})")
    print(f"ref - fbank(x32768) = {d_i16:+.4f}")
    print(f"2*ln(32768)         = {OFFSET_I16:.4f}")

    if abs(d_unit) < LILCOM_TOL:
        print("=> lhotse dùng thang [-1,1]. Mic int16 PHẢI chia 32768 trước khi tính fbank.")
    elif abs(d_i16) < LILCOM_TOL:
        print("=> lhotse dùng thang int16. Mic KHÔNG chia 32768.")
    else:
        raise SystemExit(f"Không khớp thang nào (d_unit={d_unit:.4f}, d_i16={d_i16:.4f}) "
                         f"-> config Fbank khác giả định, dừng lại")

    # Kiểm tra chéo: trên các bin KHÔNG chạm sàn, hai thang phải lệch đúng
    # 2*ln(32768). So bằng trung bình toàn bộ sẽ ra số nhỏ hơn (xem LOG_FLOOR).
    alive = f_unit > LOG_FLOOR + 1e-3
    pct = 100.0 * alive.float().mean().item()
    gap = (f_i16 - f_unit)[alive]
    print(f"bin không chạm sàn: {pct:.1f}%  |  gap mean = {gap.mean():.4f} "
          f"std = {gap.std():.6f}")
    if abs(gap.mean().item() - OFFSET_I16) > 1e-3:
        raise SystemExit(f"gap = {gap.mean():.4f}, không bằng 2*ln(32768) "
                         f"-> hiểu sai bản chất feature, không phải sai thang")
    print(f"OK: gap = {gap.mean():.4f} = 2*ln(32768)")


if __name__ == "__main__":
    main()