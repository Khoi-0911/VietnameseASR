#!/usr/bin/env python3
"""
task : M4 - giải phẫu mô hình. Dựng các khối của Zipformer transducer, kiểm tra
       hình dạng tensor và chạy thử pruned RNN-T loss trên dữ liệu giả.
       KHÔNG train, KHÔNG đọc fbank thật.
Vào  : data/lang_bpe_100/tokens.txt   (lấy vocab_size, xác nhận <blk> ở id 0)
       thư mục zipformer của icefall   (truyền qua --zipformer-dir)
Ra   : in ra stdout; sai bất kỳ chỗ nào là AssertionError

Chạy ở đâu: KAGGLE. scaling.py của icefall `import k2` ở cấp module và
SwooshL/SwooshR gọi thẳng k2.swoosh_l trong forward, nên máy local không có k2
chết ngay ở dòng import.

    python smoke/check_model.py \
        --zipformer-dir /kaggle/working/icefall/egs/librispeech/ASR/zipformer \
        --tokens data/lang_bpe_100/tokens.txt

BẪY: icefall dùng import phẳng (`from scaling import ...`, không phải
`from .scaling import ...`), nên thư mục zipformer PHẢI nằm trong sys.path.
Vì vậy argparse chạy TRƯỚC các lệnh import ở dưới - đây là cố ý, không phải
lỗi trình bày.
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--zipformer-dir",
        required=True,
        help="thư mục zipformer của icefall, ví dụ "
             "/kaggle/working/icefall/egs/librispeech/ASR/zipformer",
    )
    p.add_argument("--tokens", default="data/lang_bpe_100/tokens.txt")
    p.add_argument("--prune-range", type=int, default=5)
    return p.parse_args()


args = parse_args()

zf = Path(args.zipformer_dir).resolve()
if not (zf / "zipformer.py").is_file():
    raise SystemExit(f"Không thấy zipformer.py trong {zf}")
sys.path.insert(0, str(zf))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import k2  # noqa: E402
from decoder import Decoder  # noqa: E402
from joiner import Joiner  # noqa: E402
from subsampling import Conv2dSubsampling  # noqa: E402
from zipformer import Zipformer2  # noqa: E402

# Cấu hình "small streaming" của icefall.
# 74 state là dấu vân tay của num_encoder_layers = 2,2,2,2,2,2
# (6 state mỗi layer x 12 layer + 2). Sửa dòng này là số state đổi theo,
# và mọi checkpoint cũ hết dùng được.
CFG = dict(
    num_encoder_layers="2,2,2,2,2,2",
    downsampling_factor="1,2,4,8,4,2",
    encoder_dim="192,256,256,256,256,256",
    encoder_unmasked_dim="192,192,192,192,192,192",
    feedforward_dim="512,768,768,768,768,768",
    num_heads="4,4,4,8,4,4",
    query_head_dim="32,32,32,32,32,32",
    value_head_dim="12,12,12,12,12,12",
    pos_head_dim="4,4,4,4,4,4",
    cnn_module_kernel="31,31,15,15,15,31",
)
DECODER_DIM = 512
JOINER_DIM = 512
CONTEXT_SIZE = 2          # bigram: decoder chỉ nhìn 2 token gần nhất
BLANK_ID = 0              # phải trùng id của <blk> trong tokens.txt
EXPECTED_STATES = 74


def t(s):
    return tuple(int(x) for x in s.split(","))


def read_vocab(tokens_path: Path) -> int:
    """Đọc tokens.txt của M3. Vocab của model PHẢI bằng vocab của tokenizer."""
    pieces = {}
    for line in tokens_path.read_text(encoding="utf-8").splitlines():
        sym, i = line.rsplit(" ", 1)
        pieces[int(i)] = sym
    assert pieces[0] == "<blk>", f"id 0 phải là <blk>, đang là {pieces[0]!r}"
    assert pieces[1] == "<sos/eos>", pieces[1]
    assert pieces[2] == "<unk>", pieces[2]
    return len(pieces)


def build(vocab_size: int):
    """Sáu module, không phải bốn: hai lớp chiếu 'simple' nằm trong model.py."""
    enc = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=t(CFG["downsampling_factor"]),
        num_encoder_layers=t(CFG["num_encoder_layers"]),
        encoder_dim=t(CFG["encoder_dim"]),
        encoder_unmasked_dim=t(CFG["encoder_unmasked_dim"]),
        query_head_dim=t(CFG["query_head_dim"]),
        pos_head_dim=t(CFG["pos_head_dim"]),
        value_head_dim=t(CFG["value_head_dim"]),
        pos_dim=48,
        num_heads=t(CFG["num_heads"]),
        feedforward_dim=t(CFG["feedforward_dim"]),
        cnn_module_kernel=t(CFG["cnn_module_kernel"]),
        causal=True,                      # streaming, bắt buộc cho M8
        chunk_size=t("16,32,64,-1"),
        left_context_frames=t("64,128,256,-1"),
    )
    edim = max(enc.encoder_dim)
    embed = Conv2dSubsampling(in_channels=80, out_channels=edim, dropout=0.1)
    dec = Decoder(vocab_size=vocab_size, decoder_dim=DECODER_DIM,
                  blank_id=BLANK_ID, context_size=CONTEXT_SIZE)
    joi = Joiner(encoder_dim=edim, decoder_dim=DECODER_DIM,
                 joiner_dim=JOINER_DIM, vocab_size=vocab_size)
    simple_am_proj = nn.Linear(edim, vocab_size)
    simple_lm_proj = nn.Linear(DECODER_DIM, vocab_size)
    return embed, enc, dec, joi, simple_am_proj, simple_lm_proj, edim


def main():
    V = read_vocab(Path(args.tokens))
    print(f"[tokenizer] vocab_size = {V}, blank_id = {BLANK_ID}")

    embed, enc, dec, joi, am_proj, lm_proj, edim = build(V)
    mods = {"embed": embed, "encoder": enc, "decoder": dec, "joiner": joi,
            "simple_am_proj": am_proj, "simple_lm_proj": lm_proj}
    for m in mods.values():
        m.eval()
    tot = sum(q.numel() for m in mods.values() for q in m.parameters())
    print("[tham số] " + " | ".join(
        f"{k} {sum(q.numel() for q in m.parameters()) / 1e6:.3f}M"
        for k, m in mods.items()) + f"  =>  TỔNG {tot / 1e6:.2f}M")

    # --- 1. số state streaming ---
    states = enc.get_init_states(batch_size=1)
    states.append(embed.get_init_states(batch_size=1))   # cached left pad ConvNeXt
    states.append(torch.zeros(1, dtype=torch.int32))     # processed_lens
    L = sum(t(CFG["num_encoder_layers"]))
    assert len(states) == 6 * L + 2 == EXPECTED_STATES, len(states)
    print(f"[streaming] {len(states)} state = 6 x {L} layer + 2")

    # --- 2. hình dạng tensor qua từng khối ---
    N, T, U = 2, 500, 5
    x = torch.randn(N, T, 80)
    x_lens = torch.tensor([T, T - 40])
    with torch.no_grad():
        e, e_lens = embed(x, x_lens)
        eo, eo_lens = enc(e.permute(1, 0, 2), e_lens, src_key_padding_mask=None)
        eo = eo.permute(1, 0, 2)                          # (N, T', edim)
    assert eo.shape[-1] == edim
    ratio = T / eo.shape[1]
    print(f"[hình dạng] fbank {tuple(x.shape)} -> embed {tuple(e.shape)} "
          f"-> encoder {tuple(eo.shape)}  (hạ mẫu {ratio:.2f}x, "
          f"1 frame = {10 * ratio:.0f} ms)")
    assert 3.8 < ratio < 4.2, ratio

    # --- 3. pruned RNN-T loss, hai giai đoạn ---
    y_padded = torch.randint(3, V, (N, U), dtype=torch.int64)
    sos_y = torch.cat(
        [torch.full((N, 1), BLANK_ID, dtype=torch.int64), y_padded], dim=1
    )                                                     # (N, U+1)
    boundary = torch.zeros((N, 4), dtype=torch.int64)
    boundary[:, 2] = U
    boundary[:, 3] = eo_lens

    with torch.no_grad():
        decoder_out = dec(sos_y)                          # (N, U+1, DECODER_DIM)
        # giai đoạn 1: loss "simple" trên lưới đầy đủ T' x (U+1) x V,
        # không đi qua joiner, chỉ để lấy gradient
        simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
            lm=lm_proj(decoder_out),
            am=am_proj(eo),
            symbols=y_padded,
            termination_symbol=BLANK_ID,
            lm_only_scale=0.25,
            am_only_scale=0.0,
            boundary=boundary,
            reduction="sum",
            return_grad=True,
        )
        # giai đoạn 2: dùng gradient đó chọn s_range ô mỗi frame,
        # chỉ những ô sống sót mới đi qua joiner thật
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad, py_grad=py_grad,
            boundary=boundary, s_range=args.prune_range,
        )
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=joi.encoder_proj(eo), lm=joi.decoder_proj(decoder_out), ranges=ranges
        )
        logits = joi(am_pruned, lm_pruned, project_input=False)
        pruned_loss = k2.rnnt_loss_pruned(
            logits=logits.float(), symbols=y_padded, ranges=ranges,
            termination_symbol=BLANK_ID, boundary=boundary, reduction="sum",
        )

    assert logits.shape == (N, eo.shape[1], args.prune_range, V), logits.shape
    full = N * eo.shape[1] * (U + 1) * V
    print(f"[pruned] ranges {tuple(ranges.shape)}, logits {tuple(logits.shape)} "
          f"= (N, T', s_range, V)")
    print(f"[pruned] lưới đầy đủ {full} ô -> sau prune {logits.numel()} ô "
          f"({logits.numel() / full:.1%})")
    print(f"[loss] simple = {float(simple_loss):.2f}, "
          f"pruned = {float(pruned_loss):.2f}")

    print("\n[OK] các khối khớp nhau, 74 state, pruned RNN-T chạy được.")


if __name__ == "__main__":
    main()