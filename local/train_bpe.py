#!/usr/bin/env python3
"""
task : Huấn luyện tokenizer SentencePiece cho tiếng Việt, theo đúng quy ước icefall.
Vào  : data/split_<tag>/train.tsv          (M1)
       [tuỳ chọn] fbank_<tag>/train_cuts.jsonl.gz  (M2) để loại các utt đã bị drop
Ra   : data/lang_bpe_<V>/
         corpus.txt          text đã chuẩn hoá NFC, 1 câu/dòng
         unigram_<V>.model   model gốc do sentencepiece sinh
         unigram_<V>.vocab
         bpe.model           bản sao (tên mà icefall train.py/decode.py đi tìm)
         tokens.txt          "<piece> <id>" mỗi dòng
         meta.json           tham số + số liệu để truy vết

Quy ước icefall (KHÔNG được đổi):
    id 0 = <blk>   (blank của RNN-T)
    id 1 = <sos/eos>
    id 2 = <unk>
    bos_id = eos_id = -1
    character_coverage = 1.0

BẪY đã đo được, đọc trước khi sửa file này:
  (1) character_coverage < 1.0 KHÔNG báo lỗi mà âm thầm ném ký tự hiếm thành <unk>
      -> round-trip sai, model học nhãn thiếu chữ. Luôn để 1.0 để nó fail to.
  (2) sentencepiece mặc định normalization_rule_name="nmt_nfkc", tức là tự động
      NFC-hoá input. Nếu TSV đang ở NFD thì decode() trả về NFC, khác chuỗi gốc,
      round-trip fail và WER ở M6 bị thổi phồng. -> ép NFC ngay từ corpus.
  (3) Đếm ký tự bằng Python, đừng bằng shell: `grep -o . | sort -u` đếm theo byte
      trong locale C, đo được 64 thay vì 69 trên chính corpus này.
"""

import argparse
import csv
import json
import shutil
import subprocess
import unicodedata
from pathlib import Path

import sentencepiece as spm

FIELDS = ["utt_id", "speaker", "audio_path", "text"]
USER_SYMBOLS = ["<blk>", "<sos/eos>"]  # unk_id = 2 = len(USER_SYMBOLS)


def read_tsv(path: Path):
    """Đọc TSV của M1. Bắt buộc có header, sai header là raise (bẫy số 2 ở M1)."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Header sai: {reader.fieldnames} != {FIELDS}")
        return [dict(r) for r in reader]


def surviving_ids(cuts_path: Path):
    """Lấy tập supervision id còn sống sau bộ lọc thời lượng ở M2.

    Nhớ: cut.id có hậu tố kênh (…-0), supervision.id mới khớp utt_id trong TSV.
    """
    from lhotse import load_manifest

    cuts = load_manifest(cuts_path).to_eager()
    return {sup.id for c in cuts for sup in c.supervisions}


def build_corpus(rows, out_path: Path):
    """Chuẩn hoá NFC + gộp khoảng trắng, ghi 1 câu/dòng. Assert không còn dòng rỗng."""
    texts, n_renorm = [], 0
    for r in rows:
        raw = r["text"]
        t = unicodedata.normalize("NFC", raw).strip()
        t = " ".join(t.split())
        if not t:
            raise ValueError(f"Câu rỗng: {r['utt_id']}")
        if t != raw:
            n_renorm += 1
        texts.append(t)
    out_path.write_text("\n".join(texts) + "\n", encoding="utf-8")
    if n_renorm:
        print(f"[CẢNH BÁO] {n_renorm}/{len(rows)} câu trong TSV bị đổi khi ép NFC. "
              f"M6 phải chuẩn hoá reference y hệt, nếu không WER sẽ sai.")
    return texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", default="transcripts_matched/train.tsv", help="train.tsv của M1")
    p.add_argument("--cuts", default=None, help="train_cuts.jsonl.gz của M2 (tuỳ chọn)")
    p.add_argument("--lang-dir", default=None, help="mặc định data/lang_bpe_<vocab-size>")
    p.add_argument("--vocab-size", type=int, default=100)
    p.add_argument("--model-type", default="unigram", choices=["unigram", "bpe"])
    args = p.parse_args()

    V = args.vocab_size
    lang_dir = Path(args.lang_dir or f"data/lang_bpe_{V}")
    lang_dir.mkdir(parents=True, exist_ok=True)

    rows = read_tsv(Path(args.tsv))
    n_all = len(rows)
    if args.cuts:
        keep = surviving_ids(Path(args.cuts))
        rows = [r for r in rows if r["utt_id"] in keep]
        print(f"[lọc theo cuts] {n_all} -> {len(rows)} câu")
    if not rows:
        raise ValueError("Không còn câu nào để train tokenizer")

    corpus = lang_dir / "corpus.txt"
    texts = build_corpus(rows, corpus)

    # --- kiểm tra ngân sách vocab TRƯỚC khi train (bẫy 1 + bẫy 3) ---
    alphabet = set("".join(texts))
    need = len(alphabet) + len(USER_SYMBOLS) + 1  # need = số ký tự đơn + 2 ký hiệu <blk>/<sos/eos> + 1 cho <unk>
    print(f"[alphabet] {len(alphabet)} ký tự khác nhau (đếm bằng Python)")
    print(f"[ngân sách] cần tối thiểu {need}, vocab_size = {V}, "
          f"còn {V - need} slot cho subword")
    if need > V:
        raise SystemExit(
            f"vocab_size={V} quá nhỏ: riêng ký tự đơn đã cần {need}. "
            f"Tăng vocab_size, ĐỪNG hạ character_coverage."
        )

    prefix = str(lang_dir / f"{args.model_type}_{V}")
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        vocab_size=V,
        model_type=args.model_type,
        model_prefix=prefix,
        character_coverage=1.0,          # bẫy 1: không được hạ
        input_sentence_size=100000000,
        user_defined_symbols=USER_SYMBOLS,
        unk_id=len(USER_SYMBOLS),        # = 2
        bos_id=-1,
        eos_id=-1,
    )

    #--- copy model sang tên mà icefall hard-code: bpe.model ---
    shutil.copyfile(prefix + ".model", lang_dir / "bpe.model")

    sp = spm.SentencePieceProcessor()
    sp.load(str(lang_dir / "bpe.model"))

    # --- self-check, fail to ---
    assert sp.vocab_size() == V, sp.vocab_size()
    assert sp.id_to_piece(0) == "<blk>", sp.id_to_piece(0)
    assert sp.id_to_piece(1) == "<sos/eos>"
    assert sp.unk_id() == 2 and sp.bos_id() == -1 and sp.eos_id() == -1

    bad, n_unk, n_tok = [], 0, 0
    for t in texts:
        ids = sp.encode(t)
        n_tok += len(ids)
        n_unk += sum(1 for i in ids if i == sp.unk_id())
        if sp.decode(ids) != t:
            bad.append(t)
    if bad:
        raise SystemExit(f"Round-trip FAIL ở {len(bad)} câu, ví dụ: {bad[0]!r}")
    if n_unk:
        raise SystemExit(f"Có {n_unk} token <unk> — nhãn đã mất chữ")

    with open(lang_dir / "tokens.txt", "w", encoding="utf-8") as f:
        for i in range(V):
            f.write(f"{sp.id_to_piece(i)} {i}\n")

    n_char = sum(len(t) for t in texts)
    
    # Lấy commit hiện tại để truy vết, nếu không có git thì ghi "unknown". 
    # Ở M6, nếu WER cao thì có thể check xem có phải do model bpe khác commit hay không.
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"
    meta = {
        "tsv": args.tsv,
        "cuts": args.cuts,
        "n_rows_tsv": n_all,
        "n_rows_used": len(rows),
        "vocab_size": V,
        "model_type": args.model_type,
        "character_coverage": 1.0,
        "normalization": "NFC (ép trong build_corpus)",
        "alphabet_size": len(alphabet),
        "alphabet": "".join(sorted(alphabet)),
        "n_tokens": n_tok,
        "tokens_per_char": round(n_tok / n_char, 4),
        "git_sha": sha,
    }
    (lang_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[OK] round-trip khớp {len(texts)}/{len(texts)} câu, 0 <unk>")
    print(f"[OK] {n_tok} token / {n_char} ký tự = {meta['tokens_per_char']} token/ký tự")
    print(f"[OK] đã ghi {lang_dir}/bpe.model, tokens.txt, meta.json")


if __name__ == "__main__":
    main()