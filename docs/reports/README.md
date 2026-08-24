# Research Reports — FANet

Index tổng hợp các báo cáo nghiên cứu của dự án FANet (Feedback Attention Network).

## Timeline

| # | File | Ngày | Nội dung chính |
|---|------|------|----------------|
| 1 | [2026-08-10_week2_setup.md](2026-08-10_week2_setup.md) | 10/08/2026 | Setup env, chạy code FANet, đọc sơ paper |
| 2 | [2026-08-20_uacanet-mixpool-gap.md](2026-08-20_uacanet-mixpool-gap.md) | 20/08/2026 | Đào sâu UACANet + research gap MixPool + kế hoạch analysis |
| 3 | [2026-08-24_e3-feedback-analysis.md](2026-08-24_e3-feedback-analysis.md) | 24/08/2026 | E3 feedback help/hurt + uncertainty correlation + 2 điểm yếu xác nhận |

## Experiment Tracking

| ID | Giả thuyết / Mục đích | Trạng thái | Kết quả chính | Artifacts |
|----|----------------------|------------|---------------|-----------|
| E1 | Gate binary `(fmask > 0.5)` cắt gradient → nhánh fmask không học | ✅ Done | fmask: **0/… params có gradient** sau 10 steps; conv1/conv2: 160 params có grad; weight fmask không đổi sau optimizer step | `logs/analysis_grad_log.txt`, `analysis/grad_flow.py` |
| E2 | Baseline: test-time refinement trên checkpoint 53 epochs | ✅ Done | Iter 1→2: Jaccard 0.2166→0.2251, F1 0.3147→0.3268 (feedback giúp nhẹ) | `results/test_results.csv`, `scripts/evaluate.py` |
| E3 | Hard binary feedback: khi nào giúp / khi nào hại; uncertainty trước threshold có correlate với lỗi không | ✅ Done | 65 giúp / 46 hại; mean delta +0.031; corr(prev_dice)=+0.355, corr(unc)=−0.297; err 48.7% (conf<0.1) vs 2.7% (conf>0.35); binary 0.334 vs soft 0.311 vs none 0.301 vs oracle 0.370 | `logs/feedback_analysis_log.txt`, `results/feedback_summary.json`, `analysis/feedback_analysis.py` |
| E4 | Đề xuất cơ chế feedback mới nhắm đúng điểm yếu đã xác nhận (confidence-gated + hồi sinh fmask) | 🔮 Planned | — | — |

## Quy trình cập nhật

1. **Cuối mỗi phiên làm việc**: cập nhật report hiện tại (Done, Findings, Experiments).
2. **Mỗi milestone** (xong 1 analysis / 1 experiment): tạo file report mới theo `_TEMPLATE.md` + cập nhật bảng Timeline và Experiment Tracking ở file này.
3. **Số liệu luôn trỏ về artifacts trong repo** (`logs/`, `results/`, `checkpoints/`) thay vì chép tay.
4. Khi đủ các mảnh (reproduce → gap → analysis → đề xuất), viết 1 bản **research summary** nối mạch toàn bộ quá trình.

## Artifacts chính

- Model checkpoint: `checkpoints/checkpoint.pth` (53 epochs, best val loss 0.505)
- Training log: `logs/train_log.txt`
- Papers: `docs/2103.17235v3.pdf` (FANet), `docs/3474085.3475375.pdf` (UACANet)
- Guide kiến trúc: `docs/FANet_Complete_Guide.md`
