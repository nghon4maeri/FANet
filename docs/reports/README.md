# Research Reports — FANet

Index tổng hợp các báo cáo nghiên cứu của dự án FANet (Feedback Attention Network).

## Timeline

| # | File | Ngày | Nội dung chính |
|---|------|------|----------------|
| 1 | [2026-08-10_week2_setup.md](2026-08-10_week2_setup.md) | 10/08/2026 | Setup env, chạy code FANet, đọc sơ paper |
| 2 | [2026-08-20_uacanet-mixpool-gap.md](2026-08-20_uacanet-mixpool-gap.md) | 20/08/2026 | Đào sâu UACANet + research gap MixPool + kế hoạch analysis |
| 3 | [2026-08-24_feedback-help-hurt-analysis.md](2026-08-24_feedback-help-hurt-analysis.md) | 24/08/2026 | Phân tích feedback help/hurt + uncertainty correlation + 2 điểm yếu xác nhận |
| 4 | [2026-08-27_confidence-gated-feedback.md](2026-08-27_confidence-gated-feedback.md) | 27/08/2026 | Checkpoint 200ep Kaggle + kiểm định confidence-gating (bác bỏ) + hướng negative feedback |
| 5 | [2026-08-31_oracle-background-analysis.md](2026-08-31_oracle-background-analysis.md) | 31/08/2026 | Phase 1: oracle pixel analysis xác nhận giả thuyết background (84.6% khác biệt ở vùng FP) |
| 6 | [2026-08-31_phase2-3-ablation.md](2026-08-31_phase2-3-ablation.md) | 31/08/2026 | Phase 2+3: literature + hypotheses + ablation 2x2 (STE hồi sinh fmask, dual-path); P_B1 bác bỏ, P_A1 xác nhận |
| 7 | [2026-08-31_phase4-train-end-to-end.md](2026-08-31_phase4-train-end-to-end.md) | 31/08/2026 | Phase 4: code train 4 cells sẵn sàng + smoke-test CPU + fallback loss; chờ chạy Kaggle T4 |
| 8 | [2026-08-31_phase4-results.md](2026-08-31_phase4-results.md) | 31/08/2026 | Phase 4 kết quả: T_A bác bỏ (binary 0.2806 > STE 0.1951); bug dual-path phát hiện + sửa; cần chạy lại T01/T11 |
| 9 | [2026-09-04_next-directions.md](2026-09-04_next-directions.md) | 04/09/2026 | Research strategy: audit bằng chứng (train/frozen/invalid), literature 25 paper đa nguồn + 29 PDF, novelty assessment (FANetv2 + predictive-coding 2026), xếp hạng X1–X6 → ưu tiên loss-side FP penalty + multi-seed; STE/dual-path bế tắc |

## Experiment Tracking

| Giả thuyết / Mục đích | Trạng thái | Kết quả chính | Artifacts |
|----------------------|------------|---------------|-----------|
| Gate binary `(fmask > 0.5)` cắt gradient → nhánh fmask không học | Done | fmask: 0/… params có gradient sau 10 steps; conv1/conv2: 160 params có grad; weight fmask không đổi sau optimizer step | `logs/analysis_grad_log.txt`, `analysis/grad_flow.py` |
| Baseline: test-time refinement trên checkpoint 53 epochs | Done | Iter 1→2: Jaccard 0.2166→0.2251, F1 0.3147→0.3268 (feedback giúp nhẹ) | `results/test_results.csv`, `scripts/evaluate.py` |
| Hard binary feedback: khi nào giúp / khi nào hại; uncertainty trước threshold có correlate với lỗi không | Done | 65 giúp / 46 hại; mean delta +0.031; corr(prev_dice)=+0.355, corr(unc)=−0.297; err 48.7% (conf<0.1) vs 2.7% (conf>0.35); binary 0.334 vs soft 0.311 vs none 0.301 vs oracle 0.370 | `logs/feedback_analysis_log.txt`, `results/feedback_summary.json`, `analysis/feedback_analysis.py` |
| Đề xuất cơ chế feedback mới nhắm đúng điểm yếu đã xác nhận (confidence-gated + hồi sinh fmask) | Planned | — | — |
| Kiểm định confidence-gated feedback: 9 variants (binary, conf-weighted, gated tau 0.05-0.30, soft, none, oracle) | Done | Bác bỏ: binary 0.2390 tốt nhất, mọi biến thể confidence-based thấp hơn; hurt không giảm; oracle gap 0.092 | `logs/confgated_log.txt`, `results/confgated_summary.json`, `analysis/confgated_feedback.py` |
| Baseline refinement 10 iterations trên checkpoint 200ep Kaggle | Done | F1 0.2430 → 0.2414 (plateau, feedback gần như không giúp) | `results/test_results.csv` |
| Oracle vs prediction feedback: oracle tốt hơn nhờ thông tin gì (background hay foreground)? | Done | 84.6% khác biệt ở vùng prediction-FP; FP/FN = 2.48; FP cách biên GT 40.7px → xác nhận thiếu thông tin background | `logs/oracle_analysis_log.txt`, `results/oracle_summary.json`, `analysis/oracle_analysis.py` |
| STE/soft hồi sinh gradient fmask (P_A1) | Done | binary=0 grad; STE=0.125, soft=0.106 tổng grad 8 MixPool | `logs/grad_flow_gates_log.txt`, `analysis/grad_flow_gates.py` |
| Ablation 2x2 dual-path + gate variants (P_B1/P_B2/P_B3/H_AB) | Done | P_B1 BÁC BỎ (FP +2.92pp, p=1e-7); Dice +0.030 nhưng p=0.035; neg-ctrl sụp 0.083 (P_B3 ok); không synergy (C11<C10) | `logs/abl2x2_log.txt`, `results/abl2x2_summary.json`, `results/abl2x2_stats.json` |
| Phase 4: train end-to-end 4 cells (T00/T10/T01/T11) loại BN mismatch | ⏳ Chờ chạy Kaggle T4 | Code + smoke-test CPU OK (4 cells 1 epoch, loss ~0.79); fallback negdice OK (loss ~1.25) | `notebooks/fanet_kaggle_phase4.py`, `scripts/train.py`, `src/fanet/losses.py` |
| Phase 4 kết quả: T_A (STE train vs binary train) | Done | T_A KHÔNG ỦNG HỘ: binary 0.2806 > STE 0.1951 (delta -0.0855, p=0.277); train lại loại BN mismatch (frozen 0.239 → trained 0.281) | `results/phase4_eval.json`, `results/phase4_stats.json` |
| Phase 4 bug: T01/T11 == T00/T10 (m_bg zeros khi train) | Fixed | Phát hiện bằng weight diff = 0; sửa m_bg = 1 - m_fg; cần chạy lại T01/T11 trên Kaggle | `notebooks/fanet_kaggle_phase4.py` |
| Phase 4 run mới (papermill 9/4): T00/T10/T01/T11 + bug complement m_bg=1−m_fg | Done (chẩn đoán) | T00 0.5590 / T10 0.5611 / T01 0.6478 / T11 collapse (1.11–1.13 từ ep 40) — T01/T11 INVALID (complement triệt tiêu fmask); STE ≈ binary ở run này | `kaggle/fanet-phase4.ipynb` (cell 17) |
| Literature grounding đa nguồn cho next directions (OpenAlex/arXiv/Europe PMC) | Done | 25 paper curated + novelty assessment (FANetv2, predictive-coding 2026); 29 PDF tải về docs/ | `docs/literature_grounding_next.md`, `docs/papers_not_accessible.md` |
| Xếp hạng hướng tiếp theo X1–X6 | Planned | Ưu tiên X2+X5 (loss-side FP penalty, multi-seed ≈21+7 GPU-h); dự phòng X4 (0 GPU); X1 chỉ nếu X2 thất bại; pivot rule: FP không giảm ≥1pp ở seed đầu → reframe negative-result | `docs/reports/2026-09-04_next-directions.md` |

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
