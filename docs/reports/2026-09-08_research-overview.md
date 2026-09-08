# Báo cáo Tổng quan Nghiên cứu (Comprehensive Research Overview Report)

**Dự án:** FANet — Reproduction + Cải tiến cho phân đoạn ảnh y sinh (Polyp Segmentation)
**Ngày:** 08/09/2026
**Vai trò:** Research Assistant / Co-author overview
**Mục đích:** Tổng hợp toàn bộ hành trình nghiên cứu (reproduce → tìm research gap →
phân tích → thử nghiệm → định hướng paper) để phục vụ slide presentation và lập kế
hoạch viết bài báo khoa học.

> **Quy ước trình bày bằng chứng:** Mỗi phát hiện quan trọng được trình bày theo
> **4 lớp**:
> - **Lập luận** — *tại sao chúng tôi nói vậy* (chuỗi suy luận logic).
> - **Bằng chứng** — *số liệu cụ thể đo được*.
> - **Cơ chế** — *tại sao lại có kết quả đó* (giải thích nguyên nhân).
> - **Nguồn** — *bằng chứng nằm ở file nào trong repo* (để tái lập được).
>
> **Quy ước độ tin cậy:** theo AGENTS.md, mỗi con số được phân loại **TRAIN** (từ
> mô hình train end-to-end, đáng tin, thường 1 seed/1 split), **FROZEN EVAL** (đo trên
> model cũ — chỉ phản ánh "khả năng khai thác feedback của model hiện có", KHÔNG phải
> giá trị đầy đủ của cơ chế), **INVALID** (do bug — không dùng cho claim nào).

---

## MỤC 1. Tóm tắt & Phân tích Paper gốc (FANet)

### 1.1. Thông tin paper

| Thuộc tính | Giá trị |
|---|---|
| Tên | **FANet: A Feedback Attention Network for Improved Biomedical Image Segmentation** |
| Tác giả | Tomar, Jha, Riegler, Johansen, Johansen, Rittscher, Halvorsen, Ali |
| Venue | IEEE Transactions on Neural Networks and Learning Systems (**TNNLS**), 2022 |
| DOI | 10.1109/tnnls.2022.3159394 |
| arXiv | 2103.17235 |
| Nguồn PDF | `papers/original/2103.17235v3.pdf` (+ bản dịch `papers/translated/2103.17235v3-vi.pdf`) |
| Tài liệu phân tích kiến trúc | `docs/FANet_Complete_Guide.md` |

### 1.2. Bài toán cốt lõi (Core Problem)

Phân đoạn pixel của các cấu trúc y sinh (polyp nội soi, mạch máu võng mạc, tổn thương
da...) — **binary semantic segmentation** (mỗi pixel ∈ {foreground, background}).

### 1.3. Động lực & Ý tưởng chính

- **Động lực:** mask dự đoán ở epoch $t$ bị vứt bỏ hoàn toàn khi sang epoch $t+1$ —
  không có "ký ức" ở dạng mask. FANet đề xuất tái sử dụng nó.
- **Ý tưởng:** dùng **mask epoch trước** làm hard-attention map đưa vào input; module
  **MixPool** dùng mask này để **prune feature** không liên quan; tạo **vòng phản hồi
  xuyên epoch** + **test-time iterative refinement**.

### 1.4. Kiến trúc (mô tả lại từ `docs/FANet_Complete_Guide.md` + `src/fanet/models/`)

- U-Net encoder-decoder 4 tầng (3→32→64→128→256), ~7.72M params (`scripts/report.py`).
- Input: cặp `[RGB image, previous mask]`.
- **MixPool** (`src/fanet/models/blocks.py`):
  ```python
  fmask = (self.fmask(x) > 0.5).float()               # nhánh attention "tự học" (cứng)
  x1 = x * logical_or(fmask, m > 0).float()           # gate binary OR với mask feedback
  x2 = self.conv2(x)                                  # 50% feature gốc (safety net)
  x = concat([x1, x2], 1)
  ```
- Loss: **DiceBCELoss** (0.5·BCE + 0.5·Dice) — `src/fanet/losses.py`.
- Training: mask khởi tạo Otsu, nén RLE, cập nhật khi val loss cải thiện — `scripts/train.py`.

### 1.5. Ưu điểm (Strengths)

| # | Ưu điểm | Nguồn xác nhận |
|---|---|---|
| 1 | Ý tưởng feedback attention mới lạ | Paper gốc + `docs/literature_grounding_next.md` (loạt work theo sau: FANetv2, FEGNet) |
| 2 | Kiến trúc nhẹ (~7.7M), chạy được CPU | `logs/report_log.txt`, train CPU 1.5–2 phút/epoch |
| 3 | Test-time refinement có thể giúp | `results/test_results.csv` (checkpoint 53ep: F1 0.315→0.327) |
| 4 | Safety net 50% feature gốc | `src/fanet/models/blocks.py` (thiết kế) |
| 5 | Baseline được cộng đồng dùng | Literature: nhiều paper trích dẫn + so sánh |

### 1.6. Điểm yếu / Hạn chế — KÈM BẰNG CHỨNG

> Đây là phần then chốt: **mọi hướng cải tiến của dự án xuất phát từ 2 điểm yếu đã
> xác nhận bằng dữ liệu** dưới đây.

#### Điểm yếu 1. Gate binary cắt gradient → nhánh attention "tự học" chưa từng được học

- **Lập luận:** Trong MixPool, nhánh `fmask` dùng phép so sánh `(x > 0.5)` để tạo gate.
  Phép so sánh này có đạo hàm bằng 0 ở mọi điểm (không khả vi) → gradient không thể
  truyền ngược về trọng số của nhánh `fmask`. Nếu đúng, nhánh này giữ nguyên giá trị
  khởi tạo suốt quá trình train.
- **Bằng chứng (TRAIN/analytic):** Sau 10 training steps: **0/160** tham số `fmask` có
  gradient, trong khi `conv1`/`conv2` đủ 160/160; trọng số `fmask` không đổi sau
  `optimizer.step()`. Bằng chứng quyết định: so model khởi tạo, **BN affine diff
  (gamma/beta init = 1/0) lệch chính xác 0.000** sau 53 epochs — chỉ có thể xảy ra nếu
  nhánh chưa bao giờ được update.
- **Cơ chế:** "Feature-based attention" mà paper quảng cáo thực chất là **filter
  random-init**; toàn bộ sức mạnh MixPool đến từ mask feedback ngoài. Đây là bug thật
  của implementation, không chỉ là hạn chế thiết kế.
- **Nguồn:** `analysis/grad_flow.py` + `logs/analysis_grad_log.txt`; tái khẳng định ở
  `docs/reports/2026-08-24_feedback-help-hurt-analysis.md` (F1).

#### Điểm yếu 2. Hard threshold vứt bỏ confidence/uncertainty

- **Lập luận:** Threshold 0.5 biến xác suất liên tục thành {0,1} — mọi pixel được đối
  xử ngang nhau trong feedback mask, dù mức chắc chắn khác xa nhau. Cần kiểm tra xem
  độ chắc chắn (confidence trước threshold) có tương quan với lỗi không.
- **Bằng chứng (TRAIN):** Error rate giảm monotonic theo confidence: **48.7%** (conf
  < 0.1, tương đương đoán mò) → **2.7%** (conf > 0.35). FP tập trung ở vùng confidence
  thấp (35.3% ở bin thấp nhất, 0.0% ở bin cao nhất).
- **Cơ chế:** Vùng model "lưỡng lự" (conf≈0.5) chính là vùng khó (thường là biên
  polyp). Threshold 0.5 đưa cả pixel sai 48.7% lẫn pixel sai 2.7% vào feedback với
  trọng số bằng nhau → **khuếch đại lỗi vùng không chắc chắn**.
- **Nguồn:** `analysis/feedback_analysis.py` → `logs/feedback_analysis_log.txt`,
  `results/feedback_uncertainty_bins.csv`, `kaggle/figures/fig3_uncertainty_error.png`.

#### Điểm yếu 3. Feedback chỉ chứa foreground — background bị zero-hoá

- **Lập luận:** Mask feedback là mask nhị phân foreground (pixel 1 = polyp, 0 = nền bị
  bỏ). Nếu model over-segment (dự đoán thừa ra nền), feedback sẽ *khuếch đại* sai lầm
  này thay vì sửa. Để kiểm chứng, so sánh khi feedback = ground truth (oracle) vs
  feedback = prediction: nếu oracle tốt hơn hẳn ở vùng nào, đó là thông tin mà
  prediction feedback đang thiếu.
- **Bằng chứng (FROZEN, checkpoint 200ep):** FP/FN ratio = **2.48** (FP rate 11.7% vs
  FN rate 4.7%); trong 6.98% pixel mà oracle khác prediction feedback, **84.6%** nằm ở
  vùng prediction-FP (dự đoán thừa). Oracle gap = **+0.093** sau 3 iterations, mở ra
  ngay từ iter 2 và giữ ổn định.
- **Cơ chế:** Oracle "dạy" model chỗ nào KHÔNG phải polyp — thông tin background chính
  xác mà hard binary feedback (foreground-only) không bao giờ có. Lỗi FP trải **40.7px**
  xa biên GT, FN chỉ **8.6px** → over-segmentation tràn sâu ra nền, không chỉ sai rìa.
- **Nguồn:** `analysis/oracle_analysis.py` → `logs/oracle_analysis_log.txt`,
  `results/oracle_summary.json`, `results/oracle_analysis.csv`; tóm tắt `docs/reports/
  2026-08-31_oracle-background-analysis.md`.

#### Điểm yếu 4. DiceBCELoss thuần không phân biệt vùng thừa

- **Lập luận:** Root cause đã xác nhận là over-segmentation (FP). Loss chính của FANet
  (Dice+BCE) không có cơ chế nào phạt riêng vùng FP — Dice bất đối xứng thấp, BCE
  cân bằng 2 lớp. Chưa từng có thí nghiệm can thiệp FP từ phía loss.
- **Bằng chứng:** Không có artifact nào trong chuỗi Phase 1–4 dùng loss phạt FP; đây
  là khoảng trống được xác định trong `docs/reports/2026-09-04_next-directions.md` (F3).
- **Cơ chế:** Hướng loss-side (wIoU, weighted Dice, Tversky β<0.5...) là cách trực tiếp
  nhất tấn công over-segmentation mà không vướng confounder kiến trúc.
- **Nguồn:** `docs/reports/2026-09-04_next-directions.md`, `docs/literature_grounding_next.md`
  (các dòng WSDice/Lovász/Tversky).

---

## MỤC 2. Tổng quan các Nghiên cứu & Paper liên quan (Related Works)

### 2.1. Bối cảnh & Nguồn literature

- **Quy trình:** literature grounding đa nguồn (OpenAlex 33 queries, arXiv 8, Europe
  PMC 5, Semantic Scholar bị 429, PubMed bị chặn) → 41 ứng viên → **32 paper curated**;
  29+ PDF tải về; đọc full-text 7 paper (FEGNet, RefineU-Net, Conditional Boundary
  Loss, BCNet, CTNet, CFA-Net, BUNet).
- **Nguồn:** `docs/literature_grounding_next.md` (bảng 32 paper + đánh giá novelty +
  provenance từng DOI), `docs/literature_grounding_phase2.md`,
  `docs/literature_grounding_phase4.md`, `docs/paper_download_log.json`,
  `papers_not_accessible.md`.

### 2.2. Đột phá chính của các paper liên quan/đối chứng

| Nhóm | Paper (venue, năm) | Đột phá chính | Nguồn |
|---|---|---|---|
| Feedback mask loop | **FANetv2** (ICASSP 2025, cùng lab) | Mở rộng FANet: iterative mask feedback + text-guided + refinement test | `literature_grounding_next.md` (row 2) |
| Feedback prediction-error | **Predictive coding polyp** (RS 2026) | Top-down prediction-error feedback; đo thẳng FPR/FDR + FP-trên-nếp-gấp | Row 3 |
| Feature-level feedback | **FEGNet** (JBHI 2023), **RefineU-Net** (PRL 2020) | Feedback = feature/attention nội mạng, soft gate, deep-sup, ở skip — cố ý tránh mask-at-input | Row 4, 27 + full-text PDF |
| Background/reverse attention | **PraNet** (MICCAI 2020), **UACANet** (ACM MM 2021) | Reverse attention (1−pred); tách foreground/background/uncertain + context pool | Row 7, 8 + `docs/reports/2026-08-20_*.md` |
| Loss-side | **WSDice** (Access 2020), **Lovász** (CVPR 2018), **Tversky** (2017) | Phạt FP trong loss; **5/7 paper polyp vừa đọc dùng wIoU+wBCE** | Row 14–16; report 09/05 F1 |
| Boundary/edge | **CFA-Net** (PR 2023), **BCNet** (JBHI 2022), **MEGANet** (WACV 2024), **BUNet** (NN 2024) | Boundary-weighted wIoU+wBCE; dual output; edge-guided | Row 29, 31, 12, 32 |
| Methodology | **Metrics Reloaded** (Nature Methods 2024) | Chuẩn metric/CI/report | Row 20 |

### 2.3. So sánh ngắn gọn FANet vs các nghiên cứu liên quan

| Khía cạnh | FANet (gốc) | FEGNet / RefineU-Net | PraNet / UACANet | Predictive-coding 2026 | WSDice / wIoU-family |
|---|---|---|---|---|---|
| Loại feedback | Mask ở **input**, hard | Feature **nội mạng**, soft | Single-pass | Prediction-error inference | Không |
| Gate | Hard threshold (non-diff) | Soft sigmoid | Soft attention | — | — |
| Giám sát | Chỉ loss cuối | Deep-sup từng step | Reverse attention | — | — |
| Background | Bị zero-hoá | Attention tự học xen kẽ bg | Context rõ ràng | Ngầm qua error map | Trọng số phạt FP |
| Loss | DiceBCE thuần | wIoU+wBCE | wIoU+wBCE-ish | — | wIoU+wBCE, WSDice |
| Tấn công over-seg | Không trực tiếp | — | Reverse attention | Feedback bg | **FP-penalty trong loss** |

### 2.4. Kết luận đối chiếu — kèm bằng chứng

#### KL2.1. Novelty "background-aware feedback loop" CÒN NHƯNG MỎNG

- **Lập luận:** (1) FANetv2 (cùng lab) đã chiếm không gian "iterative mask feedback";
  (2) predictive-coding 2026 đã làm "dọn FP qua feedback" với error map ngầm mang thông
  tin background; (3) không paper nào có **kênh background tường minh** → hình thức
  còn khác biệt, nhưng phạm vi hẹp.
- **Bằng chứng:** `literature_grounding_next.md` mục "Đánh giá novelty" (5 luận điểm) +
  bảng audit PDF.
- **Cơ chế:** Novelty không chỉ là "chưa ai làm" mà còn phải có **bằng chứng thực
  nghiệm ủng hộ**. Bằng chứng hiện có (Phase 3 FP tăng p=1e-7; Phase 4 dual-path train
  xấu/collapse) **chống lại** giá trị thực tiễn → rủi ro đặt cược paper cao.
- **Nguồn:** `docs/literature_grounding_next.md`, `docs/reports/2026-09-04_next-directions.md` (F2), `docs/reports/2026-09-05_new-papers-review.md` (F4).

#### KL2.2. Các work feedback "thành công" làm NGƯỢC với FANet gốc

- **Lập luận:** FEGNet (JBHI 2023) và RefineU-Net (PRL 2020) cùng gọi là "feedback"
  nhưng đều ở **feature/attention nội mạng** (soft gate + deep supervision từng
  time-step + đặt ở skip connection), không phải mask-at-input. Nếu kiểu "mask-at-input
  + hard gate + không giám sát" của FANet là thiết kế kém, ta sẽ thấy các work này cố
  ý né nó.
- **Bằng chứng:** FEGNet trích dẫn FANet và cố ý đặt feedback ở skip thay vì input
  ("rather than merely connecting the input and output"). RefineU-Net quan sát attention
  tự học xen kẽ ROI/background → thông tin background đã tồn tại nội tại, không cần
  kênh tường minh.
- **Cơ chế:** mask-at-input đổi phân phối đầu vào mỗi epoch → BN shock là hệ quả thiết
  kế; soft gate giữ gradient; deep supervision ép gate học có mục tiêu. Đây là 3 điểm
  FANet gốc thiếu.
- **Nguồn:** full-text `papers/original/FEGNet_*.pdf`, `papers/original/1-s2.0-S0167865520302592-main.pdf`
  (RefineU-Net); tóm tắt `docs/reports/2026-09-05_new-papers-review.md` (F2).

#### KL2.3. Lỗ hổng lớn nhất: over-segmentation chưa từng bị tấn công từ phía loss

- **Lập luận:** Root cause = over-segmentation (FP, cách biên 40.7px). Các paper polyp
  hiện đại dùng wIoU+wBCE làm loss chính trên chính Kvasir-SEG — một hướng rẻ, drop-in,
  có bằng chứng từ nhiều nhóm, và FANet gốc chưa dùng.
- **Bằng chứng:** 5/7 paper đọc full-text dùng wIoU+wBCE (FEGNet, BCNet, CFA-Net...);
  row WSDice đã verify công thức gốc (07/09).
- **Cơ chế:** wIoU phạt vùng thừa mạnh hơn Dice; wBCE có trọng số pixel. Không thay đổi
  kiến trúc → không vướng confounder BatchNorm.
- **Nguồn:** `docs/reports/2026-09-05_new-papers-review.md` (F1), `docs/literature_grounding_next.md` (row 14 WSDice verified), `src/fanet/losses.py`.

---

## MỤC 3. Tổng hợp công việc đã thực hiện & Kết quả (Current Work & Findings)

### 3.1. Các phase nghiên cứu — Mục tiêu, Công việc, Trạng thái

> Toàn bộ dự án gồm các giai đoạn tuần tự từ khởi động đến Phase 5. Mỗi giai đoạn
> có **mục tiêu rõ ràng (làm gì)**, **công việc đã làm** và **trạng thái**. Phase 5
> (loss-side) hiện chia 2 phần: **setup đã xong, phần chạy & đánh giá đang chờ Kaggle**.

| Giai đoạn | Mục tiêu (làm gì) | Công việc đã thực hiện | Trạng thái | Nguồn |
|---|---|---|---|---|
| **Khởi động — Setup & Reproduce** (10/08) | Chạy lại được FANet, hiểu cơ chế feedback, tìm research gap ban đầu | Setup env (Python 3.12, PyTorch); train FANet trên sessile-Kvasir-SEG (196 ảnh: 156/40), 53 epochs, val loss 0.505; viết guide kiến trúc | ✅ Done | `docs/reports/2026-08-10_*.md`, `logs/train_log.txt`, `docs/FANet_Complete_Guide.md` |
| **Chẩn đoán cơ chế** (20–24/08) | Xác định FANet đang mất thông tin gì, cơ chế feedback hoạt động thực sự ra sao | Đọc sâu UACANet, so MixPool (6 gap); phát hiện nhánh attention **zero-gradient**; baseline test-time refinement; help/hurt + uncertainty | ✅ Done | `docs/reports/2026-08-20_*.md`, `2026-08-24_*.md`, `analysis/grad_flow.py`, `logs/analysis_grad_log.txt`, `results/test_results.csv` |
| **Kiểm định confidence-gating** (27/08) | Kiểm tra cơ chế feedback mới (confidence-gated) có giảm hurt không | Checkpoint 200ep Kaggle; 9 variants → **BÁC BỎ** confidence-gating; oracle gap 0.092 | ✅ Done | `docs/reports/2026-08-27_*.md`, `analysis/confgated_feedback.py`, `results/confgated_summary.json` |
| **Phase 1 — Oracle & giả thuyết background** (31/08) | Trả lời: oracle tốt hơn nhờ thông tin gì → định vị root cause | Oracle pixel analysis: FP/FN=2.48, 84.6% khác biệt ở vùng FP, FP cách biên 40.7px → root cause = **over-segmentation** | ✅ Done | `docs/reports/2026-08-31_oracle-*.md`, `analysis/oracle_analysis.py`, `results/oracle_summary.json` |
| **Phase 2+3 — Ablation 2×2 (frozen)** (31/08) | Kiểm tra 2 cơ chế sửa: hồi sinh nhánh attention (STE/soft) + dual-path background feedback | STE/soft hồi sinh gradient (0→0.125/0.106); dual-path **FP tăng +2.92pp (p=1e-7)** → bác bỏ; negative control OK; không synergy | ✅ Done (FROZEN) | `docs/reports/2026-08-31_phase2-3-*.md`, `analysis/abl2x2_*.py`, `results/abl2x2_stats.json` |
| **Phase 4 — Train end-to-end** (31/08→04/09) | Loại bỏ confounder BatchNorm-mismatch bằng cách train từng cơ chế | Train 4 mô hình trên Kaggle T4 (seed 42); baseline binary Dice 0.2806 vs STE 0.1951; phát hiện + sửa 2 bug dual-path | ✅ Done (1 seed; mô hình dual-path bị bug) | `docs/reports/2026-08-31_phase4-*.md`, `results/phase4_eval.json`, `kaggle/fanet-phase4.ipynb` |
| **Strategy — Next directions** (04/09) | Đối chiếu toàn bộ bằng chứng, ground literature, xếp hạng hướng đi | Audit bằng chứng 3 mức; 25 paper; chẩn đoán 6 hướng → ưu tiên **loss-side FP penalty**; đóng STE rẻ | ✅ Done | `docs/reports/2026-09-04_*.md`, `docs/literature_grounding_next.md` |
| **Review 7 paper mới** (05/09) | Đọc full-text 7 paper mới, đối chiếu cơ chế, cập nhật plan | Xác nhận pattern wIoU+wBCE (5/7 paper); feedback thành công là feature-level; audit sửa 6 PDF sai | ✅ Done | `docs/reports/2026-09-05_*.md` |
| **Phase 5 — setup loss-side** (07/09) | Xây dựng thí nghiệm phạt FP từ phía loss (hướng ưu tiên sau audit) | Verify công thức WSDice + CFA-Net (pivot far-weighted); implement 5 loss; code train + smoke-test 3 cells; chẩn đoán STE → **đóng STE vĩnh viễn**; notebook Kaggle sẵn sàng | ✅ Code xong, smoke OK | `docs/reports/2026-09-07_*.md`, `src/fanet/losses.py`, `scripts/train.py`, `kaggle/fanet-phase5.ipynb`, `results/x4_ste_diagnosis.json` |
| **Phase 5 — chạy & đánh giá** (đang chờ) | Chạy no-feedback + 2 loss mới trên GPU, ra phán quyết | 3 cells × 200 epochs trên Kaggle T4 (seed 43) → download checkpoint → `analysis/phase5_eval.py` → phán quyết bậc 1/2 | ⏳ **CHỜ user chạy Kaggle** | `kaggle/fanet-phase5.ipynb`, `analysis/phase5_eval.py`, `analysis/phase5_figures.py` |

### 3.2. Các phát hiện chính — TRUNG TÂM BẰNG CHỨNG (4 lớp)

#### F1. Nhánh attention "tự học" của FANet chưa bao giờ được học (TRAIN/analytic)

- **Lập luận:** non-differentiable gate → gradient = 0 → trọng số bất biến. Kiểm chứng
  bằng 2 cách độc lập: đếm tham số có gradient, và so trọng số/BN với model khởi tạo.
- **Bằng chứng:** 0/160 params có gradient (conv: 160/160); BN affine diff = **0.000**;
  weight không đổi sau optimizer step.
- **Cơ chế:** `(fmask > 0.5)` là hard binary; đạo hàm bằng 0 mọi nơi → backprop cụt.
  BN running stats vẫn trôi (do forward) nhưng affine (gamma/beta) không đổi — chứng
  cứ quyết định.
- **Nguồn:** `analysis/grad_flow.py` → `logs/analysis_grad_log.txt`; phân tích chi tiết
  `docs/reports/2026-08-24_feedback-help-hurt-analysis.md` (F1).

#### F2. STE/soft hồi sinh gradient được (TRAIN/analytic) — điều kiện cần đạt

- **Lập luận:** nếu thay hard gate bằng straight-through estimator (STE) hoặc soft
  gate, gradient sẽ chảy vào nhánh attention. Kiểm chứng bằng tổng gradient qua 8
  MixPool.
- **Bằng chứng:** binary = 0; **STE = 0.125; soft = 0.106** (3 steps).
- **Cơ chế:** STE chặn gradient qua phép so sánh nhưng truyền thẳng gradient đầu vào;
  soft gate là hàm khả vi.
- **Nguồn:** `analysis/grad_flow_gates.py` → `logs/grad_flow_gates_log.txt`;
  `docs/reports/2026-08-31_phase2-3-ablation.md` (F1).

#### F3. Uncertainty correlate mạnh với lỗi — nhưng gating KHÔNG giúp (âm tính có giá trị)

- **Lập luận:** vì vùng conf thấp là nơi lỗi cao, giả thuyết "cắt/giảm trọng số vùng
  conf thấp sẽ giảm hurt". Kiểm định 9 biến thể (binary, conf-weighted, gated tau
  0.05–0.30, soft, none, oracle).
- **Bằng chứng:** **BÁC BỎ** — binary 0.2390 tốt nhất, mọi biến thể confidence-based
  thấp hơn; hurt không giảm (ví dụ gated-0.3 giữ 2.8% foreground, delta âm). Help/hurt:
  soft giúp nhiều case nhất (24) nhưng mean delta chỉ +0.0118.
- **Cơ chế:** lỗi và thông tin hữu ích nằm **cùng một chỗ** (vùng conf thấp = biên).
  Cắt vùng lỗi = cắt luôn vùng model cần guidance nhất.
- **Nguồn:** `analysis/confgated_feedback.py` → `logs/confgated_log.txt`,
  `results/confgated_summary.json`, `results/confgated_variants.csv`,
  `kaggle/figures/fig1_variant_dice.png`, `fig2_help_hurt.png`, `fig3_uncertainty_error.png`;
  `docs/reports/2026-08-27_confidence-gated-feedback.md`.

#### F4. Root cause = over-segmentation vùng nền xa biên (FROZEN, nhất quán)

- **Lập luận:** oracle gap là dư địa thật; phân rã vị trí khác biệt oracle-vs-prediction
  sẽ cho biết thông tin còn thiếu.
- **Bằng chứng:** FP/FN = **2.48**; **84.6%** khác biệt ở vùng prediction-FP; FP cách
  biên **40.7px**, FN **8.6px**; oracle gap **+0.093** ổn định qua iterations.
- **Cơ chế:** feedback foreground-only khuếch đại over-segmentation; model thiếu thông
  tin background để "dọn" vùng nền tràn.
- **Nguồn:** `analysis/oracle_analysis.py` → `logs/oracle_analysis_log.txt`,
  `results/oracle_summary.json`; `docs/reports/2026-08-31_oracle-background-analysis.md`.

#### F5. Dual-path background feedback KHÔNG giảm FP (FROZEN) — và train không ổn định

- **Lập luận:** nếu thêm kênh background tin cậy (mask background) vào feedback, FP sẽ
  giảm. Thiết kế ablation 2×2 có negative control.
- **Bằng chứng (FROZEN, đúng thiết kế):** FP **+2.92pp** (p=1.27e-7) — ngược mục tiêu;
  Dice +0.030 (p=0.035, không qua Bonferroni α=0.0083). Negative control (mask background
  ngẫu nhiên) làm Dice sụp 0.239→0.083 → hiệu ứng là do thông tin background THẬT, không
  phải "thêm kênh". Không có synergy giữa dual-path và soft gate.
- **Bằng chứng (TRAIN):** 2/2 lần train dual-path đều xấu hơn hoặc collapse (xem F6/D
  phần bug) — pattern "dual-path làm mất ổn định training" lặp lại.
- **Cơ chế:** checkpoint được train với hard gate nên BatchNorm của `conv1` đã fit phân
  phối feature bị prune cứng; dual-path làm mềm phân phối → BN mismatch (confounder).
  Đó là lý do con số frozen chỉ là "khả năng khai thác của model cũ", không phải giá
  trị cơ chế — nhưng chiều hiệu ứng (FP tăng) vẫn ngược mục tiêu nên prior thấp.
- **Nguồn:** `analysis/abl2x2_eval.py`, `analysis/abl2x2_stats.py` → `logs/abl2x2_log.txt`,
  `results/abl2x2_summary.json`, `results/abl2x2_stats.json`; `kaggle/figures/fig_abl_*.png`;
  `docs/reports/2026-08-31_phase2-3-ablation.md`.

#### F6. Train bằng STE không ủng hộ; chẩn đoán sâu đóng hướng STE vĩnh viễn (TRAIN)

- **Lập luận:** F2 mới là điều kiện cần; phải train end-to-end mới biết giá trị thật.
  Train 2 mô hình cùng seed 42, cùng split: baseline binary vs STE. Sau đó chẩn đoán
  STE bằng checkpoint có sẵn (0 GPU).
- **Bằng chứng (TRAIN, 1 seed):** baseline binary Dice **0.2806** vs STE **0.1951**
  (delta −0.0855, p=0.277, 20/40 ảnh binary thắng — xu hướng rõ, không significant do
  variance). Run mới: val loss STE ≈ binary → "STE hại" không được xác lập tuyệt đối.
- **Bằng chứng chẩn đoán (X4):** STE gradient variance **thấp hơn** soft (13107 vs
  31099) → giả thuyết "STE nhiễu" bị bác; fmask học được chỉ ở encoder (corr ~0.2,
  decoder ~0) → **pattern yếu, không có ích**; BatchNorm dịch chuyển lớn (`d1.r1.bn1`
  mean_abs = 18.76) → refinement trên distribution lệch.
- **Cơ chế:** STE chảy gradient nhưng lượng thông tin học được không đủ để tạo pattern
  ổn định khác mask foreground; BN shift làm suy giảm hiệu quả refinement.
- **Phán quyết:** **ĐÓNG STE vĩnh viễn — "vô hại về gradient, vô dụng"** (theo stopping
  rule đã định trước trong `docs/reports/2026-09-04_next-directions.md`).
- **Nguồn:** `results/phase4_eval.json`, `results/phase4_stats.json`,
  `logs/phase4_eval_log.txt`, `kaggle/figures/fig_p4_*.png`; `analysis/x4_ste_diagnosis.py`
  → `results/x4_ste_diagnosis.json`, `kaggle/figures/fig_x4_*.png`;
  `docs/reports/2026-08-31_phase4-results.md`, `docs/reports/2026-09-07_phase5-x2-gate01.md` (F3).

#### F7. Confidence-gating, dual-path, STE đều bác bỏ → pivot sang loss-side (F3/F5/F6)

- **Lập luận:** 3 hướng "động vào cơ chế feedback" đều thất bại; root cause
  (over-segmentation) chưa từng bị tấn công từ loss; literature ủng hộ wIoU+wBCE →
  pivot ưu tiên **phạt FP từ phía loss** (2 hướng: WSDice faithful; far-weighted
  wIoU+wBCE).
- **Bằng chứng:** xem F3–F6; verify công thức loss từ full-text (CFA-Net/F³Net: trọng
  số biên (1+5μ) — hoá ra nặng BIÊN, sai root cause → **đảo thành far-weighted
  (1+5(1−μ))**; WSDice công thức gốc đã chốt).
- **Cơ chế:** loss-side không đổi kiến trúc → không vướng BN mismatch; trực tiếp phạt
  vùng FP xa biên.
- **Nguồn:** `src/fanet/losses.py`, `scripts/train.py` (`--loss`, `--no-feedback`,
  `--seed`), `notebooks/fanet_kaggle_phase5.py`, `kaggle/fanet-phase5.ipynb`;
  `docs/reports/2026-09-07_phase5-x2-gate01.md` (F1, F2, F4).

### 3.3. Bảng kết quả định lượng chính (kèm nguồn từng số)

| Metric | Giá trị | Loại | Nguồn |
|---|---|---|---|
| Baseline train 53ep (local) | val loss 0.505 | TRAIN | `logs/train_log.txt` |
| Refinement 53ep | F1 0.315→0.327 | FROZEN | `results/test_results.csv` |
| Checkpoint 200ep Kaggle | val loss 0.5415; F1 0.243→0.241 | FROZEN | `results/test_results.csv`, `kaggle/figures/fig4_refinement_10iter.png` |
| **Baseline binary (train E2E, seed 42)** | **Dice 0.2806, IoU 0.1955** | TRAIN | `results/phase4_eval.json` |
| Mô hình STE (train E2E) | Dice 0.1951 | TRAIN | `results/phase4_eval.json` |
| Oracle bound | Dice 0.3312–0.370 | FROZEN | `results/oracle_summary.json`, `results/confgated_summary.json` |
| Confidence-gating | binary 0.2390 > mọi variant | FROZEN | `results/confgated_summary.json` |
| Dual-path ablation | FP +2.92pp (p=1e-7); Dice +0.030 (p=0.035) | FROZEN | `results/abl2x2_stats.json` |
| Negative control | Dice sụp 0.239→0.083 | FROZEN | `results/abl2x2_summary.json` |
| STE chẩn đoán | grad var 13107 vs soft 31099; BN diff 18.76 | TRAIN/analytic | `results/x4_ste_diagnosis.json` |

### 3.4. Các bug phát hiện (INVALID — bài học methodology)

| Bug | Bằng chứng | Cơ chế | Nguồn |
|---|---|---|---|
| Mask-background = 0 khi train | Mô hình dual-path train trùng hệt single-path (max weight diff = 0) | `feedback_tensor` thêm kênh background = zeros → gate `× (1−0)` = giữ nguyên | `notebooks/fanet_kaggle_phase4.py`, `docs/reports/2026-08-31_phase4-results.md` (F1) |
| Phần bù triệt tiêu nhánh attention | Mô hình kết hợp collapse (val loss 1.11–1.13 từ ep 40) | `mask_bg = 1 − mask_fg` → gate = `max(fmask, m_fg) × (1 − (1−m_fg)) = m_fg` → nhánh attention bị vô hiệu | `docs/reports/2026-09-04_next-directions.md` (F1) |
| 6/8 PDF tải nhầm | Trang đầu sai title (arXiv ID sai) | Danh sách ID gốc sai nguồn | `docs/literature_grounding_next.md` (Audit PDF), `papers_not_accessible.md` |

**Bài học methodology:** phải sanity-check weight/behavior khác nhau giữa các biến thể
TRƯỚC khi tin kết quả (đã ghi rõ `docs/reports/2026-08-31_phase4-results.md`).

### 3.5. Cải tiến kỹ thuật đã triển khai (kèm nguồn)

- **MixPool** mở rộng: 3 gate modes (binary/ste/soft) + dual-path — `src/fanet/models/blocks.py`.
- **5 loss classes mới** (`SoftIoU`, `IoUBCE`, `WeightedSoftDiceLoss` faithful, loss
  cell-A kết hợp, `FarWeightedIoUBCELoss`) + `NegativeAreaDiceBCELoss` — `src/fanet/losses.py`.
- **Train script** thêm cờ `--gate`, `--dual-path`, `--loss`, `--no-feedback`, `--seed` — `scripts/train.py`.
- **Pipeline analysis** đầy đủ: `analysis/grad_flow*.py`, `feedback_analysis.py`,
  `confgated_feedback.py`, `oracle_analysis.py`, `abl2x2_*.py`, `phase4_*.py`,
  `x4_ste_diagnosis.py`, `phase5_*.py`.
- **Verification loss từ full-text** CFA-Net/F³Net/WSDice — `docs/reports/2026-09-07_phase5-x2-gate01.md`.

### 3.6. Quan hệ FANet gốc → Phát hiện mới → Giá trị (tóm tắt cho slide)

| FANet gốc | Phát hiện mới | Giá trị mang lại |
|---|---|---|
| Nhánh attention "tự học" | Zero-gradient suốt 53ep (bug non-diff) | Đóng hướng STE bằng chẩn đoán, không claim sai |
| Hard feedback vứt confidence | Uncertainty correlate lỗi, nhưng gating vô dụng | Phát hiện âm tính có giá trị methodology |
| Feedback foreground-only | 84.6% dư địa oracle ở vùng FP; FP cách biên 40.7px | Định vị root cause = over-segmentation |
| DiceBCE thuần | Over-seg chưa từng bị tấn công từ loss | Pivot sang wIoU+wBCE (+far-weighted) |
| Refinement không ổn định | Oracle gap ~0.09 là dư địa thật | Quy trình quyết định 3 bậc dựa trên bằng chứng |
| Claim thiếu kiểm chứng | Nhiều số chỉ là FROZEN/INVALID | Phân loại bằng chứng 3 mức + power analysis |

---

## MỤC 4. Định hướng & Kế hoạch viết Paper (Paper Roadmap & Action Plan)

### 4.1. Điểm đóng góp mới (Novelty / Contributions) — kèm bằng chứng

> **Cảnh báo trung thực:** novelty "background-aware feedback loop" hiện **mỏng**
> (FANetv2 + predictive-coding 2026 cạnh tranh) và bằng chứng thực nghiệm chống lại
> (F5, F6). Không nên đặt cược paper vào hướng này.

| # | Contribution | Mức khả thi | Bằng chứng hiện có | Cần thêm |
|---|---|---|---|---|
| C1 | Negative-result analysis có hệ thống (FP/FN decomposition, BN mismatch, bug-tracking, bằng chứng 3 mức) | **CAO** | F3–F7, mục 3.4 | Khung Metrics Reloaded |
| C2 | Diagnostic finding: nhánh attention FANet non-differentiable → feedback thực chất là hard mask thuần | **CAO** | F1, F2, F6 | — |
| C3 | Loss-side FP penalty (wIoU+wBCE + far-weighted, WSDice) | TRUNG BÌNH (đang chạy) | KL2.3, F7 | Kết quả 3 bậc quyết định (multi-seed) |
| C4 | Oracle-gap định vị dư địa (84.6% FP, cách biên 40.7px) | CAO | F4 | Trình bày như analysis thiết kế |

**Khuyến nghị:** loss-side thành công → "positive method + diagnostic analysis"
(C3 + C1/C2). Thất bại → negative-result paper chuẩn (C1 + C2).

### 4.2. Cấu trúc dàn ý sơ bộ

**Đề xuất tên:** "Diagnosing and Mitigating Over-Segmentation in Feedback-Attention
Polyp Segmentation: A Systematic Analysis of FANet" (tùy kết quả loss-side).

```
1. Introduction
   - Vai trò polyp segmentation; feedback-attention.
   - Khoảng trống: cơ chế feedback-attention chưa được kiểm chứng hoạt động thực tế
     (gradient, confidence, background); over-segmentation chưa được xử lý từ loss.

2. Related Work
   - Feedback/recurrent segmentation (FANet, FANetv2, FEGNet, RefineU-Net, CFL, predictive-coding 2026).
   - Background/reverse attention (PraNet, UACANet, CaraNet).
   - Loss functions (Dice family, wIoU+wBCE, WSDice, Lovász, Tversky, boundary).
   - Metrics & validation (Metrics Reloaded).

3. Method
   3.1 FANet baseline & MixPool (mô tả + điểm non-differentiable).
   3.2 Diagnostic framework (phân loại bằng chứng TRAIN/FROZEN/INVALID).
   3.3 Cải tiến (loss-side FP penalty — nếu pass; hoặc khung negative-result).

4. Experiments
   4.1 Setup (dataset, split, metrics, statistics, power).
   4.2 Chẩn đoán cơ chế (gradient flow, uncertainty, oracle gap, help/hurt).
   4.3 Kiểm định giả thuyết (confidence-gating bác bỏ; dual-path; STE).
   4.4 Loss-side FP penalty (no-feedback vs có-feedback; screening; multi-seed).

5. Results & Discussion
   - Bảng số liệu chính + thống kê paired (CI, effect size).
   - Vì sao feedback mask-at-input không hiệu quả; đâu là dư địa thật.
   - Hạn chế (1 split, n=40, 1–3 seeds).

6. Conclusion + Future Work.

References (chuẩn DOI, từ literature_grounding_next.md)
Appendix: bug-tracking, reproducibility (scripts, seeds).
```

### 4.3. Các công việc / thử nghiệm cần làm tiếp theo (Future Work)

| Ưu tiên | Công việc | Điều kiện / Stopping rule | Effort | Nguồn |
|---|---|---|---|---|
| 1 | **Chạy Phase 5 (loss-side) trên Kaggle**: 3 cells (no-feedback + 2 loss mới WSDice kết hợp / far-weighted), seed 43 | User submit `kaggle/fanet-phase5.ipynb` trên GPU T4 (3 cells × 200 epochs) | ~7 GPU-h | `notebooks/fanet_kaggle_phase5.py` |
| 2 | Đánh giá → phán quyết bậc 1 (no-feedback vs baseline) + bậc 2 (loss mới) | Bậc 1: nếu no-feedback đạt Dice ≥ hoặc FP ≤ baseline binary → feedback là gánh nặng → pivot; Bậc 2: FP giảm ≥1pp, Dice ≥ baseline −0.02, FN ≤ +2pp | — | `analysis/phase5_eval.py` |
| 3 | Multi-seed (3 seeds) nếu bậc 2 pass | Effect cùng dấu ≥2/3 seeds, Wilcoxon per-seed, Bonferroni | ~14 GPU-h | — |
| 4 | Nếu feedback là gánh nặng → pivot no-feedback / feedback ở skip (soft gate + deep-sup kiểu FEGNet) | Bậc 1 cho thấy | ~7 GPU-h | report 09/05 F2 |
| 5 | Multi-dataset (full Kvasir-SEG, CVC-ClinicDB, DRIVE, CHASE) | Sau khi chốt phương pháp | — | — |
| 6 | Hoàn thiện khung negative-result + reproducibility appendix | Song song | — | — |
| 7 | Novelty re-check (Semantic Scholar với key, Google Scholar) | Bổ sung coverage thiếu | — | report 09/04 |

**Đã chốt bỏ/trì hoãn:** STE (đóng vĩnh viễn, F6); dual-path background tường minh
(prior thấp, F5); boundary-loss family (nhắm biên, không chữa FP xa biên); prediction-error
feedback kiểu predictive-coding (chi phí cao).

### 4.4. Hướng dẫn tái lập nhanh các con số chính (cho reviewer/co-author)

| Con số | Lệnh / cách tái lập |
|---|---|
| fmask zero-grad (F1) | `python analysis/grad_flow.py --config configs/kvasir_sessile.yaml` |
| Gradient STE/soft (F2) | `python analysis/grad_flow_gates.py` |
| Help/hurt + uncertainty (F3) | `python analysis/feedback_analysis.py` |
| Confidence-gating (F3) | `python analysis/confgated_feedback.py` |
| Oracle/background (F4) | `python analysis/oracle_analysis.py` |
| Ablation 2×2 (F5) | `python analysis/abl2x2_eval.py && python analysis/abl2x2_stats.py` |
| Train E2E (F6) | `python scripts/train.py --config ... --gate binary` (notebook `kaggle/fanet-phase4.ipynb`) |
| STE diagnosis (F6) | `python analysis/x4_ste_diagnosis.py` |
| Phase 5 loss-side (F7) | `python scripts/train.py --loss cella|farwiou --no-feedback --seed 43` + `python analysis/phase5_eval.py` |

---

## Kết luận tổng thể (Executive Summary)

Hành trình nghiên cứu đi trọn vòng lặp **reproduce → gap → hypothesis → experiment →
verdict → pivot**, mỗi bước đều có bằng chứng gắn artifact:
1. **Reproduce FANet**, xác nhận 2 điểm yếu cấu trúc (nhánh attention chết do
   non-differentiable; hard threshold vứt confidence + background) — F1, F3.
2. **Định vị root cause** = over-segmentation vùng nền xa biên (FP/FN=2.48; 84.6% dư
   địa oracle ở FP; cách biên 40.7px) — F4.
3. **Kiểm định có hệ thống** 3 hướng cải tiến: confidence-gating (bác bỏ), dual-path
   (bác bỏ/không ổn định), STE (đóng vĩnh viễn) — F3, F5, F6.
4. **Pivot sang loss-side FP penalty**: code + smoke-test đã xong (Phase 9), **đang chờ
   chạy Kaggle** (Phase 10) — lỗ hổng lớn nhất còn bỏ ngỏ, literature ủng hộ rộng,
   không vướng confounder BatchNorm — KL2.3, F7.
5. **Chiến lược paper linh hoạt**: positive method nếu loss-side thành công; negative-result
   paper chuẩn nếu thất bại — cả hai publish được nhờ khung methodology chặt chẽ.

**Giá trị cốt lõi:** không phải một cơ chế "hack" số, mà là **khung chẩn đoán và kiểm
định khoa học** xác định đúng vị trí điểm yếu của feedback-attention và cách xử lý
đúng — nâng chuẩn reproducibility/negative-result trong cộng đồng polyp segmentation.

---

## Đính kèm (Sources)

- Papers: `papers/original/`, `papers/translated/`; literature: `docs/literature_grounding_*.md`
- Guide kiến trúc: `docs/FANet_Complete_Guide.md`
- Reports theo phase: `docs/reports/` (index: `docs/reports/README.md`)
- Hypotheses & design: `docs/hypotheses_*.md`, `docs/experiment_design_*.md`
- Code: `src/fanet/`, `scripts/`, `analysis/`, `notebooks/`, `kaggle/`
- Logs & results: `logs/`, `results/`, `checkpoints/`, `checkpoints_phase4/`
- Figures: `kaggle/figures/`