# Báo cáo Tổng quan Nghiên cứu (Comprehensive Research Overview Report)

**Dự án:** FANet — Reproduction + Cải tiến cho phân đoạn ảnh y sinh (Polyp Segmentation)
**Ngày:** 08/09/2026
**Vai trò:** Research Assistant / Co-author overview
**Mục đích:** Tổng hợp toàn bộ hành trình nghiên cứu (reproduce → tìm research gap →
phân tích → thử nghiệm → định hướng paper) để phục vụ slide presentation và lập
kế hoạch viết bài báo khoa học.

> **Ghi chú về độ tin cậy bằng chứng:** theo quy ước repo, mọi con số được phân
> loại thành 3 mức — **TRAIN** (kết quả từ mô hình train end-to-end, đáng tin nhưng
> thường 1 seed/1 split), **FROZEN EVAL** (đo trên model cũ, chỉ phản ánh "khả năng
> khai thác feedback của model hiện có", KHÔNG phải giá trị đầy đủ của cơ chế), và
> **INVALID** (do bug, không dùng cho bất kỳ claim nào). Báo cáo này gắn nhãn tương
> ứng cho từng con số quan trọng.

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
| Task | Phân đoạn ảnh y sinh (biomedical image segmentation), tiêu biểu là polyp segmentation |

### 1.2. Bài toán cốt lõi (Core Problem)

Phân đoạn pixel của các cấu trúc y sinh (polyp trong nội soi đại tràng, mạch máu
võng mạc, tổn thương da...) từ ảnh y khoa. Đây là **binary semantic segmentation**
(mỗi pixel ∈ {foreground, background}).

### 1.3. Động lực & Ý tưởng chính

**Động lực:** Trong huấn luyện deep learning, mask dự đoán ở epoch $t$ bị **vứt bỏ
hoàn toàn** khi sang epoch $t+1$ — không có "ký ức" về những gì mô hình đã học ở dạng
mask. FANet cho rằng thông tin này quý giá và đáng được tái sử dụng.

**Ý tưởng chính (Feedback Attention):**
- Đưa **mask dự đoán của epoch trước** làm **thêm một kênh input** (hard attention map).
- Qua module **MixPool** (feedback attention), dùng mask này để **prune (tỉa)** các
  feature không liên quan đến foreground, hướng sự chú ý của mô hình vào vùng polyp.
- Tạo thành **vòng phản hồi xuyên epoch** (cross-epoch feedback loop) và **test-time
  iterative refinement** (tinh chỉnh lặp khi suy luận).

### 1.4. Kiến trúc

- U-Net encoder-decoder 4 tầng (3→32→64→128→256), ~7.72M params.
- Input: cặp `[RGB image, previous mask]`.
- **MixPool** (trái tim của FANet): kết hợp attention mask (hard binary) với 50%
  feature gốc làm "safety net":
  ```python
  fmask = (self.fmask(x) > 0.5).float()               # nhánh attention "tự học" (nhưng cứng)
  x1 = x * logical_or(fmask, m > 0).float()           # gate binary OR với mask feedback
  x2 = self.conv2(x)                                  # 50% feature gốc
  x = concat([x1, x2], 1)
  ```
- Loss: **DiceBCELoss** (0.5·BCE + 0.5·Dice).
- Training: mask khởi tạo bằng Otsu, nén RLE, chỉ cập nhật mask khi val loss cải thiện.

### 1.5. Ưu điểm (Strengths)

| # | Ưu điểm | Cơ sở |
|---|---|---|
| 1 | **Ý tưởng feedback attention mới lạ, trực quan** | Tái sử dụng mask cũ như hard attention — hình thức hiếm, tạo tiền đề cho loạt work theo sau (FANetv2, FEGNet...) |
| 2 | **Kiến trúc nhẹ, khả thi triển khai** | ~7.7M params, chạy được cả trên CPU; phù hợp cài đặt lâm sàng |
| 3 | **Có cơ chế tinh chỉnh lúc test (iterative refinement)** | Khi checkpoint tốt, refinement giúp cải thiện Dice nhẹ (F1 0.315→0.327, checkpoint 53ep, FROZEN) |
| 4 | **Design "safety net" 50% feature gốc** | Giảm rủi ro mất hoàn toàn feature khi gate sai |
| 5 | **Được cộng đồng công nhận** | Công bố TNNLS, trở thành baseline phổ biến cho polyp segmentation |

### 1.6. Điểm yếu / Hạn chế (Limitations) — đã xác nhận bằng dữ liệu

**Đây là phần then chốt của dự án — mọi hướng cải tiến sau này đều xuất phát từ đây.**

| # | Điểm yếu | Bằng chứng (độ tin) |
|---|---|---|
| 1 | **Gate binary cắt gradient → nhánh "attention tự học" `fmask` KHÔNG BAO GIỜ được học** | `(fmask > 0.5)` không khả vi → 0/160 tham số fmask có gradient sau 10 steps; BN affine diff = **chính xác 0.000** qua 53 epochs → nhánh attention thực chất là filter random-init; toàn bộ sức mạnh MixPool đến từ mask feedback (TRAIN/analytic) |
| 2 | **Hard threshold vứt bỏ confidence/uncertainty** | Error rate giảm monotonic theo confidence: 48.7% (conf < 0.1) → 2.7% (conf > 0.35). Threshold 0.5 đối xử pixel sai 48.7% ngang hàng pixel sai 2.7% (TRAIN) |
| 3 | **Feedback chỉ chứa foreground, background bị zero-hoá** | Oracle khác prediction feedback tới **84.6% ở vùng prediction-FP**; model over-segment nặng (FP/FN = **2.48**), FP trải sâu **40.7px** xa biên (FROZEN) |
| 4 | **DiceBCELoss thuần không phân biệt vùng thừa** | Root cause (over-segmentation) **chưa từng** được can thiệp từ phía loss |
| 5 | **Test-time refinement phụ thuộc chất lượng checkpoint** | Checkpoint 200ep: F1 0.243→0.241 (plateau, không giúp); 53ep: F1 0.315→0.327 (giúp) — kết quả không nhất quán (FROZEN) |

**Tóm tắt bản chất:** "Feature-based attention" mà paper quảng cáo **không hoạt động
như quảng cáo** (nhánh tự học chết do non-differentiable), còn "feedback" chỉ là một
mask nhị phân foreground-only vứt bỏ thông tin confidence lẫn background.

---

## MỤC 2. Tổng quan các Nghiên cứu & Paper liên quan (Related Works)

### 2.1. Bối cảnh literature

Dự án đã thực hiện **literature grounding đa nguồn** (OpenAlex, arXiv, Europe PMC,
Semantic Scholar, tải tay): 41 ứng viên → **32 paper curated**, 29+ PDF tải về, đọc
full-text 7 paper quan trọng (FEGNet, RefineU-Net, CBL, BCNet, CTNet, CFA-Net, BUNet).

### 2.2. Đột phá chính của các paper liên quan/đối chứng

| Nhóm | Paper (venue, năm) | Đột phá chính |
|---|---|---|
| **Feedback mask loop** | **FANetv2** (ICASSP 2025, cùng lab) | Mở rộng FANet: iterative mask feedback + text-guided + refinement lúc test — **chiếm không gian "iterative mask feedback" của chính FANet** |
| **Feedback prediction-error** | **Predictive coding polyp** (Research Square 2026) | Top-down prediction-error feedback lúc inference; đo thẳng FPR/FDR + FP-trên-nếp-gấp — **"dọn FP qua feedback" là bài toán mở 2026** |
| **Feature-level feedback** | **FEGNet** (JBHI 2023), **RefineU-Net** (PRL 2020) | Feedback = feature/attention **nội mạng** (soft gate, deep supervision từng step, đặt ở skip connections) — **cố ý TRÁNH mask-at-input của FANet** |
| **Background / reverse attention** | **PraNet** (MICCAI 2020), **UACANet** (ACM MM 2021) | Reverse attention (1−pred) "dọn" over-segmentation; tách m_f/m_b/m_u + background context pool — tiền lệ trực tiếp của kênh background |
| **Loss-side (IoU/weighted)** | **WSDice** (IEEE Access 2020), **Lovász** (CVPR 2018), **Tversky** (2017) | Phạt FP trực tiếp trong loss (surrogate IoU, trọng số vùng thừa) — **5/7 paper polyp vừa đọc dùng wIoU+wBCE** |
| **Boundary / edge** | **CFA-Net** (PR 2023), **BCNet** (JBHI 2022), **MEGANet** (WACV 2024), **BUNet** (Neural Networks 2024) | Boundary-weighted wIoU+wBCE, dual output area+boundary, edge-guided attention |
| **Methodology** | **Metrics Reloaded** (Nature Methods 2024) | Chuẩn metric/CI/report cho validation |

### 2.3. So sánh ngắn gọn FANet vs các nghiên cứu liên quan

| Khía cạnh | FANet (gốc) | FEGNet / RefineU-Net | PraNet / UACANet | Predictive-coding 2026 | WSDice / wIoU-family |
|---|---|---|---|---|---|
| **Loại feedback** | Mask prediction **ở input**, hard | Feature/attention **nội mạng**, soft | Single-pass (không loop) | Prediction-error ở inference | Không feedback |
| **Gate** | Hard threshold (non-diff) | Soft sigmoid (grad luôn chảy) | Soft attention | — | — |
| **Giám sát** | Chỉ loss cuối | Deep supervision từng time-step | Reverse attention | — | — |
| **Background** | Bị zero-hoá | Attention tự học xen kẽ bg | m_b context rõ ràng | Prediction error ngầm mang bg | Trọng số phạt FP |
| **Loss** | DiceBCE thuần | wIoU+wBCE | wIoU+wBCE-ish | — | wIoU+wBCE, WSDice |
| **Góc tấn công over-seg** | Không trực tiếp | — | Reverse attention | Feedback bg | **FP-penalty trong loss** |

**Kết luận đối chiếu quan trọng:**
1. **Không paper nào có kênh background tường minh trong mask feedback loop** → novelty
   hình thức của hướng dual-path còn nhưng **mỏng**, và bị cạnh tranh trực tiếp bởi
   FANetv2 (cùng lab) + predictive-coding 2026.
2. Các work feedback **thành công** (FEGNet, RefineU-Net) đều làm **feature-level,
   soft gate, deep supervision, ở skip** — chính xác là những thứ FANet gốc **thiếu**.
   FEGNet còn trích dẫn thẳng FANet và cố ý bỏ mask-at-input ("rather than merely
   connecting the input and output").
3. **Lỗ hổng lớn nhất của plan FANet-gốc**: root cause (over-segmentation) chưa từng
   được tấn công từ phía **loss** — đây là hướng literature ủng hộ mạnh nhất (5/7
   paper dùng wIoU+wBCE trên chính Kvasir-SEG).

---

## MỤC 3. Tổng hợp công việc đã thực hiện & Kết quả (Current Work & Findings)

### 3.1. Timeline thực hiện

| Phase | Ngày | Nội dung |
|---|---|---|
| Setup + reproduce | 10/08 | Setup env, chạy FANet trên sessile-Kvasir-SEG (196 ảnh: 156/40), val loss 0.505 |
| Gap analysis | 20/08 | Đọc sâu UACANet, xác định 6 gap của MixPool, phát hiện fmask zero-grad |
| Feedback help/hurt | 24/08 | Xác nhận 2 điểm yếu (fmask không học; hard threshold vứt confidence) |
| Confidence-gating | 27/08 | Checkpoint 200ep Kaggle; **bác bỏ** confidence-gating |
| Oracle/background | 31/08 | Xác nhận giả thuyết background (84.6% khác biệt ở vùng FP; FP/FN=2.48) |
| Phase 2+3 ablation | 31/08 | STE/soft hồi sinh fmask; dual-path 2x2; negative control |
| Phase 4 train E2E | 31/08→09/04 | Train 4 cells trên Kaggle T4; phát hiện + sửa bug dual-path |
| Next directions | 04/09 | Audit bằng chứng 3 mức, literature 25 paper, xếp hạng X1–X6 |
| New papers review | 05/09 | Đọc 7 paper full-text; pivot loss sang wIoU+wBCE; audit 38 PDF |
| Phase 5 X2/X4 | 07/09 | Verify WSDice/CFA-Net công thức; đóng STE (X4); chuẩn bị Gate 0/1 |

### 3.2. Các kết quả đạt được (kèm số liệu & đánh giá)

#### Kết quả A — Chẩn đoán cơ chế (đáng tin, nền tảng)

| # | Kết quả | Số liệu | Loại bằng chứng | Ý nghĩa |
|---|---|---|---|---|
| A1 | Gate binary cắt gradient hoàn toàn | fmask: 0/160 params có grad; BN affine diff **0.000** | TRAIN/analytic | Nhánh attention "tự học" thực chất chết |
| A2 | STE/soft hồi sinh gradient | binary=0, **STE=0.125, soft=0.106** (8 MixPool) | TRAIN/analytic | Điều kiện cần để hồi sinh fmask đạt được |
| A3 | Uncertainty correlate mạnh với lỗi | err 48.7% (conf<0.1) vs 2.7% (conf>0.35); FP tập trung conf thấp | TRAIN | Cơ sở cho giả thuyết confidence-gating |

#### Kết quả B — Các giả thuyết đã kiểm định (FROZEN + TRAIN)

| # | Giả thuyết | Phán quyết | Số liệu | Loại bằng chứng |
|---|---|---|---|---|
| B1 | Confidence-gating feedback giảm hurt/tăng dice | **BÁC BỎ** | binary 0.2390 > mọi biến thể confidence-based; cắt conf không chuyển hurt→help | FROZEN (200ep) |
| B2 | Feedback giúp khi mask trước tốt, hại khi kém | XÁC NHẬN | 65 giúp/46 hại; corr(prev_dice)=+0.355; mean delta +0.031 | FROZEN |
| B3 | Dual-path giảm over-segmentation (P_B1) | **BÁC BỎ** | FP **+2.92pp** (p=1.27e-7); Dice +0.030 (p=0.035, không qua Bonferroni) | FROZEN (BN mismatch confounder) |
| B4 | Negative control m_bg ngẫu nhiên | XÁC NHẬN | Dice sụp 0.239→0.083 → hiệu ứng là do thông tin bg thật | FROZEN |
| B5 | STE train ≥ binary train (T_A) | **KHÔNG ỦNG HỘ** | binary 0.2806 vs STE 0.1951 (delta -0.0855, p=0.277) | TRAIN (1 seed) |
| B6 | Oracle feedback tốt hơn nhờ thông tin background | XÁC NHẬN | 84.6% khác biệt ở vùng FP; oracle gap +0.093 ổn định | FROZEN |

#### Kết quả C — Kết quả định lượng chính

| Metric | Giá trị | Ghi chú |
|---|---|---|
| Baseline train (53ep, local) | val loss 0.505 | Chưa hết 500ep theo paper |
| Baseline refinement 53ep | F1 0.315→0.327 | Feedback giúp nhẹ |
| Checkpoint 200ep Kaggle | val loss 0.5415; F1 0.243→0.241 (plateau) | Feedback gần như không giúp |
| **T00 (binary, trained, seed42)** | **Dice 0.2806, IoU 0.1955** | Baseline mới chuẩn (TRAIN) |
| T10 (STE, trained) | Dice 0.1951 | Kém hơn T00 |
| Oracle bound | Dice 0.3312–0.370 | Upper bound / dư địa khai thác |

#### Kết quả D — Các bug phát hiện (INVALID, quan trọng về methodology)

| Bug | Mô tả | Hệ quả |
|---|---|---|
| m_bg=zeros khi train | T01==T00, T11==T10 (max weight diff=0) | Phase 4 run 1 dual-path INVALID; đã sửa m_bg=1−m_fg |
| complement triệt tiêu fmask | m_bg=1−m_fg → keep=max(fmask,m_fg)·(1−m_bg)=m_fg | T01/T11 run mới INVALID; T11 collapse từ ep40 |
| 6/8 PDF tải nhầm | arXiv ID sai trong danh sách gốc | Sửa 6, xóa 2 — đảm bảo chỉ dùng PDF đúng |

### 3.3. Những phát hiện chính (Key Findings) và cải tiến đã thực hiện

**Quan hệ: FANet gốc → Phát hiện mới → Giá trị mang lại**

| FANet gốc | Phát hiện mới của dự án | Cải tiến / Giá trị |
|---|---|---|
| Nhánh attention `fmask` quảng cáo "tự học" | Thực tế **zero-gradient suốt 53 epochs** (bug non-differentiable thật sự) | Chẩn đoán STE/soft → **đóng STE vĩnh viễn** sau X4 ("vô hại-vô dụng"); tránh claim sai, tiết kiệm GPU |
| Hard feedback vứt confidence | Uncertainty correlate mạnh với lỗi (48.7% vs 2.7%) | Kiểm định confidence-gating → **bác bỏ**; kết luận "lỗi và thông tin hữu ích nằm cùng chỗ (biên)" — **phát hiện âm tính có giá trị methodology** |
| Feedback foreground-only | **84.6% dư địa oracle nằm ở vùng background/FP**; FP/FN=2.48; FP cách biên 40.7px | Định vị chính xác root cause = **over-segmentation vùng nền xa biên**; nền tảng cho mọi hướng sau |
| DiceBCELoss thuần | Root cause over-seg **chưa từng** bị tấn công từ loss | **Pivot sang loss-side FP penalty (X2)**: wIoU+wBCE (+WSDice/far-weighted), literature ủng hộ mạnh |
| Test-time refinement không ổn định | Phụ thuộc checkpoint; oracle gap ~0.09 là dư địa thật | Thiết lập protocol Gate 0/1/2 (no-feedback → screening → multi-seed) để quyết định dựa trên bằng chứng |
| Bất kỳ claim nào | Nhiều con số chỉ là FROZEN/INVALID | Thiết lập **hệ phân loại bằng chứng 3 mức** + đánh giá sức mạnh thống kê (δ≈0.13 với n=40) — nâng chuẩn khoa học cho paper |

**Cải tiến kỹ thuật đã triển khai trong repo:**
- Mở rộng MixPool với 3 gate modes (binary/ste/soft) + dual-path feedback `[m_fg, m_bg]`.
- Bổ sung 5 loss classes mới (`SoftIoU`, `IoUBCE`, `WeightedSoftDiceLoss` faithful,
  `Phase5CellALoss`, `FarWeightedIoUBCELoss`) + `NegativeAreaDiceBCELoss`.
- `scripts/train.py`: thêm `--gate`, `--dual-path`, `--loss`, `--no-feedback`, `--seed`.
- Pipeline analysis đầy đủ (grad flow, help/hurt, conf-gating, oracle, ablation, stats paired).
- Quy trình verification công thức loss từ full-text paper (CFA-Net/F³Net/WSDice).

---

## MỤC 4. Định hướng & Kế hoạch viết Paper (Paper Roadmap & Action Plan)

### 4.1. Điểm đóng góp mới (Novelty / Contributions) đề xuất

> **Cảnh báo trung thực (từ audit novelty 04–05/09):** novelty của "background-aware
> feedback loop" (kênh m_bg tường minh) hiện **mỏng** — bị FANetv2 (cùng lab) và
> predictive-coding 2026 cạnh tranh, và bằng chứng thực nghiệm (Phase 3 FP↑ p=1e-7;
> Phase 4 dual xấu/collapse) không ủng hộ giá trị thực tiễn. **Không nên đặt cược
> paper vào novelty này.**

Các novelty khả dĩ, xếp theo độ khả thi hiện tại:

| # | Contribution | Mức khả thi | Cần gì |
|---|---|---|---|
| C1 | **Phân tích negative-result có hệ thống cho feedback-attention segmentation** (FP/FN decomposition, BN-mismatch confounder, bug-tracking, phân loại bằng chứng 3 mức) | **CAO** (đã có toàn bộ dữ liệu) | Khung Metrics Reloaded; viết đúng chuẩn negative-result |
| C2 | **Diagnostic finding: nhánh attention "tự học" của FANet bị non-differentiable (zero-gradient) → cơ chế feedback thực chất là hard mask thuần** | **CAO** (bằng chứng TRAIN chắc chắn) | Đây là phát hiện mới, có thể publish như reproducibility/analysis |
| C3 | **Loss-side FP penalty cho polyp over-segmentation** (wIoU+wBCE + far-weighted boundary, WSDice) — nếu X2 Gate 1/2 pass | TRUNG BÌNH (đang chạy) | Kết quả Gate 0/1/2 (multi-seed) |
| C4 | **Đánh giá có đối chứng oracle-gap về vị trí dư địa (84.6% ở vùng FP, cách biên 40.7px)** | CAO | Đã có; trình bày như phân tích định hướng thiết kế |

**Khuyến nghị chiến lược:** nếu X2 (loss-side) thành công → bài báo có "positive
method + diagnostic analysis" (C3 + C1/C2). Nếu X2 thất bại → chuyển sang **negative-result
paper** chuẩn (C1 + C2) vẫn publish được nếu phân tích/bug-tracking chuẩn mực.

### 4.2. Cấu trúc dàn ý sơ bộ (draft outline)

**Đề xuất tên:** "Diagnosing and Mitigating Over-Segmentation in Feedback-Attention
Polyp Segmentation: A Systematic Analysis of FANet" (hoặc tương tự, tùy kết quả X2).

```
1. Introduction
   - Vai trò polyp segmentation; feedback-attention là hướng quan trọng.
   - Khoảng trống: cơ chế feedback-attention chưa được kiểm chứng sâu về mặt
     hoạt động thực tế (gradient, confidence, background); over-segmentation chưa
     được xử lý từ phía loss.

2. Related Work
   - Feedback / recurrent segmentation (FANet, FANetv2, FEGNet, RefineU-Net,
     Context Feedback Loop, predictive-coding 2026).
   - Background / reverse attention (PraNet, UACANet, CaraNet).
   - Loss functions (Dice family, wIoU+wBCE, WSDice, Lovász, Tversky, boundary).
   - Metrics & validation (Metrics Reloaded).

3. Method
   3.1 FANet baseline & MixPool (mô tả lại + điểm non-differentiable).
   3.2 Diagnostic framework: phân loại bằng chứng TRAIN/FROZEN/INVALID.
   3.3 Đề xuất cải tiến (loss-side FP penalty — nếu X2 pass; hoặc khung phân tích).

4. Experiments
   4.1 Setup (dataset, split, metrics, statistics, power).
   4.2 Chẩn đoán cơ chế (gradient flow, uncertainty, oracle gap, help/hurt).
   4.3 Kiểm định giả thuyết (confidence-gating bác bỏ; dual-path; STE).
   4.4 Loss-side FP penalty (Gate 0/1/2 — kết quả X2).

5. Results & Discussion
   - Bảng số liệu chính + phân tích thống kê (paired, CI, effect size).
   - Thảo luận: vì sao feedback mask-at-input không hiệu quả; đâu là dư địa thật.
   - Hạn chế (1 split, n=40, 1–3 seeds).

6. Conclusion
   - Kết luận + Future Work.

References (chuẩn DOI, từ literature_grounding_next.md)
Appendix: bug-tracking, reproducibility (scripts/analysis, seeds).
```

### 4.3. Các công việc / thử nghiệm cần làm tiếp theo (Future Work)

**Trạng thái hiện tại (Gate 0/1 của X2 đang chờ chạy Kaggle):**

| Ưu tiên | Công việc | Điều kiện / Stopping rule | Effort |
|---|---|---|---|
| **1** | Chạy Phase 5 Gate 0 (T0N no-feedback) + Gate 1 (TA WSDice / TB far-weighted), seed 43 | User submit `kaggle/fanet-phase5.ipynb` trên T4 | ~7 GPU-h |
| **2** | Chạy `analysis/phase5_eval.py` → Gate 0/1 verdict | Gate 0: nếu Dice(T0N)≥T00 hoặc FP(T0N)≤T00 → feedback là liability → pivot; Gate 1: FP(winner)<FP(T00)−0.01 và Dice≥T00−0.02 và FN≤+0.02 | — |
| **3** | Gate 2 multi-seed (44,45) nếu Gate 1 pass | Chỉ khi Gate 1 pass; effect cùng dấu ≥2/3 seeds, Wilcoxon per-seed Bonferroni | ~14 GPU-h |
| **4** | Nếu feedback là liability → pivot no-feedback / skip-feedback (FEGNet-style soft gate + deep supervision ở skip) | Gate 0 cho thấy điều đó | ~7 GPU-h |
| **5** | Multi-dataset (full Kvasir-SEG, CVC-ClinicDB, DRIVE, CHASE) | Sau khi chốt phương pháp | — |
| **6** | Hoàn thiện khung negative-result + reproducibility appendix | Song song; độc lập kết quả X2 | — |
| **7** | Novelty re-check (Semantic Scholar với key, Google Scholar) | Bổ sung coverage còn thiếu (S2 429, PubMed chặn) | — |

**Các hướng đã chốt bỏ/trì hoãn (để tiết kiệm GPU):**
- **STE (X4): ĐÓNG vĩnh viễn** — "vô hại về gradient nhưng vô dụng" (fmask chỉ học
  encoder corr ~0.2, decoder ~0, BN shift lớn d1.r1.bn1=18.8, không có Dice benefit).
- **Dual-path background feedback (X1): prior thấp** — 2/2 run train xấu/collapse +
  frozen FP↑ p=1e-7; chỉ làm nếu X2 thất bại và cần đóng novelty.
- **CBL/boundary loss: KHÔNG vào Gate 1** — nhắm biên, không chữa FP xa biên 40.7px.
- **X6 (prediction-error feedback theo predictive-coding 2026): trì hoãn** — chi phí cao.

---

## Kết luận tổng thể (Executive Summary)

Hành trình nghiên cứu đã đi qua trọn vẹn vòng lặp **reproduce → gap → hypothesis →
experiment → verdict → pivot**:
1. **Reproduce FANet** và **phát hiện 2 điểm yếu cấu trúc** (nhánh attention chết do
   non-differentiable; hard threshold vứt confidence + background).
2. **Định vị root cause** bằng oracle analysis: over-segmentation vùng nền xa biên
   (FP/FN=2.48; 84.6% dư địa oracle ở FP; cách biên 40.7px).
3. **Kiểm định có hệ thống** các giả thuyết cải tiến: confidence-gating bác bỏ,
   dual-path bác bỏ (frozen) / không ổn định (train), STE đóng vĩnh viễn — mỗi cái
   đều có stopping rule rõ ràng.
4. **Pivot sang hướng mạnh nhất còn bỏ ngỏ**: loss-side FP penalty (X2, Gate 0/1
   đang chờ chạy) — lỗ hổng lớn nhất của plan A+B, literature ủng hộ rộng (5/7
   paper dùng wIoU+wBCE), không vướng confounder BN.
5. **Chiến lược paper linh hoạt**: thành công → positive method; thất bại →
   negative-result paper chuẩn — **cả hai đều publish được** nhờ phân tích/audit
   methodology chặt chẽ (phân loại bằng chứng 3 mức, bug-tracking, power analysis).

**Giá trị cốt lõi mang lại so với FANet gốc & related works:** không phải một cơ chế
mới "hack" số, mà là **một khung chẩn đoán và kiểm định khoa học** cho thấy đúng vị
trí điểm yếu của feedback-attention và đúng cách xử lý — góp phần nâng chuẩn
reproducibility/negative-result trong cộng đồng polyp segmentation.

---

## Đính kèm (Sources)

- Papers gốc & liên quan: `papers/original/`, `papers/translated/`, `docs/literature_grounding_next.md`
- Guide kiến trúc FANet: `docs/FANet_Complete_Guide.md`
- Reports theo phase: `docs/reports/` (index: `docs/reports/README.md`)
- Hypotheses & design: `docs/hypotheses_*.md`, `docs/experiment_design_*.md`
- Code: `src/fanet/`, `scripts/`, `analysis/`, `notebooks/`, `kaggle/`
- Logs & results: `logs/`, `results/`, `checkpoints*/`
