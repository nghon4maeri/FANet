# Literature Grounding — Next Directions (04/09/2026)

Search date: 04/09/2026. Nguồn: OpenAlex (33 queries title.search/search),
arXiv API (8 queries), Europe PMC (5 queries), Semantic Scholar (bị 429 — thất
bại), PubMed e-utilities (bị NCBI chặn IP — thất bại). 41 ứng viên → chọn 25
paper cho bảng chính. PDF: 29 paper đã tải về `docs/` (log:
`docs/paper_download_log.json`; không tải được: `papers_not_accessible.md` ở
root repo).

## Bảng papers liên quan (25 curated)

| # | Paper | Venue | Năm | DOI | Mức liên quan | Ý tưởng áp dụng cho FANet |
|---|---|---|---|---|---|---|
| 1 | FANet: A Feedback Attention Network for Improved Biomedical Image Segmentation | IEEE TNNLS | 2022 | 10.1109/tnnls.2022.3159394 | GỐC | Baseline; 2 điểm yếu đã xác nhận (fmask zero-grad, feedback fg-only) |
| 2 | Transformer-Enhanced Iterative Feedback Mechanism for Polyp Segmentation (FANetv2) | IEEE ICASSP | 2025 | 10.1109/icassp49660.2025.10890567 (arXiv:2409.05875) | CAO | Cùng lab FANet: iterative mask feedback + text-guided + refinement lúc test; feedback VẪN fg-only → novelty m_bg còn nhưng bị cạnh tranh trực tiếp |
| 3 | A Cortically-Inspired Predictive Coding Framework for Polyp Segmentation | Research Square preprint | 2026 | 10.21203/rs.3.rs-9145958/v1 | CAO | Top-down prediction-error feedback lúc inference; đo trực tiếp FPR/FDR + FP-trên-nếp-gấp — "dọn FP qua feedback" là bài toán mở 2026; đối thủ novelty gần nhất |
| 4 | FEGNet: A Feedback Enhancement Gate Network for Automatic Polyp Segmentation | IEEE JBHI | 2023 | 10.1109/jbhi.2023.3272168 | TB–CAO | Feedback refinement gate nội mạng, không thêm tham số; gợi ý đặt feedback ở decoder thay vì toàn bộ (→ X3) |
| 5 | Learning With Context Feedback Loop for Robust Medical Image Segmentation | IEEE TMI | 2021 | 10.1109/tmi.2021.3060497 | TB | Feedback = context features (không phải mask bg) — dùng để phân biệt claim của mình |
| 6 | F³Net: Fusion, Feedback and Focus for Salient Object Detection | AAAI | 2020 | 10.1609/aaai.v34i07.6916 | TB | Cascaded feedback decoder; mẫu ablation feedback đa tầng |
| 7 | PraNet: Parallel Reverse Attention Network for Polyp Segmentation | MICCAI | 2020 | 10.1007/978-3-030-59725-2_26 | CAO | Reverse attention (vùng 1−pred) "dọn" over-segmentation — tiền lệ trực tiếp của m_bg; single-pass |
| 8 | UACANet: Uncertainty Augmented Context Attention for Polyp Segmentation | ACM MM | 2021 | 10.1145/3474085.3475375 | CAO (đã đọc sâu) | Tách m_f/m_b/m_u; background context pool; residual refinement |
| 9 | Shallow Attention Network for Polyp Segmentation | MICCAI | 2021 | 10.1007/978-3-030-87193-2_66 | TB | Polyp nhỏ; baseline so sánh |
| 10 | Polyp-PVT: Polyp Segmentation with Pyramid Vision Transformers | arXiv (CAAI AIR 2023) | 2021 | arXiv:2108.06932 | TB | CAC module attention vào prediction mềm — tinh thần "feedback mềm" |
| 11 | CaraNet: context axial reverse attention network for segmentation of small medical objects | J. Medical Imaging | 2023 | 10.1117/1.jmi.10.1.014005 | TB | Axial reverse attention cho object nhỏ |
| 12 | MEGANet: Multi-Scale Edge-Guided Attention Network for Weak Boundary Polyp Segmentation | WACV | 2024 | 10.1109/wacv57701.2024.00780 (arXiv:2309.03329) | TB | Edge-guidance cho polyp biên mờ — liên quan boundary (X2) |
| 13 | Colorectal Polyp Segmentation Based on Deep Learning Methods: A Systematic Review | J. Imaging | 2025 | 10.3390/jimaging11090293 | TB | Landscape 2025 + chuẩn metric/report |
| 14 | An Improved Dice Loss for Pneumothorax Segmentation by Mining the Information of Negative Areas | IEEE Access | 2020 | 10.1109/access.2020.3020475 | CAO (X2) | Phạt FP bằng cách "đào" vùng âm tính trong loss — gần như drop-in cho DiceBCE |
| 15 | The Lovász-Softmax Loss | CVPR | 2018 | 10.1109/cvpr.2018.00464 | CAO (X2) | Surrogate IoU — IoU phạt FP mạnh hơn Dice khi vùng thừa lớn |
| 16 | Tversky Loss Function for Image Segmentation Using 3D FCN | MLMI (LNCS) | 2017 | 10.1007/978-3-319-67389-9_44 | TB–CAO (X2) | β<0.5 → weight FP > FN; một knob duy nhất cân bằng FP/FN |
| 17 | Boundary loss for highly unbalanced segmentation | Medical Image Analysis | 2020 | 10.1016/j.media.2020.101851 | TB (X2) | Distance-map loss cho vùng không cân bằng |
| 18 | How Distance Transform Maps Boost Segmentation CNNs: An Empirical Study | MIDL | 2020 | arXiv:1912.13403 | TB (X2) | Bằng chứng distance map giúp CNN học biên |
| 19 | Active Boundary Loss for Semantic Segmentation | AAAI | 2022 | 10.1609/aaai.v36i2.20139 (arXiv:2102.02696) | TB (X2) | Boundary loss mới, có so sánh với các loss cũ |
| 20 | Metrics reloaded: recommendations for image analysis validation | Nature Methods | 2024 | 10.1038/s41592-023-02151-z | CAO (methodology) | Chuẩn metric/CI/report — áp cho toàn bộ eval pipeline |
| 21 | Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation | arXiv | 2013 | arXiv:1308.3432 | TB (X4) | Nguồn gốc STE; điều kiện gradient qua ngưỡng hoạt động |
| 22 | Categorical Reparameterization with Gumbel-Softmax | ICLR | 2017 | arXiv:1611.01144 | TB (X4) | Alternative cho gate rời rạc nếu STE chết |
| 23 | STAR-Caps: Capsule Networks with Straight-Through Attentive Routing | NeurIPS | 2019 | arXiv:1909.11974 | TB | STE cho attention routing — tiền lệ gần nhất (khác domain) |
| 24 | Tent: Fully Test-time Adaptation by Entropy Minimization | ICLR | 2021 | arXiv:2006.10726 | TB | Quan điểm khác cho refinement lúc test (entropy thay vì mask loop) |
| 25 | Polyp segmentation with consistency training and continuous update of pseudo-label | Scientific Reports | 2022 | 10.1038/s41598-022-17843-3 | TB | Semi-supervised polyp; pseudo-label = "feedback mềm" ngoài vòng lặp |

## Đánh giá novelty của "background-aware feedback loop" (sau Phase 4)

1. **FANetv2 (ICASSP 2025)** — chính lab của FANet đã mở rộng: iterative mask
   feedback từ epoch trước + text-guided + refinement lúc test. Feedback vẫn
   foreground-only → kênh m_bg tường minh vẫn là điểm khác biệt, nhưng không
   gian "iterative mask feedback" đã có người chiếm.
2. **Predictive coding polyp (2026)** — dùng top-down prediction-error feedback
   lúc inference; prediction error ngầm mang thông tin background và paper này
   đo thẳng FPR/FDR + FP trên nếp gấp. Không có kênh m_bg tường minh, không có
   cross-epoch mask loop → khác biệt còn lại của FANet-dual-path là **"kênh
   background tin cậy tường minh trong vòng phản hồi mask xuyên epoch"** — phạm
   vi novelty CÒN NHƯNG MỎNG, và bằng chứng thực nghiệm (Phase 3: FP↑ p=1e-7;
   Phase 4: train dual xấu/collapse) không ủng hộ giá trị thực tiễn của cơ chế.
3. PraNet reverse attention / UACANet m_b: single-pass, không phải feedback loop.
4. → **Khuyến nghị claim**: đừng dựa paper vào novelty của dual-path; novelty
   khả dĩ hơn nằm ở (a) phân tích negative-result có hệ thống (FP/FN decomposition,
   BN mismatch confounder, bug-tracking), hoặc (b) hướng loss-side nếu X2 thành công.

## Provenance

- DB: OpenAlex (`api.openalex.org/works`), arXiv (`export.arxiv.org/api/query`),
  Europe PMC (`ebi.ac.uk/europepmc/webservices/rest`). Access date 04/09/2026.
- Thất bại: Semantic Scholar (HTTP 429, không key), PubMed e-utilities (NCBI
  chặn IP datacenter) — ghi nhận giới hạn coverage.
- Scripts: `analysis/lit_search*.py` (phase 2), script tạm trong
  `%TEMP%\opencode\lit_*.py` (phiên 04/09; không commit).
- Verify metadata: 14 DOI kiểm tra ngược qua OpenAlex (title/venue/year khớp).
