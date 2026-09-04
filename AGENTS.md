# AGENTS.md — Working conventions for this repository

## Git workflow (BẮT BUỘC)

- Sau MỖI lần tạo file mới hoặc sửa file code (scripts/, src/, analysis/,
  notebooks/, configs/, kaggle/), **commit và push lên GitHub ngay**.
- Dùng message tiếng Anh, theo phong cách hiện có của repo (vd:
  `feat: ...`, `fix: ...`, `docs: ...`, `analysis: ...`).
- Chỉ stage các file liên quan đến công việc đang làm; không commit file
  không liên quan. Commit rồi `git push` ngay trong cùng lệnh.
- Thứ tự: `git status` -> `git diff` (xem lại) -> `git add <files>` ->
  `git commit -m "..."` -> `git push`.

## Research workflow

- Mỗi phase nghiên cứu: viết report theo `docs/reports/_TEMPLATE.md`,
  cập nhật index `docs/reports/README.md` (timeline + experiment tracking).
- Dùng scientific agent skills khi được yêu cầu (literature-review,
  paper-lookup, citation-management, hypothesis-generation,
  experimental-design, statistical-analysis, scientific-visualization,
  markdown-mermaid-writing, ...).
- Phân biệt bằng chứng: TRAIN (đáng tin) vs FROZEN EVAL (chỉ là khả năng
  khai thác feedback của model cũ) vs INVALID (bug) khi viết kết luận.