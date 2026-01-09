# MTUP v2 - Multi-Task Unified Prompting for Vietnamese AMR Parsing

## Mục tiêu
Cải thiện F1 score từ baseline (0.47) bằng phương pháp MTUP với 2 sub-tasks:
1. **Stage 1**: Vietnamese Sentence → AMR without variables (skeleton)
2. **Stage 2**: Skeleton + Sentence → Full AMR with variables (chuẩn PENMAN)

## Các cải tiến so với v1

### 1. Variable Reference Handling
- Xử lý tham chiếu biến (co-reference): Khi có đại từ hoặc nhắc lại, tái sử dụng biến đã định nghĩa
- Ví dụ: "Anh ấy là bác sĩ giỏi" → biến cho "anh ấy" và "bác sĩ" phải giống nhau

### 2. Strict PENMAN Format Compliance
- Đảm bảo output khớp với format trong public_test_ground_truth.txt
- Validation sau mỗi prediction để đảm bảo cấu trúc hợp lệ
- Kiểm tra cân bằng ngoặc, format biến, concept naming

### 3. Improved Prompting
- Prompt rõ ràng hơn về quy tắc tái sử dụng biến
- Ép buộc model học pattern từ training data
- Thêm ví dụ về co-reference trong prompt

### 4. Better Evaluation
- Tính SMATCH score trực tiếp với ground truth
- So sánh chi tiết: precision, recall, F1
- Phân tích lỗi thường gặp

## Cấu trúc thư mục

```
mtup_v2/
├── scripts/
│   ├── train_stage1.py        # Training cho Stage 1
│   ├── train_stage2.py        # Training cho Stage 2
│   ├── predict_mtup.py        # Prediction pipeline
│   └── evaluate.py            # Tính SMATCH score
├── preprocessing/
│   ├── create_stage1_data.py  # Tạo dữ liệu Stage 1
│   ├── create_stage2_data.py  # Tạo dữ liệu Stage 2
│   └── validate_amr.py        # Validate AMR format
├── evaluation/
│   └── compute_smatch.py      # SMATCH evaluation
└── docs/
    ├── README.md              # File này
    ├── TRAINING_GUIDE.md      # Hướng dẫn training
    └── EXAMPLES.md            # Ví dụ về co-reference

```

## Workflow

### Bước 1: Preprocessing
```bash
python mtup_v2/preprocessing/create_stage1_data.py
python mtup_v2/preprocessing/create_stage2_data.py
```

### Bước 2: Training (trên server)
```bash
# Stage 1
python mtup_v2/scripts/train_stage1.py \
    --data_path data/train_stage1.txt \
    --output_dir outputs/mtup_v2_stage1 \
    --epochs 5

# Stage 2
python mtup_v2/scripts/train_stage2.py \
    --data_path data/train_stage2.txt \
    --output_dir outputs/mtup_v2_stage2 \
    --epochs 5
```

### Bước 3: Prediction
```bash
python mtup_v2/scripts/predict_mtup.py \
    --stage1_adapter outputs/mtup_v2_stage1/final_adapter \
    --stage2_adapter outputs/mtup_v2_stage2/final_adapter \
    --input_file data/public_test.txt \
    --output_file outputs/predictions.txt
```

### Bước 4: Evaluation
```bash
python mtup_v2/scripts/evaluate.py \
    --predictions outputs/predictions.txt \
    --ground_truth data/public_test_ground_truth.txt
```

## Key Features

### Co-reference Resolution
Model được train để:
1. Nhận diện khi một entity được nhắc lại
2. Tái sử dụng biến đã định nghĩa thay vì tạo mới
3. Xử lý các trường hợp: đại từ, lặp lại concept, implicit reference

### Variable Naming Convention
- Sử dụng chữ cái đầu của concept: `(t / tôi)`, `(b / bác_sĩ)`
- Nếu trùng tên: thêm số `(t2 / tôi)`, `(b2 / bác_sĩ)`
- Tái sử dụng: chỉ viết tên biến, không viết lại concept

### Example
```
Input: Anh ấy là một bác sĩ giỏi. Anh rất tận tâm.

Stage 1 Output:
(bác_sĩ :domain(anh_ấy) :mod(giỏi))
(tận_tâm :pivot(anh) :degree(rất))

Stage 2 Output (WITH co-reference):
(b / bác_sĩ :domain(a / anh_ấy) :mod(g / giỏi))
(t / tận_tâm :pivot a :degree(r / rất))  ← Reuse variable 'a'
```

## Notes
- Baseline F1: 0.47
- Target: F1 > 0.47
- Model: Qwen2.5-7B-Instruct with QLoRA
- Format: Strict PENMAN compliance
