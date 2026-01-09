# MTUP - Multi-Task Unified Prompting

## Khái niệm chính

### Sai lầm trong v1
Trong version cũ, chúng ta train **2 model riêng biệt**:
- Model 1: Sentence → No-var AMR (Stage 1)
- Model 2: No-var AMR → Full AMR (Stage 2)

Đây là **TWO-STAGE PIPELINE**, KHÔNG phải MTUP!

### Đúng theo MTUP
MTUP = **Multi-Task Unified Prompting** nghĩa là:
- **1 MODEL duy nhất**
- **1 PROMPT duy nhất**
- Model học **CẢ 2 TASK cùng lúc** trong cùng 1 lần training
- Tận dụng shared knowledge giữa 2 task

## Cấu trúc Data cho MTUP

Format mỗi sample trong training data:

```
#::snt <câu tiếng Việt>
#::task1 <AMR no variables>
#::task2 <AMR with variables>
```

Example:
```
#::snt Anh ấy là một bác sĩ giỏi.
#::task1 (bác_sĩ :domain(anh_ấy) :mod(giỏi))
#::task2 (b / bác_sĩ :domain(a / anh_ấy) :mod(g / giỏi))
```

## Unified Prompt Template

```
<|im_start|>system
Bạn là chuyên gia phân tích AMR (Abstract Meaning Representation).
Nhiệm vụ: Cho câu tiếng Việt, sinh 2 output:
1. Task 1: AMR Skeleton (không có biến)
2. Task 2: Full AMR (có biến, theo chuẩn PENMAN)

Quy tắc:
- Task 1: Chỉ có concept và relation, không có biến định danh
- Task 2: Thêm biến (x / concept), xử lý co-reference
- Đảm bảo cân bằng ngoặc
<|im_end|>
<|im_start|>user
Câu: {sentence}
<|im_end|>
<|im_start|>assistant
Task 1: {no_var_amr}
Task 2: {full_amr}
<|im_end|>
```

## Training Process

### Single Model Training
```python
# Load model một lần duy nhất
model = AutoModelForCausalLM.from_pretrained(...)

# Data chứa cả 2 task trong cùng 1 prompt
dataset = load_mtup_data()  # Format như trên

# Train 1 lần
trainer.train()
```

### Inference
Khi predict, model sẽ output cả 2 task:
```
Input: "Anh ấy là bác sĩ"

Output:
Task 1: (bác_sĩ :domain(anh_ấy))
Task 2: (b / bác_sĩ :domain(a / anh_ấy))
```

Chúng ta chỉ lấy **Task 2** làm kết quả cuối cùng.

## Lợi ích của MTUP

1. **Transfer Learning**: Task 1 giúp model hiểu structure, Task 2 học variable assignment
2. **Shared Representation**: Model học feature chung cho cả 2 task
3. **Auxiliary Task**: Task 1 làm auxiliary task giúp Task 2 tốt hơn
4. **Efficient**: Chỉ 1 model thay vì 2 model riêng

## Điểm khác biệt vs Baseline

| Aspect | Baseline | MTUP v2 |
|--------|----------|---------|
| Model | 1 model, 1 task | 1 model, 2 tasks |
| Prompt | Direct Sent→AMR | Unified 2-task prompt |
| Training | End-to-end | Multi-task learning |
| Inference | 1 output | 2 outputs (chọn task 2) |

## Implementation Key Points

1. **Data Preprocessing**: Merge train_stage1 + train_stage2 vào 1 file với format unified
2. **Prompt Design**: Thiết kế prompt rõ ràng cho cả 2 task
3. **Loss Calculation**: Loss tính trên cả 2 task output
4. **Inference**: Parse output để extract Task 2
5. **Evaluation**: So sánh Task 2 với ground truth

## Next Steps
1. Tạo preprocessing script để merge data
2. Viết training script với unified prompt
3. Viết prediction script để extract Task 2
4. Evaluate với SMATCH score
