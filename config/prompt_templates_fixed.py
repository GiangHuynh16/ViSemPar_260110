"""
MTUP Fixed - Minimal Prompt Templates with Penman Examples
Based on successful Baseline approach
"""

# ==============================================================================
# MTUP MINIMAL TEMPLATE - INSPIRED BY BASELINE SUCCESS
# ==============================================================================

MTUP_MINIMAL_TEMPLATE = """Chuyển câu tiếng Việt sau sang AMR (Abstract Meaning Representation) theo chuẩn PENMAN.

Ví dụ:
Câu: Tôi nhớ lời anh chủ tịch xã.

Bước 1 - AMR không biến:
(nhớ :pivot (tôi) :theme (lời :poss (chủ_tịch :mod (anh) :mod (xã))))

Bước 2 - AMR có biến (chuẩn PENMAN):
(n / nhớ
    :pivot (t / tôi)
    :theme (l / lời
        :poss (c / chủ_tịch
            :mod (a / anh)
            :mod (x / xã))))

---

Câu: {sentence}

Bước 1 - AMR không biến:
{amr_no_vars}

Bước 2 - AMR có biến (chuẩn PENMAN):
{amr_with_vars}"""


# ==============================================================================
# MTUP ULTRA MINIMAL - FOR BEST PERFORMANCE
# ==============================================================================

MTUP_ULTRA_MINIMAL = """Chuyển câu tiếng Việt sau sang AMR theo chuẩn PENMAN.

VÍ DỤ:
Câu: Anh ấy đã hoàn thành công việc.
AMR không biến: (hoàn_thành :agent (anh) :theme (công_việc) :aspect (đã))
AMR chuẩn PENMAN:
(h / hoàn_thành
    :agent (a / anh)
    :theme (c / công_việc)
    :aspect (đ / đã))

---

Câu: {sentence}

AMR không biến:
{amr_no_vars}

AMR chuẩn PENMAN:
{amr_with_vars}"""


# ==============================================================================
# INFERENCE-ONLY TEMPLATES - MATCHING TRAINING FORMAT
# CRITICAL: Must match MTUP_ULTRA_MINIMAL format for correct inference
# ==============================================================================

# Step 1: Generate AMR without variables
MTUP_INFERENCE_TEMPLATE = """Chuyển câu tiếng Việt sau sang AMR theo chuẩn PENMAN.

VÍ DỤ:
Câu: Anh ấy đã hoàn thành công việc.
AMR không biến: (hoàn_thành :agent (anh) :theme (công_việc) :aspect (đã))

---

Câu: {sentence}

AMR không biến:"""

# Step 2: Add variables to AMR
MTUP_INFERENCE_STEP2_TEMPLATE = """Chuyển câu tiếng Việt sau sang AMR theo chuẩn PENMAN.

VÍ DỤ:
Câu: Anh ấy đã hoàn thành công việc.
AMR không biến: (hoàn_thành :agent (anh) :theme (công_việc) :aspect (đã))
AMR chuẩn PENMAN:
(h / hoàn_thành
    :agent (a / anh)
    :theme (c / công_việc)
    :aspect (đ / đã))

---

Câu: {sentence}

AMR không biến:
{amr_no_vars}

AMR chuẩn PENMAN:"""


# ==============================================================================
# RECOMMENDED TEMPLATE
# ==============================================================================

RECOMMENDED_MTUP_TEMPLATE = MTUP_ULTRA_MINIMAL


def get_mtup_template(template_type: str = 'ultra_minimal') -> str:
    """
    Get MTUP template by type

    Args:
        template_type: 'minimal', 'ultra_minimal', 'inference', or 'recommended'

    Returns:
        Template string
    """
    templates = {
        'minimal': MTUP_MINIMAL_TEMPLATE,
        'ultra_minimal': MTUP_ULTRA_MINIMAL,
        'inference': MTUP_INFERENCE_TEMPLATE,
        'recommended': RECOMMENDED_MTUP_TEMPLATE,
    }
    return templates.get(template_type, RECOMMENDED_MTUP_TEMPLATE)


def format_mtup_training_example(
    sentence: str,
    amr_no_vars: str,
    amr_with_vars: str,
    template_type: str = 'recommended'
) -> str:
    """
    Format training example for MTUP

    Args:
        sentence: Vietnamese sentence
        amr_no_vars: AMR without variables (linearized)
        amr_with_vars: AMR with variables (Penman format)
        template_type: Template to use

    Returns:
        Formatted training example
    """
    template = get_mtup_template(template_type)
    return template.format(
        sentence=sentence,
        amr_no_vars=amr_no_vars,
        amr_with_vars=amr_with_vars
    )


def format_mtup_inference(sentence: str, step: int = 1) -> str:
    """
    Format input for inference

    Args:
        sentence: Vietnamese sentence
        step: 1 for initial, 2 for binding

    Returns:
        Formatted inference prompt
    """
    if step == 1:
        return MTUP_INFERENCE_TEMPLATE.format(sentence=sentence)
    else:
        return MTUP_INFERENCE_STEP2_TEMPLATE


# ==============================================================================
# PENMAN FORMAT REFERENCE
# ==============================================================================

PENMAN_FORMAT_GUIDE = """
CHUẨN PENMAN - AMR Format Standard:

1. Cấu trúc cơ bản:
   (biến / khái_niệm
       :quan_hệ (biến2 / khái_niệm2)
       :quan_hệ2 ...)

2. Quy tắc biến:
   - Mỗi khái niệm có 1 biến duy nhất
   - Biến thường là chữ cái đầu của khái niệm
   - Khái niệm lặp lại dùng chung biến (coreference)

3. Ví dụ chuẩn:
   (n / nhớ
       :pivot (t / tôi)
       :theme (l / lời
           :poss (c / chủ_tịch)))

4. SAI: (nhớ :pivot (tôi) ...) - thiếu biến
   ĐÚNG: (n / nhớ :pivot (t / tôi) ...)

5. Khái niệm nhiều từ: dùng gạch dưới
   chủ_tịch, hoàn_thành, công_việc
"""


if __name__ == "__main__":
    # Test templates
    sentence = "Tôi đã hoàn thành công việc quan trọng."
    amr_no_vars = "(hoàn_thành :agent (tôi) :theme (công_việc :mod (quan_trọng)) :aspect (đã))"
    amr_with_vars = """(h / hoàn_thành
    :agent (t / tôi)
    :theme (c / công_việc
        :mod (q / quan_trọng))
    :aspect (đ / đã))"""

    print("=" * 80)
    print("MTUP FIXED - MINIMAL TEMPLATE TEST")
    print("=" * 80)

    example = format_mtup_training_example(
        sentence, amr_no_vars, amr_with_vars, 'ultra_minimal'
    )
    print(example)

    print("\n" + "=" * 80)
    print("PENMAN FORMAT GUIDE")
    print("=" * 80)
    print(PENMAN_FORMAT_GUIDE)
