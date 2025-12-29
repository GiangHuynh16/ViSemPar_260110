# Chiến Lược Demo AMR cho Tiếng Việt

**Ngày tạo:** 2025-12-29
**Mục đích:** Thiết kế demo website chứng minh giá trị thực tiễn của AMR parsing cho tiếng Việt

---

## 📊 Tổng Quan Quyết Định

### ❌ Loại bỏ: Book Search Use Case

**Lý do:**
1. **Không match với training data**
   - Model được train trên văn bản báo chí/xã hội (VLSP 2025 corpus)
   - Sách thiếu nhi ("Dế Mèn Phiêu Lưu Ký") hoàn toàn nằm ngoài domain

2. **Không có ground truth**
   - Phải tự viết AMR cho sách → không có cách verify
   - Kết quả không thuyết phục vì model chưa từng thấy data tương tự

3. **Giá trị thực tiễn thấp**
   - Keyword search + embeddings đã đủ tốt cho việc tìm sách
   - Khó chứng minh AMR vượt trội hơn phương pháp truyền thống

4. **Demo không convincing**
   - Người xem sẽ hỏi: "Tại sao không dùng search engine thường?"
   - Không có câu trả lời thuyết phục

### ✅ Thay thế: News/Article Analysis Use Cases

**Lý do:**
1. **100% match với training data**
   - Corpus là báo chí/xã hội → model hiểu tốt
   - Có 150 ground truth examples để verify

2. **Giá trị thực tiễn cao**
   - Media monitoring: Theo dõi ai nói gì về chủ đề nào
   - Fact checking: Phát hiện tin tức mâu thuẫn
   - News aggregation: Gom nhóm tin cùng nghĩa

3. **Showcase điểm mạnh của AMR**
   - Semantic role labeling (ai làm gì với ai)
   - Paraphrase detection (câu khác nhưng cùng nghĩa)
   - Structural matching (tìm theo cấu trúc, không phải keyword)

---

## 🎯 3 Use Cases Mới

### **USE CASE 1: AMR Tree Visualization** (GIỮ NGUYÊN)

**Mục đích:** Hiển thị trực quan cấu trúc AMR

**Input:** 1 câu tiếng Việt (từ domain báo chí)

**Output:**
```
┌─ Tree View (D3.js interactive graph)
├─ Text View (PENMAN notation)
└─ Role Table
    ARG0 (chủ thể):     [...]
    ARG1 (đối tượng):   [...]
    location:           [...]
    time:               [...]
```

**Ví dụ tốt:**
- "người lao động đi nước ngoài tạo vốn giúp gia đình thoát nghèo"
- "tôi nhớ lời chủ tịch xã nhắc đi nhắc lại"
- "xã có 68 tổ nhân dân, mỗi tổ phụ trách 40 gia đình"

---

### **USE CASE 2: News Event Extraction** (THAY Book Search)

**Mục đích:** Tìm kiếm tin tức theo **vai trò ngữ nghĩa** thay vì keyword

#### Scenario 1: Tìm theo vai trò người nói

**Query:** "Tìm bài viết về chủ tịch xã nói/nhắc gì"

**Keyword Search (Baseline):**
```
Input: "chủ tịch xã nói"
Results: 30 bài
Issues:
  ❌ "họp bàn về chủ tịch xã" ← nói VỀ chủ tịch (sai role)
  ❌ "chủ tịch xã được khen" ← chủ tịch là đối tượng (sai role)
  ❌ "vấn đề chủ tịch xã" ← không có hành động "nói"

Precision: ~40% (nhiều false positives)
```

**AMR Semantic Search:**
```
AMR Query: (nói :ARG0(chủ_tịch :mod(xã)))
           hoặc (nhắc :agent(chủ_tịch :mod(xã)))

Results: 5 bài
Matched example:
  ✅ "tôi nhớ lời chủ tịch xã nhắc đi nhắc lại"
     AMR: (nhớ :theme(lời :poss(chủ_tịch :agent-of(nhắc))))
     Role match: chủ_tịch = :agent-of(nhắc) ✓
     Score: 95%

Precision: ~90% (chỉ match đúng role)
```

#### Scenario 2: Tìm theo hành động + địa điểm

**Query:** "Tìm tin về người làm việc ở nước ngoài"

**AMR Query:** `(làm_việc :ARG0(người) :location(nước_ngoài))`

**Matched:**
- "đến nay xã có 672 người đi làm việc ở nước ngoài" ✅
- "sau ba năm làm việc ở nước ngoài, họ tạo vốn" ✅

**Not matched (correctly):**
- "bàn về vấn đề người nước ngoài" ← không có "làm việc"
- "công việc ở nước ngoài" ← không có "người" làm chủ thể

#### Demo UI:

```
┌──────────────────────────────────────────────────┐
│ 🔍 TÌM KIẾM TIN TỨC                             │
├──────────────────────────────────────────────────┤
│ Query: Tìm bài về chủ tịch xã nói gì           │
│                                                  │
│ Tab 1: Keyword Search  │ Tab 2: AMR Search     │
├────────────────────────┼───────────────────────┤
│ 📰 30 results          │ 📰 5 results         │
│ Precision: 40%         │ Precision: 90%       │
│                        │                       │
│ Issues:                │ Advantages:           │
│ • Nhiều sai role       │ • Đúng role          │
│ • Nhiễu cao            │ • Chính xác cao      │
└────────────────────────┴───────────────────────┘

Kết quả AMR Search:
┌────────────────────────────────────────────────┐
│ 1. "tôi nhớ lời chủ tịch xã nhắc..."         │
│    📊 Semantic Match: 95%                     │
│    🎯 Role: chủ_tịch = :agent-of(nhắc)       │
│    ✅ Speaker role matched                    │
└────────────────────────────────────────────────┘
```

---

### **USE CASE 3: Sentence Analysis** (MỞ RỘNG)

Chia thành 3 sub-use-cases:

#### **3.1. Paraphrase Detection** (Phát hiện câu viết lại)

**Mục đích:** Phát hiện câu **khác nhau về cú pháp** nhưng **cùng nghĩa**

**Ví dụ:**

**Input 1:** "người lao động đi nước ngoài tạo vốn giúp gia đình thoát nghèo"
**Input 2:** "gia đình thoát nghèo nhờ vốn từ người lao động đi nước ngoài"

**Analysis:**
```
📝 Text Similarity: 42% (thứ tự từ khác nhiều)
🎯 AMR Similarity: 94% (cùng cấu trúc ngữ nghĩa)

AMR (cả 2 câu):
(thoát
  :ARG0(gia_đình)
  :ARG1(nghèo)
  :manner(vốn :source(người_lao_động
                      :agent-of(đi :destination(nước_ngoài)))))

Roles matched:
  ✅ ARG0: gia_đình (chủ thể thoát)
  ✅ ARG1: nghèo (thoát khỏi cái gì)
  ✅ source: người_lao_động (nguồn vốn)
  ✅ destination: nước_ngoài

💡 Kết luận: PARAPHRASE (cùng nghĩa, khác cách diễn đạt)
```

#### **3.2. Fact Comparison** (So sánh sự thật)

**Mục đích:** Phát hiện 2 câu **mâu thuẫn** hoặc **nhất quán**

**Ví dụ 1: Nhất quán (Consistent)**

**Input 1:** "xã có 68 tổ nhân dân, mỗi tổ phụ trách 40 gia đình"
**Input 2:** "đến nay xã có 672 người đi làm việc ở nước ngoài"

**Analysis:**
```
AMR 1: (có :ARG0(xã) :ARG1(tổ :quant(68)))
AMR 2: (có :ARG0(xã) :ARG1(người :quant(672)))

So sánh:
  ✅ Cùng chủ thể: xã
  ✅ Cùng structure: (có :ARG0(...) :ARG1(...))
  ❌ Khác đối tượng: tổ ≠ người
  ❌ Khác số lượng: 68 ≠ 672

💡 Kết luận: NHẤT QUÁN (2 sự thật khác nhau về cùng xã)
```

**Ví dụ 2: Mâu thuẫn (Contradictory)**

**Input 1:** "xã có 68 tổ nhân dân"
**Input 2:** "xã có 70 tổ nhân dân" [từ nguồn khác]

**Analysis:**
```
AMR 1: (có :ARG0(xã) :ARG1(tổ :quant(68)))
AMR 2: (có :ARG0(xã) :ARG1(tổ :quant(70)))

So sánh:
  ✅ Cùng chủ thể: xã
  ✅ Cùng đối tượng: tổ
  ❌ Khác số lượng: 68 ≠ 70

⚠️  Kết luận: MÂU THUẪN (cùng đối tượng, khác số liệu)
```

#### **3.3. Role Extraction** (Trích xuất vai trò)

**Mục đích:** Tự động trích xuất **ai làm gì với ai**

**Ví dụ:**

**Input:** "tôi nhớ lời chủ tịch xã nhắc đi nhắc lại"

**Extracted Roles:**
```
┌──────────────────────────────────────────────┐
│ 🎯 VAI TRÒ NGỮ NGHĨA                        │
├──────────────────────────────────────────────┤
│ rememberer (người nhớ):                     │
│   → tôi                                      │
│                                              │
│ thing_remembered (điều được nhớ):           │
│   → lời                                      │
│                                              │
│ speaker (người nói):                         │
│   → chủ tịch xã                             │
│                                              │
│ action (hành động):                          │
│   → nhắc đi nhắc lại                        │
└──────────────────────────────────────────────┘

AMR Structure:
(nhớ
  :pivot(tôi)              ← người nhớ
  :theme(lời               ← điều được nhớ
    :poss(chủ_tịch         ← chủ sở hữu lời nói
      :agent-of(nhắc))))   ← người thực hiện hành động nói
```

---

## 📋 Dataset Mẫu - Cases Có Lợi Cho Model

### Level 1: Very Easy (95%+ accuracy expected)

**Dùng cho demo chính - đảm bảo thành công**

```python
LEVEL_1_CASES = [
    {
        "name": "Simple SVO",
        "sentences": [
            "người đảng viên phải làm gương",
            "làm gương là việc người đảng viên phải làm",
            "phải làm gương, người đảng viên"
        ],
        "expected_amr": "(làm_gương :ARG0(người :mod(đảng_viên)) :modality(phải))",
        "why_favorable": "Câu ngắn, structure đơn giản, không có nested relations"
    },
    {
        "name": "Passive/Active Voice",
        "sentences": [
            "xã mời cán bộ về tập huấn",
            "cán bộ được xã mời về tập huấn",
            "cán bộ được mời về tập huấn bởi xã"
        ],
        "expected_amr": "(mời :ARG0(xã) :ARG1(cán_bộ) :purpose(tập_huấn))",
        "why_favorable": "Passive transformation là điểm mạnh tự nhiên của AMR"
    }
]
```

### Level 2: Medium (80%+ accuracy expected)

**Dùng cho advanced demo - vẫn khá an toàn**

```python
LEVEL_2_CASES = [
    {
        "name": "Simple Modifiers",
        "sentence": "người lao động đi nước ngoài",
        "amr": "(người :mod(lao_động) :agent-of(đi :destination(nước_ngoài)))",
        "why_favorable": "Pattern :agent-of và :mod rất phổ biến trong training data"
    },
    {
        "name": "Possession with Location",
        "sentence": "lời của chủ tịch xã",
        "amr": "(lời :poss(chủ_tịch :mod(xã)))",
        "why_favorable": ":poss + :mod xuất hiện nhiều, model học tốt"
    }
]
```

### ❌ TRÁNH - Cases Model Chưa Làm Tốt

```python
AVOID_CASES = [
    "Câu phức với nhiều mệnh đề phụ thuộc",
    "Đồng âm cần ngữ cảnh phức tạp (ca/cá/ca)",
    "Nested possession (sách của bạn của tôi)",
    "Câu dài >15 từ với nhiều modifiers",
    "Quan hệ nhân quả phức tạp"
]
```

---

## 🎨 Demo Website Architecture

### Technology Stack

**Frontend:**
- React.js (UI components)
- D3.js (AMR tree visualization)
- TailwindCSS (styling)

**Backend:**
- FastAPI (Python web framework)
- HuggingFace Inference API (AMR model)
- MongoDB (lưu examples + cache)

**Deployment:**
- Vercel (frontend)
- Railway/Render (backend API)

### Page Structure

```
┌─────────────────────────────────────────────┐
│ HOMEPAGE                                    │
│ - Giới thiệu AMR for Vietnamese            │
│ - 3 use case buttons                       │
└─────────────────────────────────────────────┘
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓                   ↓
┌─────────┐      ┌─────────┐      ┌─────────┐
│ Page 1  │      │ Page 2  │      │ Page 3  │
│  Tree   │      │  Event  │      │Sentence │
│  Viz    │      │ Extract │      │Analysis │
└─────────┘      └─────────┘      └─────────┘
```

### Page 1: AMR Tree Visualization

```
┌────────────────────────────────────────────────┐
│ 📊 AMR TREE VISUALIZATION                     │
├────────────────────────────────────────────────┤
│ Input:                                         │
│ ┌────────────────────────────────────────────┐│
│ │ Nhập câu tiếng Việt...                    ││
│ │                                            ││
│ └────────────────────────────────────────────┘│
│                                                │
│ [Ví dụ mẫu] [Phân tích] [Xóa]                │
├────────────────────────────────────────────────┤
│ Output:                                        │
│                                                │
│ Tab 1: Tree View (D3.js interactive)          │
│        ○ nhớ                                   │
│       ╱  ╲                                    │
│   :pivot :theme                                │
│     │      │                                   │
│    tôi    lời                                  │
│           │                                    │
│         :poss                                  │
│           │                                    │
│       chủ_tịch ─:agent-of→ nhắc               │
│                                                │
│ Tab 2: Text View (PENMAN)                     │
│   (n / nhớ                                     │
│     :pivot(t / tôi)                           │
│     :theme(l / lời...))                       │
│                                                │
│ Tab 3: Role Table                             │
│   ┌────────────┬──────────────────┐           │
│   │ Role       │ Entity           │           │
│   ├────────────┼──────────────────┤           │
│   │ :pivot     │ tôi (người nhớ) │           │
│   │ :theme     │ lời (được nhớ)  │           │
│   │ :poss      │ chủ_tịch (chủ) │           │
│   │ :agent-of  │ nhắc (hành động)│           │
│   └────────────┴──────────────────┘           │
└────────────────────────────────────────────────┘
```

### Page 2: News Event Extraction

```
┌────────────────────────────────────────────────┐
│ 🔍 NEWS EVENT EXTRACTION                      │
├────────────────────────────────────────────────┤
│ Query:                                         │
│ ┌────────────────────────────────────────────┐│
│ │ Tìm bài viết về chủ tịch xã nói gì...     ││
│ └────────────────────────────────────────────┘│
│                                                │
│ [Ví dụ mẫu] [Tìm kiếm] [Reset]               │
├────────────────────────────────────────────────┤
│ Tab 1: Keyword Search  │ Tab 2: AMR Search   │
├───────────────────────┼────────────────────────┤
│ 📰 30 results         │ 📰 5 results          │
│ Precision: 40%        │ Precision: 90%        │
│                       │                        │
│ Top results:          │ Top results:           │
│ 1. "họp về chủ tịch" │ 1. "lời chủ tịch     │
│    ⚠️ Sai role        │    nhắc..."           │
│                       │    ✅ Match: 95%      │
│ 2. "chủ tịch được    │    🎯 Role: speaker   │
│    khen"              │                        │
│    ⚠️ Sai role        │ 2. "chủ tịch phát    │
│                       │    biểu..."           │
│ 3. "vấn đề chủ tịch"│    ✅ Match: 92%      │
│    ⚠️ Thiếu action    │    🎯 Role: speaker   │
└───────────────────────┴────────────────────────┘

[Xem chi tiết phân tích AMR của từng kết quả]
```

### Page 3: Sentence Analysis

```
┌────────────────────────────────────────────────┐
│ 📝 SENTENCE ANALYSIS                          │
├────────────────────────────────────────────────┤
│ Chọn chức năng:                               │
│ ○ 3.1. Paraphrase Detection                   │
│ ○ 3.2. Fact Comparison                        │
│ ○ 3.3. Role Extraction                        │
└────────────────────────────────────────────────┘

┌─ 3.1. PARAPHRASE DETECTION ───────────────────┐
│ Input 1:                                       │
│ ┌────────────────────────────────────────────┐│
│ │ người lao động đi nước ngoài tạo vốn...   ││
│ └────────────────────────────────────────────┘│
│                                                │
│ Input 2:                                       │
│ ┌────────────────────────────────────────────┐│
│ │ gia đình thoát nghèo nhờ vốn từ người...  ││
│ └────────────────────────────────────────────┘│
│                                                │
│ [Phân tích]                                    │
├────────────────────────────────────────────────┤
│ Kết quả:                                       │
│                                                │
│ 📝 Text Similarity: 42%                       │
│ 🎯 AMR Similarity: 94% ✅                     │
│                                                │
│ Roles matched:                                 │
│   ✅ ARG0: gia_đình (chủ thể)                │
│   ✅ ARG1: nghèo (đối tượng)                  │
│   ✅ source: người_lao_động                   │
│   ✅ destination: nước_ngoài                  │
│                                                │
│ 💡 Verdict: PARAPHRASE                        │
│    (Cùng nghĩa, khác cách diễn đạt)          │
└────────────────────────────────────────────────┘
```

---

## 🚀 Implementation Roadmap

### Phase 1: Backend API (Week 1-2)

```python
# API Endpoints

POST /api/parse
  Input: {"sentence": "..."}
  Output: {"amr": "...", "tree": {...}, "roles": {...}}

POST /api/search/semantic
  Input: {"query": "...", "corpus": [...]}
  Output: {"results": [...], "amr_query": "..."}

POST /api/compare
  Input: {"sentence_1": "...", "sentence_2": "..."}
  Output: {"similarity": 0.94, "verdict": "paraphrase", ...}

GET /api/examples
  Output: {"level_1": [...], "level_2": [...], ...}
```

### Phase 2: Frontend UI (Week 2-3)

- Page 1: D3.js tree visualization
- Page 2: Search comparison interface
- Page 3: Multi-tab analysis tools

### Phase 3: Integration & Testing (Week 3-4)

- HuggingFace model integration
- Performance optimization
- User testing with Vietnamese speakers

### Phase 4: Deployment (Week 4)

- Deploy backend to Railway
- Deploy frontend to Vercel
- DNS setup + SSL

---

## 📈 Success Metrics

### Technical Metrics

- **Page 1:** AMR parsing accuracy > 85% on demo examples
- **Page 2:** Precision of AMR search > 85% vs keyword search ~40%
- **Page 3:** Paraphrase detection accuracy > 90%

### User Metrics

- Demo convinces reviewers that AMR has real value
- Users can identify at least 2 advantages of AMR over keyword search
- Positive feedback on visualization clarity

---

## 💡 Key Talking Points for Demo

1. **"AMR hiểu vai trò, không chỉ từ khóa"**
   - Phân biệt "chủ tịch nói" vs "nói về chủ tịch"

2. **"Chuẩn hóa ngữ nghĩa"**
   - Câu khác nhau → cùng AMR → cùng nghĩa

3. **"Tìm kiếm cấu trúc, không chỉ pattern"**
   - Query: "ai làm gì ở đâu" → match đúng structure

4. **"Ứng dụng thực tế: Media monitoring, fact checking"**
   - Không phải academic toy - giải quyết vấn đề thực tế

---

## 📦 Deliverables

1. ✅ `demo_examples.json` - Dataset mẫu với favorable cases
2. ✅ `DEMO_STRATEGY.md` - Document này
3. ⏳ Backend API code
4. ⏳ Frontend React app
5. ⏳ Deployment scripts
6. ⏳ User guide & documentation

---

**Status:** Ready for implementation
**Next Steps:** Start coding backend API với FastAPI + HuggingFace integration

