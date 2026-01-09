# Co-reference Resolution in AMR

## Giới thiệu

Co-reference (tham chiếu) là khi một entity được nhắc đến nhiều lần trong câu, thường qua:
- Đại từ: anh ấy, cô ấy, nó, họ
- Lặp lại concept: bác sĩ → bác sĩ
- Tham chiếu ngầm

Trong AMR, khi có co-reference, chúng ta phải **TÁI SỬ DỤNG BIẾN** thay vì định nghĩa lại.

## Quy tắc

### ✅ ĐÚNG: Tái sử dụng biến

```
Định nghĩa lần đầu: (t / tôi)
Lần xuất hiện sau: t  (CHỈ viết tên biến)
```

### ❌ SAI: Định nghĩa lại biến

```
:ARG0 (t / tôi) ... :ARG1 (t / tôi)  ← SAI! Duplicate definition
```

## Ví dụ chi tiết

### Ví dụ 1: Đại từ đơn giản

**Câu:** Anh ấy là một bác sĩ giỏi.

**Phân tích:**
- "Anh ấy" và "bác sĩ" cùng chỉ một người
- Nên dùng cùng một biến

**AMR đúng:**
```
(b / bác_sĩ
    :domain(a / anh_ấy)
    :mod(g / giỏi))
```

Hoặc với co-reference rõ ràng hơn:
```
(b / bác_sĩ
    :ARG0-of(i / identity-01
        :ARG1(a / anh_ấy))
    :mod(g / giỏi))
```

### Ví dụ 2: Đại từ tham chiếu

**Câu:** Tôi gặp một bác sĩ. Anh ấy rất tốt bụng.

**Phân tích:**
- Câu 1: "một bác sĩ" - định nghĩa lần đầu
- Câu 2: "anh ấy" - tham chiếu lại bác sĩ

**AMR đúng:**
```
(m / multi-sentence
    :snt1(g / gặp
        :ARG0(t / tôi)
        :ARG1(b / bác_sĩ))
    :snt2(t1 / tốt_bụng
        :domain b           ← TÁI SỬ DỤNG biến b
        :degree(r / rất)))
```

**❌ SAI:**
```
:snt2(t1 / tốt_bụng
    :domain(b / bác_sĩ)    ← SAI! Đã định nghĩa b ở trên
    :degree(r / rất))
```

### Ví dụ 3: Nhiều tham chiếu

**Câu:** Tôi là sinh viên. Tôi học ở Hà Nội. Tôi thích toán học.

**Phân tích:**
- "Tôi" xuất hiện 3 lần
- Chỉ định nghĩa 1 lần, các lần sau tái sử dụng

**AMR đúng:**
```
(a / and
    :op1(s / sinh_viên
        :domain(t / tôi))      ← Định nghĩa lần đầu
    :op2(h / học
        :ARG0 t                ← Tái sử dụng
        :location(h1 / Hà_Nội))
    :op3(t1 / thích
        :ARG0 t                ← Tái sử dụng
        :ARG1(t2 / toán_học)))
```

### Ví dụ 4: Re-entrancy phức tạp

**Câu:** Người đàn ông yêu người phụ nữ cũng yêu anh ta.

**Phân tích:**
- "Người đàn ông" vừa là subject của "yêu", vừa là object của "yêu"
- Cần tái sử dụng biến để thể hiện quan hệ qua lại

**AMR đúng:**
```
(y / yêu
    :ARG0(đ / đàn_ông
        :mod(p / người))
    :ARG1(p1 / phụ_nữ
        :mod(p2 / người)
        :ARG0-of(y1 / yêu
            :ARG1 đ)))         ← Tái sử dụng biến đ
```

### Ví dụ 5: Variable naming khi trùng

**Câu:** Bác sĩ A gặp bác sĩ B.

**Phân tích:**
- 2 bác sĩ khác nhau
- Cần 2 biến khác nhau: b và b2

**AMR đúng:**
```
(g / gặp
    :ARG0(b / bác_sĩ
        :name(n / name
            :op1 "A"))
    :ARG1(b2 / bác_sĩ         ← Dùng b2 vì b đã dùng
        :name(n1 / name
            :op1 "B")))
```

## Các pattern thường gặp

### Pattern 1: Subject-Object Identity
```
X làm Y, Y cũng làm X
→ Cả hai đều tham chiếu lại nhau
```

### Pattern 2: Pronoun Resolution
```
Introduce entity → Use pronoun
→ Pronoun tái sử dụng biến của entity
```

### Pattern 3: Implicit Subject
```
(Tôi) làm X. Thích Y.
→ Subject ẩn "tôi" ở câu 2 tái sử dụng biến từ câu 1
```

## Kiểm tra Co-reference

### Checklist
- [ ] Mỗi concept chỉ được định nghĩa biến (x / concept) MỘT lần
- [ ] Khi concept xuất hiện lại, chỉ viết tên biến (x), không viết (x / concept)
- [ ] Đại từ phải tham chiếu đúng entity
- [ ] Biến phải unique trong toàn bộ graph (trừ khi tái sử dụng)

### Debugging
Nếu gặp lỗi "Duplicate node definition":
1. Tìm concept nào xuất hiện nhiều lần
2. Chỉ giữ định nghĩa đầu tiên: (x / concept)
3. Các lần sau chỉ viết: x

## So sánh với data thật

### Từ ground truth
```
#::snt mỗi phút ta phải thắp đèn và phải tắt đèn một lần !

(o / obligate-01
    :agent(t / ta)
    :frequency(r / rate-entity-91
        :ARG2(t1 / temporal-quantity
            :unit(p / phút)
            :quant 1))
    :theme(a / and
        :op1(t2 / thắp
            :patient(đ / đèn)
            :agent t)          ← Tái sử dụng biến t (ta)
        :op2(t3 / tắt
            :agent t           ← Tái sử dụng biến t
            :patient đ)))      ← Tái sử dụng biến đ (đèn)
```

**Quan sát:**
- "ta" xuất hiện 3 lần → chỉ định nghĩa 1 lần (t / ta), sau đó dùng t
- "đèn" xuất hiện 2 lần → chỉ định nghĩa 1 lần (đ / đèn), sau đó dùng đ

## Tổng kết

1. **Một concept, một định nghĩa**: `(x / concept)`
2. **Tái sử dụng**: Chỉ viết `x`, không viết `(x / concept)` lại
3. **Đại từ = tham chiếu**: Phải map đúng entity
4. **Check duplicate**: Tìm và fix nếu có concept định nghĩa nhiều lần

Đây là yếu tố QUAN TRỌNG để đạt F1 cao!
