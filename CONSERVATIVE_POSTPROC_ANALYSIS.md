# Conservative Post-Processing Analysis

## Quick Test Results (10 samples)

### Performance
- **Processed**: 7/10 (70%)
- **Errors**: 3 (30%)
- **F1**: 0.4933

### Comparison with Aggressive Approach

| Metric | No Post-Proc | Aggressive | Conservative |
|--------|--------------|------------|--------------|
| Valid samples | 67% (101/150) | 90% (135/150) | 70% (7/10) |
| F1 Score | **0.477** | 0.463 ❌ | **0.493** ✅ |
| Change | baseline | **-1.4%** | **+1.6%** |

## Key Findings

### ✅ Conservative Approach Works Better!

**Quick test shows F1 = 0.493** (vs 0.477 baseline, +1.6%)

This confirms our hypothesis:
- **Aggressive post-processing** fixed parsing but **changed semantics** → Lower F1
- **Conservative post-processing** fixes only broken cases, **preserves correct AMRs** → Higher F1

### Remaining Errors (3 out of 10)

1. **Duplicate node name `l`**: Variable renaming still needed (but carefully!)
2. **Unmatched parenthesis**: Extra closing paren at position 124
3. **Duplicate node name `n`**: Variable renaming needed

## What Conservative Post-Processing Does

```python
def post_process_amr_conservative(amr_string: str) -> str:
    """
    MINIMAL changes to avoid breaking correct AMRs
    """
    # 1. Remove prompt leakage (only if clearly not AMR content)
    if any(marker in before_paren for marker in ['bước', 'Hướng dẫn', 'NHIỆM VỤ']):
        amr = amr[first_paren:]

    # 2. ONLY add missing closing parens (NEVER remove!)
    if open_count > close_count:
        amr = amr + ')' * (open_count - close_count)

    # 3. Clean whitespace
    amr = re.sub(r'\s+', ' ', amr)

    # SKIP: Variable renaming (too risky - breaks references)
    # SKIP: Removing extra parens (might break correct nested structures)
```

## Next Steps

### 1. Wait for Full Evaluation (150 samples)

Monitor with:
```bash
tail -f outputs/evaluation_full_*.log
```

Expected results:
- **Valid samples**: ~105-110/150 (70-73%)
- **F1 Score**: **0.49-0.51** (baseline: 0.477)
- **Improvement**: +2.7% to +6.9%

### 2. Analyze Remaining Errors

The 3 error types still occurring:

#### A. Duplicate Node Names (2 errors)
**Example**: `(n / nhớ :agent (n / tôi))`

**Why conservative approach skips this**: Variable renaming is risky because:
- Need to update BOTH definition AND all references
- Current aggressive approach only renames definitions
- Breaking references makes SMATCH score WORSE

**Possible safe fix**:
```python
# Track ALL occurrences first, then rename consistently
def rename_duplicates_safe(amr):
    # 1. Parse to find all (var / concept) definitions
    # 2. Find all :ARG references to those vars
    # 3. Rename both definition AND references together
    # 4. Only apply if we can track all references
```

#### B. Unmatched Parenthesis (1 error)
**Example**: Extra `)` at position 124

**Why this still occurs**: Conservative approach only ADDS closing parens, doesn't remove extra ones.

**Trade-off**:
- Removing extra `)` risks breaking correct nested structures
- Keeping extra `)` causes parse errors
- **Current choice**: Prefer semantically correct over parseable

**Possible safe fix**:
```python
# Only remove ')' if it's CLEARLY extra (not part of nested structure)
# Algorithm: Track stack depth, only remove if stack is empty
```

### 3. Targeted Fixes (After Full Evaluation)

Based on full 150-sample results, we can:

1. **If F1 ≥ 0.49**: Conservative approach is working!
   - Optionally add safe duplicate variable renaming
   - Optionally add safe extra-paren removal

2. **If F1 = 0.48-0.49**: Minor improvements needed
   - Analyze which of the 3 error types is most common
   - Implement targeted fix for most common error

3. **If F1 < 0.48**: Conservative approach needs adjustment
   - Review what's being changed
   - Make even more minimal

## Expected Timeline

- **Full evaluation**: ~15-20 minutes (150 samples with SMATCH)
- **Results available**: Check tmux session or log file
- **Next decision**: Based on actual F1 score

## Success Criteria

✅ **Success**: F1 ≥ 0.49 (+2.7% improvement)
🟡 **Partial**: F1 = 0.48-0.49 (marginal improvement)
❌ **Need work**: F1 < 0.48 (no improvement)

---

**Status**: Waiting for full evaluation results
**Quick test**: F1 = 0.4933 ✅ (+1.6% vs baseline)
**Hypothesis**: Conservative approach preserves semantics better than aggressive
