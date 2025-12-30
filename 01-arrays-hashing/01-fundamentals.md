# Arrays & Hashing - Fundamentals

## Pattern Recognition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE ARRAYS & HASHING                             │
│                                                                             │
│  Keywords that signal this pattern:                                         │
│  ✓ "Find duplicates"                                                        │
│  ✓ "Count frequency"                                                        │
│  ✓ "Two sum" / "Find pair"                                                  │
│  ✓ "Group by" / "Anagrams"                                                  │
│  ✓ "Contains" / "Exists"                                                    │
│  ✓ "First/Last occurrence"                                                  │
│  ✓ "Unique elements"                                                        │
│                                                                             │
│  Key insight: Trade O(n) space for O(1) lookup time                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before learning this pattern, ensure you understand:
- [ ] Array basics (indexing, iteration)
- [ ] Dictionary/Hash map operations
- [ ] Time complexity basics (O(1), O(n), O(n²))

---

## Memory Map (Pattern Connections)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ARRAYS & HASHING MEMORY MAP                              │
│                                                                             │
│                    ┌─────────────────┐                                      │
│         ┌─────────│ ARRAYS & HASHING│─────────┐                             │
│         │         └─────────────────┘         │                             │
│         ▼                                     ▼                             │
│  ┌─────────────┐                       ┌─────────────┐                      │
│  │   LOOKUP    │                       │  COUNTING   │                      │
│  └──────┬──────┘                       └──────┬──────┘                      │
│         │                                     │                             │
│    ┌────┴────┐                         ┌──────┴──────┐                      │
│    ▼         ▼                         ▼             ▼                      │
│ ┌──────┐ ┌──────┐                 ┌────────┐   ┌──────────┐                │
│ │Two   │ │Contains│               │Frequency│  │ Grouping │                │
│ │Sum   │ │Dup    │                │ Count  │   │(Anagrams)│                │
│ └──────┘ └──────┘                 └────────┘   └──────────┘                │
│                                                                             │
│  Related Patterns:                                                          │
│  • Two Pointers - When array is sorted                                      │
│  • Sliding Window - For subarray problems                                   │
│  • Prefix Sum - For range sum queries                                       │
│                                                                             │
│  When to combine:                                                           │
│  • Hash + Prefix Sum: Subarray sum equals K                                 │
│  • Hash + Two Pointers: After sorting, need original indices                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ARRAYS & HASHING DECISION TREE                           │
│                                                                             │
│  Need O(1) lookup?                                                          │
│       │                                                                     │
│       ├── YES → Use Hash Map/Set                                            │
│       │                                                                     │
│       └── NO → Is array sorted?                                             │
│                    │                                                        │
│                    ├── YES → Consider Binary Search or Two Pointers         │
│                    │                                                        │
│                    └── NO → Need to count frequencies?                      │
│                                 │                                           │
│                                 ├── YES → Use Counter/defaultdict           │
│                                 │                                           │
│                                 └── NO → Need to find pairs?                │
│                                              │                              │
│                                              ├── YES → Hash Map (Two Sum)   │
│                                              │                              │
│                                              └── NO → Basic iteration       │
│                                                                             │
│  QUICK REFERENCE:                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ Problem Type           │ Approach                                  │     │
│  ├────────────────────────┼───────────────────────────────────────────┤     │
│  │ Find pair with sum     │ Hash map: complement lookup               │     │
│  │ Find duplicates        │ Hash set: seen elements                   │     │
│  │ Count frequencies      │ Counter or defaultdict(int)               │     │
│  │ Group elements         │ defaultdict(list) with key function       │     │
│  │ First unique           │ OrderedDict or Counter                    │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concept

Hash maps provide **O(1) average** lookup, insert, and delete operations. This allows us to:

1. **Reduce O(n²) to O(n)** - Instead of nested loops, use hash map lookup
2. **Count frequencies** - Track occurrences of elements
3. **Cache computations** - Store results to avoid recalculation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HASH MAP INTERNALS                                  │
│                                                                             │
│   Key: "apple"                                                              │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────┐                                                               │
│   │  HASH   │  hash("apple") = 42                                           │
│   │ FUNCTION│                                                               │
│   └─────────┘                                                               │
│        │                                                                    │
│        ▼                                                                    │
│   Index = 42 % array_size = 2                                               │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────┬─────┬─────┬─────┬─────┐                                           │
│   │     │     │apple│     │     │  ◄── Array of buckets                     │
│   │     │     │ →5  │     │     │      (bucket 2 stores "apple": 5)         │
│   └─────┴─────┴─────┴─────┴─────┘                                           │
│     0     1     2     3     4                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Essential Templates

### Template 1: Frequency Counter

```python
from collections import Counter, defaultdict

def frequency_counter(arr):
    """Count occurrences of each element"""
    # Method 1: Counter (most Pythonic)
    freq = Counter(arr)

    # Method 2: defaultdict
    freq = defaultdict(int)
    for x in arr:
        freq[x] += 1

    # Method 3: Regular dict
    freq = {}
    for x in arr:
        freq[x] = freq.get(x, 0) + 1

    return freq
```

### Template 2: Two Sum Pattern

```python
def two_sum_pattern(arr, target):
    """Find pair that sums to target"""
    seen = {}  # value -> index
    for i, num in enumerate(arr):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

### Template 3: Grouping Pattern

```python
from collections import defaultdict

def group_by_key(items, key_func):
    """Group items by a computed key"""
    groups = defaultdict(list)
    for item in items:
        key = key_func(item)
        groups[key].append(item)
    return groups

# Example: Group anagrams
def group_anagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))  # Anagrams have same sorted form
        groups[key].append(s)
    return list(groups.values())
```

### Template 4: Set Operations

```python
def find_duplicates(arr):
    """Find duplicate elements"""
    seen = set()
    duplicates = set()
    for x in arr:
        if x in seen:
            duplicates.add(x)
        seen.add(x)
    return list(duplicates)

def find_intersection(arr1, arr2):
    """Find common elements"""
    return list(set(arr1) & set(arr2))

def find_difference(arr1, arr2):
    """Find elements in arr1 but not in arr2"""
    return list(set(arr1) - set(arr2))
```

---

## Complexity Analysis

| Operation | Hash Map | Hash Set | Array |
|-----------|----------|----------|-------|
| Insert | O(1)* | O(1)* | O(n) |
| Lookup | O(1)* | O(1)* | O(n) |
| Delete | O(1)* | O(1)* | O(n) |
| Space | O(n) | O(n) | O(n) |

*Average case. Worst case O(n) due to hash collisions.

---

## Common Mistakes

```python
# ❌ WRONG: Using list for lookup (O(n) each time)
def contains_duplicate_slow(nums):
    seen = []
    for num in nums:
        if num in seen:  # O(n) lookup!
            return True
        seen.append(num)
    return False

# ✅ CORRECT: Using set for lookup (O(1) each time)
def contains_duplicate_fast(nums):
    seen = set()
    for num in nums:
        if num in seen:  # O(1) lookup
            return True
        seen.add(num)
    return False

# ❌ WRONG: Not handling missing keys
def get_frequency_wrong(freq, key):
    return freq[key]  # KeyError if key doesn't exist!

# ✅ CORRECT: Use .get() with default
def get_frequency_correct(freq, key):
    return freq.get(key, 0)
```

---

## Interview Tips

### 1. How to Explain Your Approach
```
"I'll use a hash map to achieve O(1) lookup time. As I iterate through
the array, I'll store each element and check if its complement exists.
This reduces the time complexity from O(n²) to O(n)."
```

### 2. What Interviewers Look For
- **Optimization awareness**: Know when hash map improves complexity
- **Trade-off understanding**: Space vs time trade-offs
- **Edge case handling**: Empty arrays, single elements, duplicates

### 3. Common Follow-up Questions
- "What if there are duplicates?" → Use Counter or handle in logic
- "Can you do it in O(1) space?" → Often not possible, explain why
- "What about hash collisions?" → Mention chaining/open addressing

---

## Related Patterns

- **Two Pointers**: When array is sorted, may not need hash map
- **Sliding Window**: For subarray problems with hash map tracking
- **Prefix Sum**: Combined with hash map for subarray sum problems

---

## Next: Practice Problems

Continue to:
- [02-easy-problems.md](./02-easy-problems.md) - Build foundation
- [03-medium-problems.md](./03-medium-problems.md) - Core techniques
- [04-hard-problems.md](./04-hard-problems.md) - Advanced challenges
