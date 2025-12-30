# Problem Solving Framework

## The UMPIRE Method

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    U.M.P.I.R.E. METHOD                                      │
│                                                                             │
│  U - UNDERSTAND the problem                                                 │
│  M - MATCH to known patterns                                                │
│  P - PLAN your approach                                                     │
│  I - IMPLEMENT the solution                                                 │
│  R - REVIEW your code                                                       │
│  E - EVALUATE complexity and alternatives                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: UNDERSTAND (2-3 minutes)

### Questions to Ask
```
1. INPUT:
   - What is the input format?
   - What are the constraints? (size, range, types)
   - Can input be empty? Null? Negative?

2. OUTPUT:
   - What should I return?
   - What if no solution exists?
   - Should I return indices or values?

3. EXAMPLES:
   - Walk through given examples
   - Create your own edge cases
   - Verify understanding with interviewer

4. CLARIFICATIONS:
   - Can I modify the input?
   - Is the input sorted?
   - Are there duplicates?
   - How should I handle ties?
```

### Example: Two Sum
```
Understanding:
- Input: Array of integers, target sum
- Output: Indices of two numbers that add to target
- Constraints: Exactly one solution exists
- Questions: Can same element be used twice? (No)
- Edge cases: Negative numbers? (Yes, allowed)
```

---

## Step 2: MATCH (1-2 minutes)

### Pattern Recognition Keywords

| If you see... | Think... |
|---------------|----------|
| "Sorted array" | Binary Search, Two Pointers |
| "Find all combinations" | Backtracking |
| "Shortest path" | BFS (unweighted), Dijkstra (weighted) |
| "Subarray/substring" | Sliding Window, Prefix Sum |
| "Tree traversal" | DFS, BFS |
| "Connected components" | Union-Find, DFS |
| "Top K elements" | Heap |
| "Optimal/min/max" | DP, Greedy |
| "String matching" | Trie, KMP |
| "Intervals" | Sort + Greedy |

### Common Data Structure Matches

```python
# O(1) lookup → Hash Map/Set
# O(log n) search → Binary Search, BST, Heap
# O(1) min/max → Monotonic Stack/Queue
# FIFO → Queue
# LIFO → Stack
# Prefix operations → Trie
# Dynamic connectivity → Union-Find
```

---

## Step 3: PLAN (3-5 minutes)

### Planning Template
```
1. ALGORITHM:
   - Describe in plain English
   - Step by step process

2. DATA STRUCTURES:
   - What will I use?
   - Why this choice?

3. COMPLEXITY:
   - Expected time complexity
   - Expected space complexity

4. EDGE CASES:
   - Empty input
   - Single element
   - All same values
   - Maximum/minimum values
```

### Example Plan: Two Sum
```
Algorithm:
1. Create hash map to store {value: index}
2. For each number, check if (target - number) exists in map
3. If found, return both indices
4. Otherwise, add current number to map

Data Structures:
- Hash map for O(1) lookup

Complexity:
- Time: O(n) - single pass
- Space: O(n) - hash map

Edge Cases:
- Two same numbers: [3, 3], target = 6
- Negative numbers: [-1, 2], target = 1
```

---

## Step 4: IMPLEMENT (15-20 minutes)

### Coding Best Practices

```python
def solution(input_params):
    """
    Clear description of what function does.

    Args:
        input_params: Description of inputs

    Returns:
        Description of output

    Example:
        >>> solution([1, 2, 3])
        expected_output
    """
    # Handle edge cases first
    if not input_params:
        return default_value

    # Initialize data structures
    result = []
    helper_map = {}

    # Main logic with clear comments
    for item in input_params:
        # Explain what this step does
        processed = process(item)
        result.append(processed)

    return result
```

### Common Patterns

```python
# Two Pointers
left, right = 0, len(arr) - 1
while left < right:
    # Process
    if condition:
        left += 1
    else:
        right -= 1

# Sliding Window
left = 0
for right in range(len(arr)):
    # Expand window
    while invalid():
        # Shrink window
        left += 1
    # Update result

# DFS
def dfs(node, state):
    if base_case:
        return result

    for choice in choices:
        make_choice()
        dfs(next_node, new_state)
        undo_choice()

# BFS
from collections import deque
queue = deque([start])
visited = {start}
while queue:
    current = queue.popleft()
    for neighbor in get_neighbors(current):
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append(neighbor)
```

---

## Step 5: REVIEW (3-5 minutes)

### Review Checklist

```
□ Trace through with given example
□ Check edge cases:
  □ Empty input
  □ Single element
  □ Duplicates
  □ Negative values
  □ Maximum values

□ Look for bugs:
  □ Off-by-one errors
  □ Wrong comparison operators
  □ Missing return statements
  □ Uninitialized variables
  □ Integer overflow

□ Code quality:
  □ Meaningful variable names
  □ Proper indentation
  □ Necessary comments
```

### Common Bugs to Check

```python
# Off-by-one
for i in range(n):      # 0 to n-1
for i in range(1, n+1): # 1 to n

# Comparison
< vs <=
> vs >=

# Array bounds
arr[i-1]  # Check i > 0
arr[i+1]  # Check i < len(arr) - 1

# Division
a // b    # Integer division
a / b     # Float division
b != 0    # Check for zero
```

---

## Step 6: EVALUATE (2-3 minutes)

### Complexity Analysis

```
Time Complexity:
- Count operations as function of input size
- Focus on dominant term
- Consider best, average, worst cases

Space Complexity:
- Count additional memory used
- Don't count input (usually)
- Consider recursion stack

Common Complexities:
O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2^n) < O(n!)
```

### Discussion Points

```
1. TRADE-OFFS:
   - Time vs Space
   - Readability vs Performance

2. OPTIMIZATIONS:
   - Can we reduce time complexity?
   - Can we reduce space complexity?

3. ALTERNATIVES:
   - What other approaches exist?
   - Why did we choose this one?

4. SCALABILITY:
   - How does it handle large inputs?
   - What if constraints changed?
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INTERVIEW QUICK REFERENCE                                │
│                                                                             │
│  BEFORE CODING:                                                             │
│  • Clarify inputs, outputs, constraints                                     │
│  • Work through examples                                                    │
│  • Identify pattern                                                         │
│  • Explain approach and get buy-in                                          │
│                                                                             │
│  WHILE CODING:                                                              │
│  • Talk through your thought process                                        │
│  • Handle edge cases first                                                  │
│  • Use meaningful names                                                     │
│  • Write clean, modular code                                                │
│                                                                             │
│  AFTER CODING:                                                              │
│  • Trace through with example                                               │
│  • Test edge cases                                                          │
│  • Analyze complexity                                                       │
│  • Discuss trade-offs                                                       │
│                                                                             │
│  IF STUCK:                                                                  │
│  • "Let me think about this..."                                             │
│  • Try brute force first                                                    │
│  • Ask for a hint                                                           │
│  • Simplify the problem                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Practice Exercises

### Exercise 1: Apply UMPIRE to "Valid Palindrome"
```
Given a string, determine if it is a palindrome,
considering only alphanumeric characters and ignoring cases.

Try applying each step of UMPIRE before looking at solutions.
```

### Exercise 2: Pattern Matching
```
For each problem, identify the most likely pattern:

1. Find the kth largest element in an array
2. Detect a cycle in a linked list
3. Find all subsets of a set
4. Merge k sorted lists
5. Find shortest path in unweighted graph

Answers: 1-Heap, 2-Fast/Slow, 3-Backtracking, 4-Heap, 5-BFS
```

### Exercise 3: Time Yourself
```
Practice with a timer:
- Understanding: 2-3 min
- Planning: 3-5 min
- Coding: 15-20 min
- Review: 3-5 min
- Discussion: 2-3 min

Total: ~30 minutes per problem
```

