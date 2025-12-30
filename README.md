# DSA Patterns - Complete Interview Preparation Course

A comprehensive, pattern-based guide to Data Structures and Algorithms for coding interviews. Master the **mental map** to recognize problems and apply the right patterns.

---

## Course Philosophy

> **Don't memorize solutions. Learn to recognize patterns.**

When you see a new problem, your brain should instantly connect:
- **Keywords** → Pattern
- **Pattern** → Template
- **Template** → Solution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE PATTERN RECOGNITION PIPELINE                         │
│                                                                             │
│  READ PROBLEM                                                               │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │  IDENTIFY KEYWORDS                                               │       │
│  │  • "sorted array" → Binary Search or Two Pointers               │       │
│  │  • "subarray/substring" → Sliding Window or Prefix Sum          │       │
│  │  • "all combinations/permutations" → Backtracking               │       │
│  │  • "shortest path" → BFS (unweighted) or Dijkstra (weighted)    │       │
│  │  • "optimal/min/max" → DP or Greedy                             │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │  APPLY TEMPLATE                                                  │       │
│  │  Each pattern has a reusable code template                       │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │  CUSTOMIZE FOR PROBLEM                                           │       │
│  │  Modify template based on specific constraints                   │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Learning Path

### Phase 1: Foundations (Week 1)
| # | Module | Topics | Est. Time |
|---|--------|--------|-----------|
| 00 | [Foundations](./00-foundations/) | Big O, Data Structures, Recursion | 2-3 days |

### Phase 2: Core Patterns (Weeks 2-4)

> **Note**: The "Problems" column shows **core problems** with detailed solutions. See the [Progress Tracker](#progress-tracker) for the complete count including practice problems.

| # | Pattern | Key Insight | Problems |
|---|---------|-------------|----------|
| 01 | [Arrays & Hashing](./01-arrays-hashing/) | O(1) lookup with hash maps | 12 |
| 02 | [Two Pointers](./02-two-pointers/) | O(n²) → O(n) with sorted/in-place | 12 |
| 03 | [Sliding Window](./03-sliding-window/) | Track subarray/substring efficiently | 12 |
| 04 | [Binary Search](./04-binary-search/) | O(n) → O(log n) on sorted/monotonic | 12 |
| 05 | [Prefix Sum](./05-prefix-sum/) | O(n) range queries | 10 |
| 06 | [Linked List](./06-linked-list/) | Pointer manipulation, fast-slow | 12 |
| 07 | [Stacks & Queues](./07-stacks-queues/) | LIFO/FIFO, monotonic patterns | 12 |

### Phase 3: Tree & Graph Patterns (Weeks 5-7)
| # | Pattern | Key Insight | Problems |
|---|---------|-------------|----------|
| 08 | [Trees - DFS](./08-trees-dfs/) | Recursion = tree traversal | 15 |
| 09 | [Trees - BFS](./09-trees-bfs/) | Level-by-level processing | 10 |
| 10 | [Backtracking](./10-backtracking/) | Explore all → prune invalid | 12 |
| 11 | [Graphs - Basics](./11-graphs-basics/) | DFS/BFS traversal, connected components | 12 |
| 12 | [Graphs - Advanced](./12-graphs-advanced/) | Topo sort, Union-Find, Dijkstra | 12 |

### Phase 4: Advanced Patterns (Weeks 8-12)
| # | Pattern | Key Insight | Problems |
|---|---------|-------------|----------|
| 13 | [Heaps](./13-heaps/) | Top-K, streaming data, K-way merge | 12 |
| 14 | [Dynamic Programming](./14-dynamic-programming/) | Overlapping subproblems → memoize | 20 |
| 15 | [Intervals](./15-intervals/) | Sort by start, merge overlaps | 10 |
| 16 | [Greedy](./16-greedy/) | Local optimal → global optimal | 12 |
| 17 | [Trie](./17-trie/) | Prefix matching in O(L) | 8 |
| 18 | [Bit Manipulation](./18-bit-manipulation/) | XOR tricks, bit operations | 10 |
| 19 | [Math & Geometry](./19-math-geometry/) | Number theory, cyclic sort | 10 |
| 20 | [Design Problems](./20-design-problems/) | System design, data structure design | 10 |

### Quick Reference
| Resource | Description |
|----------|-------------|
| [99-reference](./99-reference/) | Cheat sheet, complexity reference, problem index |

---

## Pattern Recognition Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHICH PATTERN SHOULD I USE?                              │
│                                                                             │
│  START: What does the problem ask for?                                      │
│       │                                                                     │
│       ├──► Find element in sorted array ────────────► BINARY SEARCH        │
│       │                                                                     │
│       ├──► Find pair/triplet with sum ──────────────► TWO POINTERS         │
│       │                                                                     │
│       ├──► Subarray with condition ─────────────────► SLIDING WINDOW       │
│       │    (max length, min length, sum equals K)        or PREFIX SUM     │
│       │                                                                     │
│       ├──► Frequency counting, duplicates ──────────► HASH MAP             │
│       │                                                                     │
│       ├──► Tree traversal/properties ───────────────► DFS (recursion)      │
│       │                                                                     │
│       ├──► Level-by-level tree processing ──────────► BFS                  │
│       │                                                                     │
│       ├──► All combinations/permutations ───────────► BACKTRACKING         │
│       │                                                                     │
│       ├──► Shortest path (unweighted) ──────────────► BFS                  │
│       │                                                                     │
│       ├──► Shortest path (weighted) ────────────────► DIJKSTRA             │
│       │                                                                     │
│       ├──► Connected components ────────────────────► UNION-FIND or DFS    │
│       │                                                                     │
│       ├──► Dependencies/ordering ───────────────────► TOPOLOGICAL SORT     │
│       │                                                                     │
│       ├──► Top K / Kth element ─────────────────────► HEAP                 │
│       │                                                                     │
│       ├──► Count ways / Min cost / Is possible ─────► DYNAMIC PROGRAMMING  │
│       │    with overlapping subproblems                                     │
│       │                                                                     │
│       ├──► Next greater/smaller ────────────────────► MONOTONIC STACK      │
│       │                                                                     │
│       ├──► Overlapping intervals ───────────────────► INTERVALS (sort)     │
│       │                                                                     │
│       ├──► Prefix matching ─────────────────────────► TRIE                 │
│       │                                                                     │
│       └──► Single number, XOR patterns ─────────────► BIT MANIPULATION     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## How to Use This Course

### For Each Pattern:

1. **Read `01-fundamentals.md`** - Understand when/why to use the pattern
2. **Study the template** - Memorize the skeleton code
3. **Solve Easy problems** - Build confidence with basics
4. **Tackle Medium problems** - Apply pattern variations
5. **Challenge with Hard problems** - Master edge cases
6. **Review common mistakes** - Avoid typical pitfalls

### Recommended Study Schedule:

```
Week 1:     Foundations + Arrays/Hashing
Week 2-3:   Two Pointers + Sliding Window + Binary Search
Week 4:     Prefix Sum + Linked List + Stacks
Week 5-6:   Trees (DFS + BFS) + Backtracking
Week 7-8:   Graphs (Basics + Advanced)
Week 9-10:  Heaps + Dynamic Programming (most important!)
Week 11-12: Intervals + Greedy + Remaining patterns
Week 13+:   Review + Mock interviews
```

---

## Progress Tracker

| Pattern | Fundamentals | Easy | Medium | Hard | Total | Mastered |
|---------|:------------:|:----:|:------:|:----:|:-----:|:--------:|
| Arrays & Hashing | ☐ | 14 | 12 | 10 | 36 | ☐ |
| Two Pointers | ☐ | 12 | 10 | 20 | 42 | ☐ |
| Sliding Window | ☐ | 12 | 12 | 20 | 44 | ☐ |
| Binary Search | ☐ | 14 | 12 | 18 | 44 | ☐ |
| Prefix Sum | ☐ | 14 | 14 | 12 | 40 | ☐ |
| Linked List | ☐ | 14 | 16 | 10 | 40 | ☐ |
| Stacks & Queues | ☐ | 16 | 16 | 12 | 44 | ☐ |
| Trees - DFS | ☐ | 16 | 20 | 20 | 56 | ☐ |
| Trees - BFS | ☐ | 20 | 20 | 12 | 52 | ☐ |
| Backtracking | ☐ | 22 | 20 | 12 | 54 | ☐ |
| Graphs - Basics | ☐ | 20 | 20 | 12 | 52 | ☐ |
| Graphs - Advanced | ☐ | 16 | 10 | 20 | 46 | ☐ |
| Heaps | ☐ | 20 | 16 | 12 | 48 | ☐ |
| Dynamic Programming | ☐ | 14 | 12 | 12 | 38 | ☐ |
| Intervals | ☐ | 16 | 20 | 12 | 48 | ☐ |
| Greedy | ☐ | 20 | 16 | 12 | 48 | ☐ |
| Trie | ☐ | 18 | 12 | 12 | 42 | ☐ |
| Bit Manipulation | ☐ | 22 | 18 | 12 | 52 | ☐ |
| Math & Geometry | ☐ | 22 | 18 | 12 | 52 | ☐ |
| Design Problems | ☐ | 16 | 12 | 12 | 40 | ☐ |
| **TOTAL** | 20 | **338** | **306** | **274** | **918** | - |

> **Note on Problem Counts**:
> - The "Total" column includes **all practice problems** across Easy, Medium, and Hard difficulties
> - Some problems appear in multiple patterns (e.g., Two Sum appears in Arrays & Hashing and Two Pointers)
> - **Unique problems**: ~260 distinct LeetCode problems
> - The "Problems" counts in the Learning Path tables above show only **core problems** with detailed solutions

---

## Course Statistics

- **Total Patterns**: 20 core patterns
- **Total Problems**: 250+ with detailed solutions
- **Difficulty Distribution**: ~25% Easy, ~55% Medium, ~20% Hard
- **Estimated Completion**: 12-16 weeks (2-3 hours/day)

---

## Related Resources

- [Blind 75](https://leetcode.com/discuss/general-discussion/460599/blind-75-leetcode-questions) - Classic curated list
- [NeetCode 150](https://neetcode.io/roadmap) - Expanded pattern-based list
- [LeetCode Patterns](https://seanprashad.com/leetcode-patterns/) - Problem categorization

---

## Contributing

Found an error or want to add a problem? Feel free to update and improve!
