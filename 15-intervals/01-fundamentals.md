# Intervals - Fundamentals

## Pattern Recognition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE INTERVAL PATTERNS                            │
│                                                                             │
│  Keywords that signal this pattern:                                         │
│  ✓ "Intervals" / "Ranges"                                                   │
│  ✓ "Overlapping"                                                            │
│  ✓ "Merge intervals"                                                        │
│  ✓ "Meeting rooms"                                                          │
│  ✓ "Insert interval"                                                        │
│  ✓ "Minimum number of intervals"                                            │
│                                                                             │
│  Key insight: Sort by start (or end) time, then process linearly            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before learning this pattern, ensure you understand:
- [ ] Sorting with custom key
- [ ] Overlap detection logic
- [ ] Greedy algorithm concept

---

## Memory Map (Pattern Connections)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INTERVALS MEMORY MAP                                     │
│                                                                             │
│                    ┌───────────┐                                            │
│         ┌─────────│ INTERVALS │─────────┐                                   │
│         │         └───────────┘         │                                   │
│         ▼                               ▼                                   │
│  ┌─────────────┐                 ┌─────────────┐                            │
│  │   MERGE     │                 │  SCHEDULING │                            │
│  │  PROBLEMS   │                 │  PROBLEMS   │                            │
│  └──────┬──────┘                 └──────┬──────┘                            │
│         │                               │                                   │
│    ┌────┴────┐                    ┌─────┴─────┐                             │
│    ▼         ▼                    ▼           ▼                             │
│ ┌──────┐ ┌──────┐             ┌──────┐   ┌──────┐                          │
│ │Merge │ │Insert│             │Meeting│  │Min   │                          │
│ │Intvls│ │Intvl │             │Rooms │   │Remove│                          │
│ └──────┘ └──────┘             └──────┘   └──────┘                          │
│                                                                             │
│  Related Patterns:                                                          │
│  • Greedy - Many interval problems use greedy                               │
│  • Sorting - Almost always sort first                                       │
│  • Heap - For meeting rooms (concurrent intervals)                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INTERVAL PROBLEM DECISION TREE                           │
│                                                                             │
│  What's the goal?                                                           │
│       │                                                                     │
│       ├── Merge overlapping → Sort by start, extend end                     │
│       │                                                                     │
│       ├── Find overlaps → Sort by start, check prev.end >= curr.start       │
│       │                                                                     │
│       ├── Min rooms needed → Sort events, track concurrent (heap/counter)   │
│       │                                                                     │
│       └── Min removals for no overlap → Sort by END, greedy selection       │
│                                                                             │
│  SORTING KEY SELECTION:                                                     │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ Problem                    │ Sort By                               │     │
│  ├────────────────────────────┼───────────────────────────────────────┤     │
│  │ Merge intervals            │ Start time                            │     │
│  │ Insert interval            │ Start time                            │     │
│  │ Min intervals to remove    │ End time (greedy)                     │     │
│  │ Meeting rooms              │ Start time (with heap for ends)       │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concept

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INTERVAL RELATIONSHIPS                                   │
│                                                                             │
│  Two intervals [a, b] and [c, d] can have these relationships:              │
│                                                                             │
│  1. NO OVERLAP (a.end < b.start):                                           │
│     [a────b]                                                                │
│               [c────d]                                                      │
│                                                                             │
│  2. OVERLAP (a.end >= b.start):                                             │
│     [a────b]                                                                │
│         [c────d]                                                            │
│     Merged: [a──────d]                                                      │
│                                                                             │
│  3. CONTAINMENT (a contains b):                                             │
│     [a──────────b]                                                          │
│        [c──d]                                                               │
│     Merged: [a──────────b]                                                  │
│                                                                             │
│  Overlap check: a.start <= b.end AND b.start <= a.end                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Common Interval Problems

### Problem 1: Merge Intervals (LC #56)

```python
def merge(intervals: list[list[int]]) -> list[list[int]]:
    """
    Merge all overlapping intervals.

    Strategy:
    1. Sort by start time
    2. Iterate through intervals
    3. If current overlaps with last merged, extend last merged
    4. Otherwise, add current as new interval

    Time: O(n log n) for sorting
    Space: O(n) for result
    """
    if not intervals:
        return []

    # Sort by start time
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]

    for current in intervals[1:]:
        last = merged[-1]

        # Check for overlap: current.start <= last.end
        if current[0] <= last[1]:
            # Merge: extend end of last interval
            last[1] = max(last[1], current[1])
        else:
            # No overlap: add as new interval
            merged.append(current)

    return merged
```

### Problem 2: Insert Interval (LC #57)

```python
def insert(intervals: list[list[int]], newInterval: list[int]) -> list[list[int]]:
    """
    Insert new interval into sorted, non-overlapping intervals.

    Strategy:
    1. Add all intervals that end before new interval starts
    2. Merge all overlapping intervals with new interval
    3. Add all intervals that start after new interval ends

    Time: O(n)
    Space: O(n)
    """
    result = []
    i = 0
    n = len(intervals)

    # Add all intervals ending before newInterval starts
    while i < n and intervals[i][1] < newInterval[0]:
        result.append(intervals[i])
        i += 1

    # Merge overlapping intervals
    while i < n and intervals[i][0] <= newInterval[1]:
        newInterval[0] = min(newInterval[0], intervals[i][0])
        newInterval[1] = max(newInterval[1], intervals[i][1])
        i += 1

    result.append(newInterval)

    # Add remaining intervals
    while i < n:
        result.append(intervals[i])
        i += 1

    return result
```

### Problem 3: Non-overlapping Intervals (LC #435)

```python
def eraseOverlapIntervals(intervals: list[list[int]]) -> int:
    """
    Minimum intervals to remove for non-overlapping.

    Strategy (Greedy):
    - Sort by END time (not start!)
    - Always keep interval that ends earliest
    - This leaves most room for future intervals

    Time: O(n log n)
    Space: O(1)
    """
    if not intervals:
        return 0

    # Sort by end time
    intervals.sort(key=lambda x: x[1])

    count = 0  # Intervals to remove
    prev_end = float('-inf')

    for start, end in intervals:
        if start >= prev_end:
            # No overlap - keep this interval
            prev_end = end
        else:
            # Overlap - remove this interval (keep previous)
            count += 1

    return count
```

### Problem 4: Meeting Rooms (LC #252)

```python
def canAttendMeetings(intervals: list[list[int]]) -> bool:
    """
    Can a person attend all meetings?

    Strategy:
    - Sort by start time
    - Check if any meeting starts before previous ends

    Time: O(n log n)
    Space: O(1)
    """
    intervals.sort(key=lambda x: x[0])

    for i in range(1, len(intervals)):
        # If current starts before previous ends -> conflict
        if intervals[i][0] < intervals[i-1][1]:
            return False

    return True
```

### Problem 5: Meeting Rooms II (LC #253)

```python
import heapq

def minMeetingRooms(intervals: list[list[int]]) -> int:
    """
    Minimum meeting rooms required.

    Strategy (Min Heap):
    - Sort by start time
    - Use heap to track end times of ongoing meetings
    - For each meeting:
      - Remove ended meetings (end time <= current start)
      - Add current meeting's end time
    - Max heap size = rooms needed

    Time: O(n log n)
    Space: O(n)
    """
    if not intervals:
        return 0

    # Sort by start time
    intervals.sort(key=lambda x: x[0])

    # Min heap of end times
    rooms = []
    heapq.heappush(rooms, intervals[0][1])

    for i in range(1, len(intervals)):
        # If earliest ending meeting has ended, reuse that room
        if rooms[0] <= intervals[i][0]:
            heapq.heappop(rooms)

        # Add current meeting's end time
        heapq.heappush(rooms, intervals[i][1])

    return len(rooms)


def minMeetingRooms_sweep(intervals: list[list[int]]) -> int:
    """
    Alternative: Line sweep algorithm.

    Strategy:
    - Create events: +1 at start, -1 at end
    - Sort events by time
    - Track running count of concurrent meetings

    Time: O(n log n)
    Space: O(n)
    """
    events = []

    for start, end in intervals:
        events.append((start, 1))   # Meeting starts
        events.append((end, -1))    # Meeting ends

    # Sort by time, with ends before starts at same time
    events.sort(key=lambda x: (x[0], x[1]))

    max_rooms = 0
    current_rooms = 0

    for time, delta in events:
        current_rooms += delta
        max_rooms = max(max_rooms, current_rooms)

    return max_rooms
```

### Problem 6: Interval List Intersections (LC #986)

```python
def intervalIntersection(firstList: list[list[int]], secondList: list[list[int]]) -> list[list[int]]:
    """
    Find intersection of two interval lists.

    Strategy (Two Pointers):
    - Use two pointers, one for each list
    - Find intersection of current intervals
    - Move pointer of interval that ends first

    Time: O(m + n)
    Space: O(1) excluding output
    """
    result = []
    i = j = 0

    while i < len(firstList) and j < len(secondList):
        a_start, a_end = firstList[i]
        b_start, b_end = secondList[j]

        # Find intersection
        start = max(a_start, b_start)
        end = min(a_end, b_end)

        # If valid intersection, add it
        if start <= end:
            result.append([start, end])

        # Move pointer of interval that ends first
        if a_end < b_end:
            i += 1
        else:
            j += 1

    return result
```

### Problem 7: Minimum Number of Arrows to Burst Balloons (LC #452)

```python
def findMinArrowShots(points: list[list[int]]) -> int:
    """
    Minimum arrows to burst all balloons.

    Strategy (Greedy):
    - Sort by end position
    - Shoot arrow at end of first balloon
    - Skip all balloons that arrow passes through
    - Repeat for remaining balloons

    Time: O(n log n)
    Space: O(1)
    """
    if not points:
        return 0

    # Sort by end position
    points.sort(key=lambda x: x[1])

    arrows = 1
    arrow_pos = points[0][1]  # Shoot at end of first balloon

    for start, end in points[1:]:
        if start > arrow_pos:
            # Balloon not hit by current arrow
            arrows += 1
            arrow_pos = end  # New arrow at end of this balloon

    return arrows
```

---

## Visual: Meeting Rooms Problem

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MEETING ROOMS II EXAMPLE                                 │
│                                                                             │
│  Meetings: [[0,30], [5,10], [15,20]]                                        │
│                                                                             │
│  Timeline:                                                                  │
│  0    5    10   15   20   25   30                                          │
│  |────|────|────|────|────|────|                                           │
│  [═══════════════════════════════]  Room 1: [0,30]                         │
│       [════]                        Room 2: [5,10]                         │
│                 [════]              Room 2: [15,20] (reused)               │
│                                                                             │
│  At time 5: 2 meetings ongoing → need 2 rooms                               │
│  At time 15: only 1 meeting ([0,30]) → Room 2 available                    │
│                                                                             │
│  Answer: 2 rooms needed                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Complexity Summary

| Problem | Time | Space | Key Technique |
|---------|------|-------|---------------|
| Merge Intervals | O(n log n) | O(n) | Sort by start |
| Insert Interval | O(n) | O(n) | Three-phase scan |
| Non-overlapping | O(n log n) | O(1) | Sort by end, greedy |
| Meeting Rooms I | O(n log n) | O(1) | Sort, check overlap |
| Meeting Rooms II | O(n log n) | O(n) | Heap or sweep line |
| Intersections | O(m + n) | O(1) | Two pointers |
| Burst Balloons | O(n log n) | O(1) | Sort by end, greedy |

---

## Common Mistakes

```python
# ❌ WRONG: Sorting by start when should sort by end
def eraseOverlapIntervals_wrong(intervals):
    intervals.sort(key=lambda x: x[0])  # Wrong! Sort by end
    # ...

# ✅ CORRECT: Sort by end for greedy interval scheduling
def eraseOverlapIntervals_correct(intervals):
    intervals.sort(key=lambda x: x[1])  # Correct!
    # ...


# ❌ WRONG: Not handling edge case of touching intervals
def merge_wrong(intervals):
    # [1,2] and [2,3] - are these overlapping?
    if current[0] < last[1]:  # Wrong! Should be <=
        # merge
    # ...

# ✅ CORRECT: Touching intervals should merge
def merge_correct(intervals):
    if current[0] <= last[1]:  # Correct!
        # merge
    # ...
```

---

## Interview Tips

### 1. How to Explain Your Approach
```
"First, I'll sort intervals by start time. Then I iterate through,
merging overlapping intervals by extending the end time. Two intervals
overlap if the current start <= previous end."
```

### 2. What Interviewers Look For
- **Sorting key choice**: Start vs end depends on problem
- **Overlap condition**: curr.start <= prev.end (not <)
- **Edge cases**: Empty input, single interval, all overlapping

### 3. Common Follow-up Questions
- "What if intervals are already sorted?" → Skip sorting step
- "How many meeting rooms needed?" → Track concurrent with heap
- "What about inserting into sorted intervals?" → Binary search + merge

---

## Next: Practice Problems

Continue to:
- [02-easy-problems.md](./02-easy-problems.md) - Build foundation
- [03-medium-problems.md](./03-medium-problems.md) - Core techniques
- [04-hard-problems.md](./04-hard-problems.md) - Advanced challenges
