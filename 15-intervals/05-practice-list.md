# Intervals - Complete Practice List

## Organized by Pattern and Difficulty

### Pattern 1: Merge/Insert Intervals

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 56 | [Merge Intervals](https://leetcode.com/problems/merge-intervals/) | Medium | Sort by start, merge overlapping |
| 57 | [Insert Interval](https://leetcode.com/problems/insert-interval/) | Medium | Find position, merge |
| 986 | [Interval List Intersections](https://leetcode.com/problems/interval-list-intersections/) | Medium | Two pointers |
| 1288 | [Remove Covered Intervals](https://leetcode.com/problems/remove-covered-intervals/) | Medium | Sort, track max end |

### Pattern 2: Meeting Rooms

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 252 | [Meeting Rooms](https://leetcode.com/problems/meeting-rooms/) | Easy | Check overlap |
| 253 | [Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/) | Medium | Min heap or sweep line |
| 1094 | [Car Pooling](https://leetcode.com/problems/car-pooling/) | Medium | Sweep line |
| 1109 | [Corporate Flight Bookings](https://leetcode.com/problems/corporate-flight-bookings/) | Medium | Difference array |

### Pattern 3: Scheduling

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 435 | [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/) | Medium | Sort by end, greedy |
| 452 | [Min Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/) | Medium | Sort by end, greedy |
| 646 | [Maximum Length of Pair Chain](https://leetcode.com/problems/maximum-length-of-pair-chain/) | Medium | Sort by end, greedy |
| 1353 | [Max Events That Can Be Attended](https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended/) | Medium | Sort + heap |

### Pattern 4: Calendar/Booking

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 729 | [My Calendar I](https://leetcode.com/problems/my-calendar-i/) | Medium | BST or sorted list |
| 731 | [My Calendar II](https://leetcode.com/problems/my-calendar-ii/) | Medium | Track double bookings |
| 732 | [My Calendar III](https://leetcode.com/problems/my-calendar-iii/) | Hard | Sweep line |

### Pattern 5: Advanced

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 759 | [Employee Free Time](https://leetcode.com/problems/employee-free-time/) | Hard | Merge all, find gaps |
| 218 | [The Skyline Problem](https://leetcode.com/problems/the-skyline-problem/) | Hard | Sweep line + heap |
| 352 | [Data Stream as Disjoint Intervals](https://leetcode.com/problems/data-stream-as-disjoint-intervals/) | Hard | TreeMap |
| 715 | [Range Module](https://leetcode.com/problems/range-module/) | Hard | Sorted intervals |

---

## Essential Templates

### 1. Merge Intervals
```python
def merge(intervals: list[list[int]]) -> list[list[int]]:
    """
    Merge overlapping intervals.

    Time: O(n log n)
    Space: O(n)
    """
    if not intervals:
        return []

    # Sort by start time
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]

    for start, end in intervals[1:]:
        # If overlaps with last merged interval
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return merged
```

### 2. Insert Interval
```python
def insert(intervals: list[list[int]], newInterval: list[int]) -> list[list[int]]:
    """
    Insert and merge new interval.

    Time: O(n)
    Space: O(n)
    """
    result = []
    i = 0
    n = len(intervals)

    # Add all intervals before newInterval
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

### 3. Meeting Rooms II (Min Heap)
```python
import heapq

def minMeetingRooms(intervals: list[list[int]]) -> int:
    """
    Minimum meeting rooms using min heap.

    Time: O(n log n)
    Space: O(n)
    """
    if not intervals:
        return 0

    # Sort by start time
    intervals.sort(key=lambda x: x[0])

    # Min heap of end times
    heap = []

    for start, end in intervals:
        # If earliest ending meeting has ended
        if heap and heap[0] <= start:
            heapq.heappop(heap)

        heapq.heappush(heap, end)

    return len(heap)
```

### 4. Meeting Rooms II (Sweep Line)
```python
def minMeetingRooms_sweep(intervals: list[list[int]]) -> int:
    """
    Minimum meeting rooms using sweep line.

    Time: O(n log n)
    Space: O(n)
    """
    events = []

    for start, end in intervals:
        events.append((start, 1))   # Meeting starts
        events.append((end, -1))    # Meeting ends

    # Sort by time, then by type (end before start at same time)
    events.sort()

    max_rooms = 0
    current_rooms = 0

    for time, delta in events:
        current_rooms += delta
        max_rooms = max(max_rooms, current_rooms)

    return max_rooms
```

### 5. Non-overlapping Intervals (Greedy)
```python
def eraseOverlapIntervals(intervals: list[list[int]]) -> int:
    """
    Minimum removals for non-overlapping intervals.

    Strategy: Sort by END, keep maximum non-overlapping.

    Time: O(n log n)
    Space: O(1)
    """
    if not intervals:
        return 0

    # Sort by end time
    intervals.sort(key=lambda x: x[1])

    count = 1
    end = intervals[0][1]

    for i in range(1, len(intervals)):
        if intervals[i][0] >= end:
            count += 1
            end = intervals[i][1]

    return len(intervals) - count
```

### 6. Interval List Intersections
```python
def intervalIntersection(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    """
    Find intersections of two interval lists.

    Time: O(m + n)
    Space: O(1) excluding output
    """
    result = []
    i = j = 0

    while i < len(A) and j < len(B):
        # Find intersection
        start = max(A[i][0], B[j][0])
        end = min(A[i][1], B[j][1])

        if start <= end:
            result.append([start, end])

        # Move pointer with smaller end
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1

    return result
```

---

## Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INTERVAL OPERATIONS                                      │
│                                                                             │
│  MERGE INTERVALS:                                                           │
│  [1,3] [2,6] [8,10] [15,18]                                                │
│  ├──┤                                                                       │
│    ├────┤                                                                   │
│           ├──┤                                                              │
│                    ├───┤                                                    │
│  Result: [1,6] [8,10] [15,18]                                              │
│                                                                             │
│  SWEEP LINE (Meeting Rooms):                                                │
│  Time:     0  5  10  15  20  25  30                                        │
│  Events:   +1 +1  -1  +1  -1  -1                                           │
│  Rooms:    1  2   1   2   1   0                                            │
│  Max rooms = 2                                                              │
│                                                                             │
│  NON-OVERLAPPING (Sort by END):                                             │
│  [1,2] [2,3] [3,4] [1,3]                                                   │
│  ├┤                        Keep (ends earliest)                             │
│    ├┤                      Keep (doesn't overlap)                           │
│      ├┤                    Keep (doesn't overlap)                           │
│  ├──┤                      Remove (overlaps with kept)                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## When to Use Each Pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INTERVAL PATTERN SELECTION                               │
│                                                                             │
│  MERGE INTERVALS:                                                           │
│  • Combine overlapping intervals                                            │
│  • Sort by START time                                                       │
│                                                                             │
│  SWEEP LINE:                                                                │
│  • Count concurrent events                                                  │
│  • Find maximum overlap                                                     │
│  • Create events for start (+1) and end (-1)                               │
│                                                                             │
│  GREEDY (Sort by END):                                                      │
│  • Maximum non-overlapping intervals                                        │
│  • Minimum removals for non-overlap                                         │
│  • Activity selection                                                       │
│                                                                             │
│  MIN HEAP:                                                                  │
│  • Track active intervals                                                   │
│  • Meeting rooms (track end times)                                          │
│  • Event scheduling                                                         │
│                                                                             │
│  TWO POINTERS:                                                              │
│  • Intersection of sorted interval lists                                    │
│  • Merge two sorted interval lists                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Study Plan

### Week 1: Fundamentals
- [ ] Merge Intervals (LC #56)
- [ ] Insert Interval (LC #57)
- [ ] Meeting Rooms (LC #252)
- [ ] Non-overlapping Intervals (LC #435)

### Week 2: Intermediate
- [ ] Meeting Rooms II (LC #253)
- [ ] Min Arrows to Burst Balloons (LC #452)
- [ ] Interval List Intersections (LC #986)
- [ ] Car Pooling (LC #1094)

### Week 3: Advanced
- [ ] My Calendar I, II, III (LC #729, 731, 732)
- [ ] Employee Free Time (LC #759)
- [ ] The Skyline Problem (LC #218)

---

## Common Mistakes

1. **Wrong sorting criteria**
   - Merge: sort by START
   - Activity selection: sort by END

2. **Boundary conditions**
   - `[1,2]` and `[2,3]`: overlapping or not?
   - Check problem definition carefully

3. **Off-by-one in sweep line**
   - Process end events before start events at same time?
   - Depends on problem (exclusive vs inclusive end)

4. **Not handling empty input**
   - Check for empty array
   - Check for single interval

---

## Complexity Reference

| Pattern | Time | Space |
|---------|------|-------|
| Merge intervals | O(n log n) | O(n) |
| Insert interval | O(n) | O(n) |
| Meeting rooms (heap) | O(n log n) | O(n) |
| Meeting rooms (sweep) | O(n log n) | O(n) |
| Non-overlapping | O(n log n) | O(1) |
| Intersection | O(m + n) | O(1) |
| Skyline | O(n log n) | O(n) |

