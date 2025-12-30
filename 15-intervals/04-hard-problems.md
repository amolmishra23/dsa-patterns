# Intervals - Hard Problems

## Problem 1: Employee Free Time (LC #759) - Hard

- [LeetCode](https://leetcode.com/problems/employee-free-time/)

### Video Explanation
- [NeetCode - Employee Free Time](https://www.youtube.com/watch?v=xNe0VN3-sxg)

### Problem Statement
Find common free time intervals for all employees.


### Visual Intuition
```
Employee Free Time
schedule = [[[1,2],[5,6]], [[1,3]], [[4,10]]]

Pattern: Flatten, Sort, Merge, Find Gaps
Why: Free time = gaps between merged busy intervals

Step 0 (Visualize Each Employee's Schedule):

  Employee 0: ██       ██
              1--2     5--6

  Employee 1: ███
              1---3

  Employee 2:       ██████████
                    4--------10

  Timeline:  1  2  3  4  5  6  7  8  9  10

Step 1 (Flatten All Intervals):

  All intervals: [1,2], [5,6], [1,3], [4,10]

  Sorted by start: [1,2], [1,3], [4,10], [5,6]

Step 2 (Merge Overlapping - Find Busy Time):

  [1,2] + [1,3]:
    1 ≤ 2 (overlap) → merge to [1,3]

  [1,3] + [4,10]:
    4 > 3 (gap!) → FREE TIME: [3,4]
    Keep [4,10]

  [4,10] + [5,6]:
    5 ≤ 10 (contained) → stays [4,10]

  Merged busy: [1,3], [4,10]

Step 3 (Find Gaps = Free Time):

  Timeline:  1  2  3  4  5  6  7  8  9  10
  Busy:      ██████  ████████████████████
                  ↑↑
               FREE TIME [3,4]

Answer: [[3,4]]

Merge Algorithm:
  ┌─────────────────────────────────────────────────┐
  │ prev_end = first interval's end                 │
  │                                                 │
  │ for each interval [start, end]:                 │
  │   if start > prev_end:                          │
  │     FREE TIME: [prev_end, start]                │
  │   prev_end = max(prev_end, end)                 │
  └─────────────────────────────────────────────────┘

Key Insight:
- Flatten all schedules into one list
- Sort by start time
- Merge overlapping = busy time
- Gaps between merged = free time
- O(n log n) for sorting
```

### Solution
```python
import heapq

def employeeFreeTime(schedule: list[list[list[int]]]) -> list[list[int]]:
    """
    Merge all intervals and find gaps.

    Time: O(n log n)
    Space: O(n)
    """
    # Flatten and sort all intervals
    intervals = []
    for emp in schedule:
        for interval in emp:
            intervals.append(interval)

    intervals.sort(key=lambda x: x[0])

    result = []
    prev_end = intervals[0][1]

    for start, end in intervals[1:]:
        if start > prev_end:
            result.append([prev_end, start])
        prev_end = max(prev_end, end)

    return result
```

### Edge Cases
- Single employee → no common free time
- No overlap → many free intervals
- All overlap → no free time
- Adjacent intervals → no gap

---

## Problem 2: Data Stream as Disjoint Intervals (LC #352) - Hard

- [LeetCode](https://leetcode.com/problems/data-stream-as-disjoint-intervals/)

### Video Explanation
- [NeetCode - Data Stream as Disjoint Intervals](https://www.youtube.com/watch?v=FavoZhDsID4)

### Problem Statement
Track numbers and return disjoint intervals.


### Visual Intuition
```
Data Stream as Disjoint Intervals
addNum sequence: 1, 3, 7, 2, 6

Pattern: Sorted Structure + Neighbor Merge
Why: New number may extend or merge existing intervals

Step 0 (Process Each Number):

  Add 1: No neighbors
         intervals = [[1,1]]
         Timeline: █
                   1

  Add 3: No neighbors (2 not present)
         intervals = [[1,1], [3,3]]
         Timeline: █   █
                   1   3

  Add 7: No neighbors
         intervals = [[1,1], [3,3], [7,7]]
         Timeline: █   █       █
                   1   3       7

  Add 2: Check neighbors!
         2-1=1 exists in [1,1] → extend right
         2+1=3 exists in [3,3] → extend left
         MERGE both: [1,3]
         intervals = [[1,3], [7,7]]
         Timeline: █████       █
                   1   3       7

  Add 6: Check neighbors!
         6-1=5 not in any interval
         6+1=7 exists in [7,7] → extend left
         intervals = [[1,3], [6,7]]
         Timeline: █████   ███
                   1   3   6 7

Step 1 (Neighbor Check Logic):

  ┌─────────────────────────────────────────────────┐
  │ For new number val:                             │
  │                                                 │
  │ 1. Check if val already covered → skip          │
  │                                                 │
  │ 2. Check left neighbor (val-1):                 │
  │    If in interval [a,b] where b = val-1         │
  │    → extend: [a, val]                           │
  │                                                 │
  │ 3. Check right neighbor (val+1):                │
  │    If in interval [a,b] where a = val+1         │
  │    → extend: [val, b]                           │
  │                                                 │
  │ 4. If both neighbors exist:                     │
  │    → merge into one interval                    │
  └─────────────────────────────────────────────────┘

Step 2 (Merge Example):

  Before adding 2:
    [1,1]     [3,3]
      █         █
      1    2    3
           ↑
         add 2

  After adding 2:
    [1,3]
    █████
    1 2 3

  Two intervals merged into one!

Data Structure:
  Use SortedList/TreeMap for O(log n) operations
  - Binary search to find position
  - Check adjacent intervals for merge
  - O(log n) add, O(n) getIntervals

Key Insight:
- Each number either: creates new interval, extends one, or merges two
- Sorted structure enables O(log n) neighbor lookup
- Disjoint = no overlapping intervals
```

### Solution
```python
from sortedcontainers import SortedList

class SummaryRanges:
    """
    Maintain sorted intervals with merge on add.

    Time: O(log n) per add, O(n) for getIntervals
    Space: O(n)
    """

    def __init__(self):
        self.intervals = SortedList(key=lambda x: x[0])

    def addNum(self, val: int) -> None:
        # Find position to insert
        left = right = None

        # Check for overlapping/adjacent intervals
        idx = self.intervals.bisect_left([val, val])

        # Check left neighbor
        if idx > 0:
            prev = self.intervals[idx - 1]
            if prev[1] >= val:
                return  # Already covered
            if prev[1] == val - 1:
                left = idx - 1

        # Check right neighbor
        if idx < len(self.intervals):
            next_int = self.intervals[idx]
            if next_int[0] <= val <= next_int[1]:
                return  # Already covered
            if next_int[0] == val + 1:
                right = idx

        # Merge or insert
        if left is not None and right is not None:
            # Merge both
            new_interval = [self.intervals[left][0], self.intervals[right][1]]
            self.intervals.pop(right)
            self.intervals.pop(left)
            self.intervals.add(new_interval)
        elif left is not None:
            # Extend left
            self.intervals[left][1] = val
        elif right is not None:
            # Extend right
            self.intervals[right][0] = val
        else:
            # New interval
            self.intervals.add([val, val])

    def getIntervals(self) -> list[list[int]]:
        return list(self.intervals)
```

### Edge Cases
- First number → creates [val, val]
- Consecutive numbers → merge intervals
- Duplicate add → no change
- Gaps in sequence → separate intervals

---

## Problem 3: Minimum Interval to Include Each Query (LC #1851) - Hard

- [LeetCode](https://leetcode.com/problems/minimum-interval-to-include-each-query/)

### Video Explanation
- [NeetCode - Minimum Interval to Include Each Query](https://www.youtube.com/watch?v=5hQ5WWW5awQ)

### Problem Statement
For each query, find smallest interval containing it.


### Visual Intuition
```
Minimum Interval to Include Each Query
intervals = [[1,4],[2,4],[3,6],[4,4]], queries = [2,3,4,5]

Pattern: Sort Both + Min-Heap by Size
Why: Process queries in order, heap gives smallest valid interval

Step 0 (Visualize Intervals):

  [1,4]:  ████████
          1  2  3  4
  [2,4]:     ██████
             2  3  4
  [3,6]:        █████████
                3  4  5  6
  [4,4]:           █
                   4

  Query points: 2, 3, 4, 5

Step 1 (Sort and Setup):

  Intervals sorted by start: [1,4], [2,4], [3,6], [4,4]
  Queries sorted by value: (2, idx0), (3, idx1), (4, idx2), (5, idx3)

  Min-heap by size: (size, end)

Step 2 (Process Query 2):

  Add intervals starting ≤ 2:
    [1,4]: size = 4-1+1 = 4, heap.push((4, 4))
    [2,4]: size = 4-2+1 = 3, heap.push((3, 4))

  Remove intervals ending < 2: none

  Heap: [(3,4), (4,4)]
         ↑
        smallest size = 3

  Answer for query 2: 3

Step 3 (Process Query 3):

  Add intervals starting ≤ 3:
    [3,6]: size = 4, heap.push((4, 6))

  Remove intervals ending < 3: none

  Heap: [(3,4), (4,4), (4,6)]
         ↑
        smallest = 3

  Answer for query 3: 3

Step 4 (Process Query 4):

  Add intervals starting ≤ 4:
    [4,4]: size = 1, heap.push((1, 4))

  Remove intervals ending < 4: none

  Heap: [(1,4), (3,4), (4,4), (4,6)]
         ↑
        smallest = 1

  Answer for query 4: 1

Step 5 (Process Query 5):

  No new intervals to add

  Remove intervals ending < 5:
    (1,4): end=4 < 5 → remove
    (3,4): end=4 < 5 → remove
    (4,4): end=4 < 5 → remove

  Heap: [(4,6)]

  Answer for query 5: 4

Final Answers (restore original order):
  Query 2 → 3
  Query 3 → 3
  Query 4 → 1
  Query 5 → 4

Key Insight:
- Sort queries to process in order (add intervals progressively)
- Min-heap by size gives smallest containing interval
- Remove expired intervals (end < query) from heap
- O((n+q) log n) time
```

### Solution
```python
import heapq

def minInterval(intervals: list[list[int]], queries: list[int]) -> list[int]:
    """
    Sort intervals and queries, use min heap.

    Time: O((n + q) log n)
    Space: O(n + q)
    """
    # Sort intervals by start
    intervals.sort()

    # Process queries in sorted order
    sorted_queries = sorted(enumerate(queries), key=lambda x: x[1])

    result = [-1] * len(queries)
    heap = []  # (size, end)
    i = 0

    for idx, q in sorted_queries:
        # Add all intervals starting <= q
        while i < len(intervals) and intervals[i][0] <= q:
            start, end = intervals[i]
            heapq.heappush(heap, (end - start + 1, end))
            i += 1

        # Remove intervals ending before q
        while heap and heap[0][1] < q:
            heapq.heappop(heap)

        if heap:
            result[idx] = heap[0][0]

    return result
```

### Edge Cases
- Query outside all intervals → return -1
- Query at interval boundary → include it
- Multiple intervals contain query → pick smallest
- All same size intervals → any containing one

---

## Problem 4: My Calendar III (LC #732) - Hard

- [LeetCode](https://leetcode.com/problems/my-calendar-iii/)

### Video Explanation
- [NeetCode - My Calendar III](https://www.youtube.com/watch?v=olRvqGUjXOg)

### Problem Statement
Return maximum k-booking at any time after each booking.

### Visual Intuition
```
My Calendar III - Maximum K-Booking
Events: [10,20], [50,60], [10,40], [5,15], [5,10], [25,55]

Pattern: Sweep Line Algorithm
Why: Track concurrent events at each time point

Step 0 (Mark Events on Timeline):

  Each interval [start, end):
    +1 at start (event begins)
    -1 at end (event ends)

  [10,20]: +1 at 10, -1 at 20
  [50,60]: +1 at 50, -1 at 60
  [10,40]: +1 at 10, -1 at 40
  [5,15]:  +1 at 5,  -1 at 15
  [5,10]:  +1 at 5,  -1 at 10
  [25,55]: +1 at 25, -1 at 55

Step 1 (Build Timeline Map):

  Time:   5   10   15   20   25   40   50   55   60
  Delta: +2  +2   -1   -1   +1   -1   +1   -1   -1
          ↑   ↑
         [5,15] and [5,10] start
         [10,20] and [10,40] start

Step 2 (Sweep and Count):

  Time    Delta   Running Count
  ────────────────────────────────
    5      +2         2
   10      +2         4... wait, let me recalculate

  Actually: at time 10, [5,10] ends (-1)

  Let me re-trace:
  Time:   5   10   15   20   25   40   50   55   60
  Delta: +2  +1   -1   -1   +1   -1   +1   -1   -1
               ↑
         +2 (two start) -1 (one ends) = +1 net

  Running count:
    5: 0 + 2 = 2
   10: 2 + 1 = 3 ★ MAX
   15: 3 - 1 = 2
   20: 2 - 1 = 1
   25: 1 + 1 = 2
   40: 2 - 1 = 1
   50: 1 + 1 = 2
   55: 2 - 1 = 1
   60: 1 - 1 = 0

Step 3 (Visualize Overlaps):

  Time:  5    10   15   20   25   40   50   55   60
         ├────┤         [5,10]
         ├─────────┤    [5,15]
              ├─────────┤    [10,20]
              ├──────────────────┤    [10,40]
                        ├─────────────────┤    [25,55]
                                  ├─────────┤    [50,60]

  At time 10: 3 events overlap ★

Answer: 3

Sweep Line Algorithm:
  ┌─────────────────────────────────────────────────┐
  │ timeline = {}                                   │
  │                                                 │
  │ def book(start, end):                           │
  │     timeline[start] += 1                        │
  │     timeline[end] -= 1                          │
  │                                                 │
  │     max_booking = current = 0                   │
  │     for time in sorted(timeline):              │
  │         current += timeline[time]               │
  │         max_booking = max(max_booking, current) │
  │     return max_booking                          │
  └─────────────────────────────────────────────────┘

Key Insight:
- Sweep line converts intervals to point events
- Running sum = concurrent count at each point
- O(n²) per book (sort each time), O(n log n) with segment tree
```


### Intuition
```
Bookings: [10,20], [50,60], [10,40], [5,15], [5,10], [25,55]

Timeline events (start = +1, end = -1):
5: +1, 10: +1 -1 +1, 15: -1, 20: -1, 25: +1, 40: -1, 50: +1, 55: -1, 60: -1

Sweep line to find max concurrent bookings.
```

### Solution
```python
from collections import defaultdict

class MyCalendarThree:
    """
    Sweep line algorithm with sorted events.

    Time: O(n²) per book (can optimize with segment tree)
    Space: O(n)
    """

    def __init__(self):
        self.timeline = defaultdict(int)

    def book(self, start: int, end: int) -> int:
        # Mark start and end events
        self.timeline[start] += 1
        self.timeline[end] -= 1

        # Sweep through timeline
        max_booking = 0
        current = 0

        for time in sorted(self.timeline.keys()):
            current += self.timeline[time]
            max_booking = max(max_booking, current)

        return max_booking
```

### Segment Tree Solution (Optimized)
```python
class MyCalendarThree:
    """
    Segment tree for O(log n) per query.
    """

    def __init__(self):
        self.tree = defaultdict(int)
        self.lazy = defaultdict(int)

    def update(self, start, end, left, right, idx):
        if start > right or end < left:
            return

        if start <= left and right <= end:
            self.tree[idx] += 1
            self.lazy[idx] += 1
            return

        mid = (left + right) // 2
        self.update(start, end, left, mid, 2 * idx)
        self.update(start, end, mid + 1, right, 2 * idx + 1)

        self.tree[idx] = self.lazy[idx] + max(
            self.tree[2 * idx],
            self.tree[2 * idx + 1]
        )

    def book(self, start: int, end: int) -> int:
        self.update(start, end - 1, 0, 10**9, 1)
        return self.tree[1]
```

### Complexity
- **Time**: O(n²) sweep line, O(log n) segment tree
- **Space**: O(n)

### Edge Cases
- First booking → return 1
- No overlap → return 1
- All overlap → return number of bookings
- Partial overlaps → track max concurrent

---

## Problem 5: Insert Interval (LC #57) - Medium/Hard

- [LeetCode](https://leetcode.com/problems/insert-interval/)

### Video Explanation
- [NeetCode - Insert Interval](https://www.youtube.com/watch?v=A8NUOmlwOlM)

### Problem Statement
Insert new interval into sorted non-overlapping intervals, merging if necessary.

### Visual Intuition
```
Insert Interval
intervals = [[1,3],[6,9]], newInterval = [2,5]

Pattern: Three-Phase Linear Scan
Why: Handle non-overlapping, overlapping, and remaining separately

Step 0 (Visualize on Timeline):

  Existing:   ███         ████
              1--3        6---9

  New:          ████
                2---5

  Timeline:  1  2  3  4  5  6  7  8  9

Step 1 (Phase 1 - Add Non-Overlapping Before):

  Add intervals where end < newInterval.start

  [1,3]: end=3, new.start=2 → 3 < 2? NO

  No intervals added in Phase 1
  result = []

Step 2 (Phase 2 - Merge Overlapping):

  While interval.start ≤ newInterval.end:

  [1,3]: start=1 ≤ 5? YES → overlap!
         Merge: new = [min(1,2), max(3,5)] = [1,5]

  [6,9]: start=6 ≤ 5? NO → stop merging

  Add merged interval: [1,5]
  result = [[1,5]]

Step 3 (Phase 3 - Add Remaining):

  Add all remaining intervals:
  [6,9] → result = [[1,5], [6,9]]

Final Result: [[1,5], [6,9]]

Visualization of Merge:

  Before:   ███         ████
            1--3        6---9
              ████
              2---5

  After:    █████       ████
            1----5      6---9

            [1,3] + [2,5] = [1,5]

Three-Phase Algorithm:
  ┌─────────────────────────────────────────────────┐
  │ Phase 1: intervals ending before new starts     │
  │   while intervals[i].end < new.start:           │
  │     result.append(intervals[i])                 │
  │                                                 │
  │ Phase 2: merge overlapping                      │
  │   while intervals[i].start <= new.end:          │
  │     new.start = min(new.start, intervals[i].start) │
  │     new.end = max(new.end, intervals[i].end)    │
  │   result.append(new)                            │
  │                                                 │
  │ Phase 3: remaining intervals                    │
  │   result.extend(remaining intervals)            │
  └─────────────────────────────────────────────────┘

Key Insight:
- O(n) single pass through intervals
- Merge expands new interval to cover all overlapping
- Works because intervals are sorted and non-overlapping
```


### Intuition
```
intervals = [[1,3],[6,9]], newInterval = [2,5]

Result: [[1,5],[6,9]]

Three phases:
1. Add intervals ending before newInterval starts
2. Merge overlapping intervals with newInterval
3. Add remaining intervals
```

### Solution
```python
def insert(intervals: list[list[int]], newInterval: list[int]) -> list[list[int]]:
    """
    Linear scan with three phases.

    Time: O(n)
    Space: O(n) for result
    """
    result = []
    i = 0
    n = len(intervals)

    # Phase 1: Add all intervals ending before newInterval starts
    while i < n and intervals[i][1] < newInterval[0]:
        result.append(intervals[i])
        i += 1

    # Phase 2: Merge overlapping intervals
    while i < n and intervals[i][0] <= newInterval[1]:
        newInterval[0] = min(newInterval[0], intervals[i][0])
        newInterval[1] = max(newInterval[1], intervals[i][1])
        i += 1

    result.append(newInterval)

    # Phase 3: Add remaining intervals
    while i < n:
        result.append(intervals[i])
        i += 1

    return result
```

### Binary Search Optimization
```python
import bisect

def insert(intervals: list[list[int]], newInterval: list[int]) -> list[list[int]]:
    """
    Binary search to find merge range.

    Time: O(n) for merge, O(log n) for search
    Space: O(n)
    """
    if not intervals:
        return [newInterval]

    # Find first interval that might overlap
    starts = [interval[0] for interval in intervals]
    ends = [interval[1] for interval in intervals]

    # Find merge range
    left = bisect.bisect_left(ends, newInterval[0])
    right = bisect.bisect_right(starts, newInterval[1])

    # Merge
    if left < right:
        newInterval[0] = min(newInterval[0], intervals[left][0])
        newInterval[1] = max(newInterval[1], intervals[right - 1][1])

    return intervals[:left] + [newInterval] + intervals[right:]
```

### Complexity
- **Time**: O(n)
- **Space**: O(n)

### Edge Cases
- Empty intervals → return [newInterval]
- No overlap → insert in order
- Complete overlap → merge all
- Insert at start/end → simple append

---

## Problem 6: Remove Covered Intervals (LC #1288) - Medium/Hard

- [LeetCode](https://leetcode.com/problems/remove-covered-intervals/)

### Video Explanation
- [NeetCode - Remove Covered Intervals](https://www.youtube.com/watch?v=nhvpYj7ex10)

### Problem Statement
Return number of remaining intervals after removing covered ones.

### Visual Intuition
```
Remove Covered Intervals
intervals = [[1,4],[3,6],[2,8]]

Pattern: Sort + Track Max End
Why: Interval is covered if end ≤ max_end seen so far

Step 0 (Understand "Covered"):

  Interval A covers interval B if:
    A.start ≤ B.start AND A.end ≥ B.end

  [2,8] covers [3,6] because:
    2 ≤ 3 ✓ (start)
    8 ≥ 6 ✓ (end)

  Visualization:
    2────────8
      3───6    ← completely inside [2,8]

Step 1 (Sort Strategy):

  Sort by: start ascending, end DESCENDING

  Why end descending?
  If same start, longer interval comes first
  → shorter one is automatically covered

  Original: [[1,4], [3,6], [2,8]]
  Sorted:   [[1,4], [2,8], [3,6]]
                      ↑
            Same start 2 and 3? No, but [2,8] before [3,6]

Step 2 (Scan with Max End Tracking):

  max_end = 0
  count = 0

  [1,4]: end=4 > max_end=0 → NOT covered
         count = 1, max_end = 4

  [2,8]: end=8 > max_end=4 → NOT covered
         count = 2, max_end = 8

  [3,6]: end=6 ≤ max_end=8 → COVERED!
         (skip, don't count)

Answer: 2 intervals remain

Timeline Visualization:

  1────4         [1,4] ✓ kept
    2────────8   [2,8] ✓ kept
      3───6      [3,6] ✗ covered by [2,8]

  ─────────────────────────────
  1  2  3  4  5  6  7  8

Why This Works:
  ┌─────────────────────────────────────────────────┐
  │ After sorting:                                  │
  │ - Earlier start = potential coverer            │
  │ - Same start, longer end = covers shorter      │
  │                                                 │
  │ If end ≤ max_end:                              │
  │   Some previous interval with earlier start    │
  │   and longer end covers this one               │
  │                                                 │
  │ If end > max_end:                              │
  │   This interval extends beyond all previous    │
  │   → cannot be covered                          │
  └─────────────────────────────────────────────────┘

Key Insight:
- Sort: start ASC, end DESC
- Covered iff end ≤ max_end seen
- O(n log n) sort, O(n) scan
```


### Intuition
```
intervals = [[1,4],[3,6],[2,8]]

Sort by start (ascending), then by end (descending):
[[1,4],[2,8],[3,6]]

[2,8] covers [3,6] since 2 <= 3 and 8 >= 6
Track max_end seen, count non-covered intervals.
```

### Solution
```python
def removeCoveredIntervals(intervals: list[list[int]]) -> int:
    """
    Sort and track maximum end.

    Strategy:
    - Sort by start ascending, end descending
    - Interval is covered if its end <= max_end seen
    - Count non-covered intervals

    Time: O(n log n)
    Space: O(1)
    """
    # Sort: start ascending, end descending
    # Same start → longer interval first (covers shorter ones)
    intervals.sort(key=lambda x: (x[0], -x[1]))

    count = 0
    max_end = 0

    for start, end in intervals:
        # If end > max_end, this interval is not covered
        if end > max_end:
            count += 1
            max_end = end
        # else: this interval is covered by a previous one

    return count
```

### Complexity
- **Time**: O(n log n)
- **Space**: O(1) excluding sort space

### Edge Cases
- No covered intervals → return n
- All covered by one → return 1
- Same start, different end → longer covers shorter
- Identical intervals → count as 1

---

## Summary

| # | Problem | Key Technique |
|---|---------|---------------|
| 1 | Employee Free Time | Merge + find gaps |
| 2 | Data Stream Intervals | Sorted list + merge |
| 3 | Min Interval for Query | Sort + min heap |
| 4 | My Calendar III | Sweep line / segment tree |
| 5 | Insert Interval | Three-phase linear scan |
| 6 | Remove Covered Intervals | Sort + max end tracking |
