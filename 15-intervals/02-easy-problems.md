# Intervals - Practice Problems

## Problem 1: Merge Intervals (LC #56) - Medium

- [LeetCode](https://leetcode.com/problems/merge-intervals/)

### Problem Statement
Merge all overlapping intervals.

### Examples
```
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
```

### Video Explanation
- [NeetCode - Merge Intervals](https://www.youtube.com/watch?v=44H3cEC2fFM)

### Intuition
```
Sort by start time, then merge overlapping intervals!

Two intervals overlap if: current.start <= prev.end

Visual: [[1,3],[2,6],[8,10],[15,18]]

        After sorting (already sorted):

        [1,3]  [2,6]  [8,10]  [15,18]
        ├──┤
           ├────┤
                   ├───┤
                           ├────┤

        Merge [1,3] and [2,6] → [1,6] (2 ≤ 3, overlap!)
        [8,10] doesn't overlap with [1,6] (8 > 6)
        [15,18] doesn't overlap with [8,10] (15 > 10)

        Result: [[1,6],[8,10],[15,18]]
```

### Solution
```python
def merge(intervals: list[list[int]]) -> list[list[int]]:
    """
    Merge overlapping intervals.

    Strategy:
    1. Sort by start time
    2. Iterate and merge overlapping intervals

    Two intervals overlap if: current.start <= last.end

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

        # Check overlap: current starts before last ends
        if current[0] <= last[1]:
            # Merge: extend end of last interval
            last[1] = max(last[1], current[1])
        else:
            # No overlap: add as new interval
            merged.append(current)

    return merged
```

### Edge Cases
- Empty intervals → return []
- Single interval → return as is
- No overlapping intervals → return sorted
- All intervals overlap → return single merged
- Intervals already merged → return as is

---

## Problem 2: Insert Interval (LC #57) - Medium

- [LeetCode](https://leetcode.com/problems/insert-interval/)

### Problem Statement
Insert new interval into sorted, non-overlapping intervals.

### Examples
```
Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]
```

### Video Explanation
- [NeetCode - Insert Interval](https://www.youtube.com/watch?v=A8NUOmlwOlM)

### Intuition
```
Three phases: BEFORE, MERGE, AFTER

Phase 1: Add intervals that end BEFORE new interval starts
Phase 2: Merge all overlapping intervals with new interval
Phase 3: Add intervals that start AFTER new interval ends

Visual: intervals = [[1,3],[6,9]], newInterval = [2,5]

        [1,3]     [6,9]
        ├──┤      ├───┤
           [2,5]
           ├───┤

        Phase 1: [1,3] ends at 3, newInterval starts at 2
                 3 >= 2, so [1,3] might overlap → skip to phase 2

        Phase 2: Merge [1,3] and [2,5] → [1,5]
                 [6,9] starts at 6 > 5 (newInterval end) → stop merging

        Phase 3: Add [6,9]

        Result: [[1,5],[6,9]]
```

### Solution
```python
def insert(intervals: list[list[int]], newInterval: list[int]) -> list[list[int]]:
    """
    Insert interval into sorted list, merging if necessary.

    Strategy: Three phases
    1. Add intervals that end before new interval starts
    2. Merge overlapping intervals
    3. Add remaining intervals

    Time: O(n)
    Space: O(n)
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

### Edge Cases
- Empty intervals → return [newInterval]
- New interval before all → prepend
- New interval after all → append
- New interval covers all → return [newInterval]
- No overlap with any → insert at correct position

---

## Problem 3: Non-overlapping Intervals (LC #435) - Medium

- [LeetCode](https://leetcode.com/problems/non-overlapping-intervals/)

### Problem Statement
Minimum intervals to remove for non-overlapping.

### Examples
```
Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
Output: 1 (remove [1,3])
```

### Video Explanation
- [NeetCode - Non-overlapping Intervals](https://www.youtube.com/watch?v=nONCGxWoUfM)

### Intuition
```
Greedy: Sort by END time, keep intervals that end earliest!

Why end time? Intervals ending earlier leave more room for others.

Visual: [[1,2],[2,3],[3,4],[1,3]]

        Sort by end time: [[1,2],[2,3],[1,3],[3,4]]

        [1,2]  ├─┤
        [2,3]     ├─┤
        [1,3]  ├────┤  ← overlaps with [2,3], remove!
        [3,4]        ├─┤

        Keep: [1,2], [2,3], [3,4]
        Remove: [1,3] (1 removal)

Alternative view: This is equivalent to finding
MAXIMUM non-overlapping intervals (activity selection).
```

### Solution
```python
def eraseOverlapIntervals(intervals: list[list[int]]) -> int:
    """
    Minimum intervals to remove for non-overlapping.

    Greedy insight: Sort by END time, keep intervals that end earliest.
    This leaves most room for future intervals.

    Time: O(n log n)
    Space: O(1)
    """
    if not intervals:
        return 0

    # Sort by end time (key insight!)
    intervals.sort(key=lambda x: x[1])

    removals = 0
    prev_end = float('-inf')

    for start, end in intervals:
        if start >= prev_end:
            # No overlap - keep this interval
            prev_end = end
        else:
            # Overlap - remove this interval (keep previous)
            removals += 1

    return removals
```

### Edge Cases
- Empty intervals → return 0
- Single interval → return 0
- No overlaps → return 0
- All overlap with each other → remove n-1
- Adjacent intervals [1,2],[2,3] → no overlap (end = start)

---

## Problem 4: Meeting Rooms II (LC #253) - Medium

- [LeetCode](https://leetcode.com/problems/meeting-rooms-ii/)

### Problem Statement
Find minimum meeting rooms required.

### Examples
```
Input: intervals = [[0,30],[5,10],[15,20]]
Output: 2
```

### Video Explanation
- [NeetCode - Meeting Rooms II](https://www.youtube.com/watch?v=FdzJmTCVyJU)

### Intuition
```
Track concurrent meetings using a min-heap of end times!

Visual: [[0,30],[5,10],[15,20]]

        Time:  0   5   10   15   20   30
               ├────────────────────────┤ [0,30]
                   ├────┤                 [5,10]
                             ├────┤       [15,20]

        At time 5: 2 meetings concurrent ([0,30] and [5,10])
        At time 15: 2 meetings concurrent ([0,30] and [15,20])

        Max concurrent = 2 rooms needed

Algorithm:
1. Sort by start time
2. Use min-heap to track end times
3. For each meeting:
   - If earliest ending meeting is done, reuse room
   - Otherwise, add new room
```

### Solution
```python
import heapq

def minMeetingRooms(intervals: list[list[int]]) -> int:
    """
    Minimum meeting rooms using min-heap.

    Strategy:
    - Sort by start time
    - Heap tracks end times of ongoing meetings
    - For each meeting: if room freed (end <= start), reuse it
    - Otherwise, need new room

    Time: O(n log n)
    Space: O(n)
    """
    if not intervals:
        return 0

    # Sort by start time
    intervals.sort(key=lambda x: x[0])

    # Min-heap of end times (rooms in use)
    rooms = []
    heapq.heappush(rooms, intervals[0][1])

    for i in range(1, len(intervals)):
        start, end = intervals[i]

        # If earliest ending meeting has ended, reuse that room
        if rooms[0] <= start:
            heapq.heappop(rooms)

        # Add current meeting's end time
        heapq.heappush(rooms, end)

    return len(rooms)


def minMeetingRooms_sweep(intervals: list[list[int]]) -> int:
    """
    Alternative: Line sweep algorithm.

    Create events for starts (+1) and ends (-1).
    Track maximum concurrent meetings.

    Time: O(n log n)
    Space: O(n)
    """
    events = []

    for start, end in intervals:
        events.append((start, 1))   # Meeting starts
        events.append((end, -1))    # Meeting ends

    # Sort: by time, then ends before starts (to handle [1,2], [2,3])
    events.sort(key=lambda x: (x[0], x[1]))

    max_rooms = 0
    current_rooms = 0

    for time, delta in events:
        current_rooms += delta
        max_rooms = max(max_rooms, current_rooms)

    return max_rooms
```

### Edge Cases
- Empty intervals → return 0
- Single meeting → return 1
- No overlapping meetings → return 1
- All meetings overlap → return n
- Back-to-back meetings [0,1],[1,2] → can reuse room

---

## Problem 5: Interval List Intersections (LC #986) - Medium

- [LeetCode](https://leetcode.com/problems/interval-list-intersections/)

### Problem Statement
Find intersection of two interval lists.

### Examples
```
Input: firstList = [[0,2],[5,10],[13,23],[24,25]]
       secondList = [[1,5],[8,12],[15,24],[25,26]]
Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
```

### Video Explanation
- [NeetCode - Interval List Intersections](https://www.youtube.com/watch?v=Qh8ZjL1RpLI)

### Intuition
```
Two Pointers: Compare intervals from both lists!

Intersection formula:
- start = max(a_start, b_start)
- end = min(a_end, b_end)
- Valid if start <= end

Visual:
        List A: [0,2]     [5,10]      [13,23]    [24,25]
                ├──┤      ├─────┤     ├─────────┤├─┤

        List B:    [1,5]     [8,12]      [15,24]   [25,26]
                   ├───┤     ├───┤       ├───────┤  ├─┤

        Intersections:
        [0,2] ∩ [1,5] = [1,2]
        [5,10] ∩ [1,5] = [5,5]
        [5,10] ∩ [8,12] = [8,10]
        [13,23] ∩ [15,24] = [15,23]
        [24,25] ∩ [15,24] = [24,24]
        [24,25] ∩ [25,26] = [25,25]

Move pointer of interval that ends FIRST!
```

### Solution
```python
def intervalIntersection(firstList: list[list[int]], secondList: list[list[int]]) -> list[list[int]]:
    """
    Find intersections of two sorted interval lists.

    Strategy (Two Pointers):
    - Compare current intervals from each list
    - Intersection exists if max(starts) <= min(ends)
    - Move pointer of interval that ends first

    Time: O(m + n)
    Space: O(1) excluding output
    """
    result = []
    i = j = 0

    while i < len(firstList) and j < len(secondList):
        # Get current intervals
        a_start, a_end = firstList[i]
        b_start, b_end = secondList[j]

        # Calculate intersection
        start = max(a_start, b_start)
        end = min(a_end, b_end)

        # Valid intersection if start <= end
        if start <= end:
            result.append([start, end])

        # Move pointer of interval that ends first
        if a_end < b_end:
            i += 1
        else:
            j += 1

    return result
```

### Edge Cases
- One list empty → return []
- No overlaps → return []
- Complete overlap → return intersection
- Point intervals [1,1] → can still intersect
- One list is subset of other → return subset intersections

---

## Problem 6: Minimum Number of Arrows (LC #452) - Medium

- [LeetCode](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/)

### Problem Statement
Minimum arrows to burst all balloons (intervals on x-axis).

### Video Explanation
- [NeetCode - Minimum Number of Arrows to Burst Balloons](https://www.youtube.com/watch?v=lPmkKnvNPrw)

### Examples
```
Input: points = [[10,16],[2,8],[1,6],[7,12]]
Output: 2 (arrows at x=6 and x=11)
```

### Solution
```python
def findMinArrowShots(points: list[list[int]]) -> int:
    """
    Minimum arrows to burst all balloons.

    Greedy insight: Sort by END, shoot at end of first balloon.
    This maximizes balloons hit by single arrow.

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
            arrow_pos = end  # Shoot at end of this balloon

    return arrows
```

### Edge Cases
- Empty points → return 0
- Single balloon → return 1
- No overlapping balloons → return n
- All balloons at same position → return 1
- Balloons touching at point → one arrow can hit both

---

## Problem 7: Remove Covered Intervals (LC #1288) - Medium

- [LeetCode](https://leetcode.com/problems/remove-covered-intervals/)

### Problem Statement
Remove intervals that are covered by another interval.

### Video Explanation
- [NeetCode - Remove Covered Intervals](https://www.youtube.com/watch?v=nhvpYj7ex10)

### Examples
```
Input: intervals = [[1,4],[3,6],[2,8]]
Output: 2 (interval [3,6] is covered by [2,8])
```

### Solution
```python
def removeCoveredIntervals(intervals: list[list[int]]) -> int:
    """
    Count intervals not covered by any other.

    Interval [a,b] is covered by [c,d] if c <= a and b <= d.

    Strategy:
    - Sort by start ascending, then by end descending
    - Track maximum end seen
    - If current end <= max_end, it's covered

    Time: O(n log n)
    Space: O(1)
    """
    # Sort: start ascending, end descending
    # This ensures longer intervals come first for same start
    intervals.sort(key=lambda x: (x[0], -x[1]))

    count = 0
    max_end = 0

    for start, end in intervals:
        if end > max_end:
            # Not covered by any previous interval
            count += 1
            max_end = end
        # else: covered (end <= max_end)

    return count
```

### Edge Cases
- Empty intervals → return 0
- Single interval → return 1
- No interval covers another → return n
- All same intervals → return 1
- Nested intervals → only outermost counted

---

## Problem 8: My Calendar I (LC #729) - Medium

- [LeetCode](https://leetcode.com/problems/my-calendar-i/)

### Problem Statement
Implement a calendar that prevents double booking.

### Video Explanation
- [NeetCode - My Calendar I](https://www.youtube.com/watch?v=SbFVL-SMMCU)

### Solution
```python
class MyCalendar:
    """
    Calendar with no double booking.

    Strategy: Store booked intervals, check overlap for each new booking.

    Time: O(n) per booking (can be O(log n) with balanced BST)
    Space: O(n)
    """

    def __init__(self):
        self.bookings = []

    def book(self, start: int, end: int) -> bool:
        """
        Book interval [start, end) if no overlap.

        Two intervals [a, b) and [c, d) overlap if:
        max(a, c) < min(b, d)
        Or equivalently: NOT (b <= c OR d <= a)
        """
        for s, e in self.bookings:
            # Check overlap
            if max(start, s) < min(end, e):
                return False

        self.bookings.append((start, end))
        return True


class MyCalendarBST:
    """
    Optimized version using sorted list with binary search.

    Time: O(log n) per booking
    Space: O(n)
    """

    def __init__(self):
        from sortedcontainers import SortedList
        self.bookings = SortedList()

    def book(self, start: int, end: int) -> bool:
        # Find position where this booking would go
        idx = self.bookings.bisect_left((start, end))

        # Check overlap with previous interval
        if idx > 0 and self.bookings[idx - 1][1] > start:
            return False

        # Check overlap with next interval
        if idx < len(self.bookings) and self.bookings[idx][0] < end:
            return False

        self.bookings.add((start, end))
        return True
```

### Edge Cases
- First booking → always succeeds
- Same booking twice → second fails
- Adjacent bookings [0,10),[10,20) → both succeed
- Booking inside existing → fails
- Booking that contains existing → fails

---

## Summary: Interval Problems

| # | Problem | Key Technique | Time |
|---|---------|---------------|------|
| 1 | Merge Intervals | Sort by start, extend end | O(n log n) |
| 2 | Insert Interval | Three-phase scan | O(n) |
| 3 | Non-overlapping | Sort by end, greedy keep | O(n log n) |
| 4 | Meeting Rooms II | Heap or sweep line | O(n log n) |
| 5 | Intersections | Two pointers | O(m + n) |
| 6 | Burst Balloons | Sort by end, greedy shoot | O(n log n) |
| 7 | Remove Covered | Sort by start, -end | O(n log n) |
| 8 | Calendar | Check all overlaps | O(n) |

---

## Key Insights

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INTERVAL PROBLEM PATTERNS                                │
│                                                                             │
│  1. MERGE/UNION: Sort by START                                              │
│     - Merge if current.start <= last.end                                    │
│                                                                             │
│  2. NON-OVERLAPPING/GREEDY: Sort by END                                     │
│     - Keep intervals that end earliest                                      │
│     - Leaves most room for future intervals                                 │
│                                                                             │
│  3. INTERSECTION: max(starts) <= min(ends)                                  │
│     - If true, intersection = [max(starts), min(ends)]                      │
│                                                                             │
│  4. CONCURRENT COUNT: Sweep line                                            │
│     - +1 at start, -1 at end                                                │
│     - Sort events, track running count                                      │
│                                                                             │
│  5. COVERED: Sort by start ASC, end DESC                                    │
│     - Track max end seen                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Practice More Problems

- [ ] LC #252 - Meeting Rooms
- [ ] LC #352 - Data Stream as Disjoint Intervals
- [ ] LC #715 - Range Module
- [ ] LC #759 - Employee Free Time
- [ ] LC #1272 - Remove Interval

