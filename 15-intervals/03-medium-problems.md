# Intervals - Medium Problems

## Advanced Interval Techniques

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ADVANCED INTERVAL STRATEGIES                             │
│                                                                             │
│  1. LINE SWEEP:                                                             │
│     - Convert intervals to events (start/end)                               │
│     - Sort events by position                                               │
│     - Process events to track active intervals                              │
│                                                                             │
│  2. INTERVAL SCHEDULING:                                                    │
│     - Maximum non-overlapping: sort by end time                             │
│     - Minimum rooms/resources: sweep line or heap                           │
│                                                                             │
│  3. MERGE INTERVALS:                                                        │
│     - Sort by start time                                                    │
│     - Merge overlapping intervals                                           │
│                                                                             │
│  4. INTERVAL INTERSECTION:                                                  │
│     - Two pointers on sorted intervals                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Problem 1: Meeting Rooms II (LC #253) - Medium

- [LeetCode](https://leetcode.com/problems/meeting-rooms-ii/)

### Problem Statement
Given an array of meeting time intervals `intervals` where `intervals[i] = [starti, endi]`, return the minimum number of conference rooms required to host all meetings.

### Video Explanation
- [NeetCode - Meeting Rooms II](https://www.youtube.com/watch?v=FdzJmTCVyJU)

### Examples
```
Input: intervals = [[0,30],[5,10],[15,20]]
Output: 2
Explanation: Meeting 1: 0-30, Meeting 2: 5-10, Meeting 3: 15-20
  At time 5: meetings 1 and 2 overlap → need 2 rooms
  At time 15: meetings 1 and 3 overlap → need 2 rooms

Input: intervals = [[7,10],[2,4]]
Output: 1
Explanation: No overlap

Input: intervals = [[0,10],[10,20]]
Output: 1
Explanation: Meeting ends at 10, next starts at 10 → can reuse room
```

### Intuition Development
```
Two classic approaches: Heap or Line Sweep!

intervals = [[0,30], [5,10], [15,20]]

┌─────────────────────────────────────────────────────────────────┐
│ APPROACH 1: Min-Heap of end times                               │
│                                                                  │
│ Sort by start time, use heap to track room end times.           │
│                                                                  │
│ [0,30]: heap = [30], rooms = 1                                  │
│ [5,10]: 5 < 30, can't reuse → heap = [10,30], rooms = 2        │
│ [15,20]: 15 ≥ 10, reuse! pop 10, push 20                       │
│          heap = [20,30], rooms = 2                              │
│                                                                  │
│ Answer: max heap size = 2                                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ APPROACH 2: Line Sweep                                          │
│                                                                  │
│ Events: (0,+1), (5,+1), (10,-1), (15,+1), (20,-1), (30,-1)     │
│                                                                  │
│ Time 0:  +1 → count = 1                                         │
│ Time 5:  +1 → count = 2 ★ max                                   │
│ Time 10: -1 → count = 1                                         │
│ Time 15: +1 → count = 2 ★ max                                   │
│ Time 20: -1 → count = 1                                         │
│ Time 30: -1 → count = 0                                         │
│                                                                  │
│ Answer: max count = 2                                           │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
import heapq

def minMeetingRooms(intervals: list[list[int]]) -> int:
    """
    Minimum meeting rooms using min-heap.

    Strategy:
    - Sort by start time
    - Use heap to track end times of ongoing meetings
    - If new meeting starts after earliest end, reuse that room

    Time: O(n log n)
    Space: O(n)
    """
    if not intervals:
        return 0

    # Sort by start time
    intervals.sort(key=lambda x: x[0])

    # Min-heap of end times
    rooms = []
    heapq.heappush(rooms, intervals[0][1])

    for i in range(1, len(intervals)):
        start, end = intervals[i]

        # If earliest ending room is free, reuse it
        if rooms[0] <= start:
            heapq.heappop(rooms)

        # Add this meeting's end time
        heapq.heappush(rooms, end)

    return len(rooms)


def minMeetingRooms_sweep(intervals: list[list[int]]) -> int:
    """
    Alternative: Line sweep algorithm.

    Time: O(n log n)
    Space: O(n)
    """
    events = []

    for start, end in intervals:
        events.append((start, 1))   # Meeting starts
        events.append((end, -1))    # Meeting ends

    # Sort by time, ends before starts at same time
    events.sort(key=lambda x: (x[0], x[1]))

    max_rooms = 0
    current_rooms = 0

    for time, delta in events:
        current_rooms += delta
        max_rooms = max(max_rooms, current_rooms)

    return max_rooms
```

### Complexity
- **Time**: O(n log n) - Sorting dominates
- **Space**: O(n) - Heap or events list

### Edge Cases
- Empty intervals: Return 0
- No overlaps: Return 1
- All same time: Return n (all need separate rooms)
- Adjacent intervals: [0,10], [10,20] share room

---

## Problem 2: Employee Free Time (LC #759) - Hard

- [LeetCode](https://leetcode.com/problems/employee-free-time/)

### Problem Statement
You are given a list `schedule` of employees' working times, where each employee's schedule is a list of non-overlapping intervals. Return the list of finite intervals representing common free time for all employees, sorted in order.

### Video Explanation
- [NeetCode - Employee Free Time](https://www.youtube.com/watch?v=PSRQqIlqfPA)

### Examples
```
Input: schedule = [[[1,2],[5,6]],[[1,3]],[[4,10]]]
Output: [[3,4]]
Explanation:
  Employee 0: works [1,2], [5,6]
  Employee 1: works [1,3]
  Employee 2: works [4,10]
  Common free time: [3,4] (between 3 and 4, no one works)

Input: schedule = [[[1,3],[6,7]],[[2,4]],[[2,5],[9,12]]]
Output: [[5,6],[7,9]]
```

### Intuition Development
```
Flatten all intervals, merge, find gaps!

schedule = [[[1,2],[5,6]], [[1,3]], [[4,10]]]

┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Flatten all intervals                                   │
│   [1,2], [5,6], [1,3], [4,10]                                   │
│                                                                  │
│ Step 2: Sort by start time                                      │
│   [1,2], [1,3], [4,10], [5,6]                                   │
│                                                                  │
│ Step 3: Find gaps while merging                                 │
│   [1,2]: prev_end = 2                                           │
│   [1,3]: 1 ≤ 2 (overlap), prev_end = max(2,3) = 3               │
│   [4,10]: 4 > 3 (gap!) → add [3,4], prev_end = 10               │
│   [5,6]: 5 ≤ 10 (overlap), prev_end = max(10,6) = 10            │
│                                                                  │
│ Gaps found: [[3,4]]                                             │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def employeeFreeTime(schedule: list[list[list[int]]]) -> list[list[int]]:
    """
    Find common free time using interval merge.

    Strategy:
    - Flatten all intervals
    - Sort by start time
    - Find gaps between merged intervals

    Time: O(n log n)
    Space: O(n)
    """
    # Flatten all intervals
    all_intervals = []
    for employee in schedule:
        for interval in employee:
            all_intervals.append(interval)

    # Sort by start time
    all_intervals.sort(key=lambda x: x[0])

    # Merge and find gaps
    result = []
    prev_end = all_intervals[0][1]

    for start, end in all_intervals[1:]:
        if start > prev_end:
            # Gap found
            result.append([prev_end, start])

        prev_end = max(prev_end, end)

    return result
```

### Complexity
- **Time**: O(n log n) - Sorting all intervals
- **Space**: O(n) - Flattened intervals list

### Edge Cases
- Single employee: No common free time with others
- No gaps: Everyone always working → return []
- All non-overlapping: Free time between each interval

---

## Problem 3: Minimum Number of Arrows (LC #452) - Medium

- [LeetCode](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/)

### Problem Statement
There are some spherical balloons on a flat wall, given as intervals `[xstart, xend]`. Arrows can be shot vertically from different points along the x-axis. A balloon is burst if an arrow is shot between `xstart` and `xend` (inclusive). Return the minimum number of arrows needed to burst all balloons.

### Video Explanation
- [NeetCode - Minimum Number of Arrows](https://www.youtube.com/watch?v=lPmkKnvNPrw)

### Examples
```
Input: points = [[10,16],[2,8],[1,6],[7,12]]
Output: 2
Explanation:
  Arrow 1 at x=6: bursts [2,8] and [1,6]
  Arrow 2 at x=11: bursts [10,16] and [7,12]

Input: points = [[1,2],[3,4],[5,6],[7,8]]
Output: 4
Explanation: No overlaps, need 4 arrows

Input: points = [[1,2],[2,3],[3,4],[4,5]]
Output: 2
Explanation: Arrow at 2 bursts [1,2],[2,3]; arrow at 4 bursts [3,4],[4,5]
```

### Intuition Development
```
Interval scheduling: Sort by END, greedily shoot!

points = [[10,16], [2,8], [1,6], [7,12]]

┌─────────────────────────────────────────────────────────────────┐
│ Sort by end coordinate:                                         │
│   [1,6], [2,8], [7,12], [10,16]                                 │
│                                                                  │
│ Greedy: Shoot at each balloon's END (maximizes coverage)        │
│                                                                  │
│ Arrow 1 at x=6:                                                  │
│   [1,6]: burst ✓ (1 ≤ 6 ≤ 6)                                   │
│   [2,8]: burst ✓ (2 ≤ 6 ≤ 8)                                   │
│   [7,12]: not burst (7 > 6)                                     │
│                                                                  │
│ Arrow 2 at x=12:                                                 │
│   [7,12]: burst ✓                                               │
│   [10,16]: burst ✓                                              │
│                                                                  │
│ Total: 2 arrows                                                  │
│                                                                  │
│ Key insight: Sorting by END ensures we burst as many overlapping │
│ balloons as possible with each arrow!                           │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def findMinArrowShots(points: list[list[int]]) -> int:
    """
    Minimum arrows to burst all balloons.

    This is interval scheduling - find maximum non-overlapping.

    Strategy:
    - Sort by end coordinate
    - Greedily shoot at each balloon's end
    - Skip balloons already burst

    Time: O(n log n)
    Space: O(1)
    """
    if not points:
        return 0

    # Sort by end coordinate
    points.sort(key=lambda x: x[1])

    arrows = 1
    arrow_pos = points[0][1]  # Shoot at first balloon's end

    for start, end in points[1:]:
        if start > arrow_pos:
            # Need new arrow
            arrows += 1
            arrow_pos = end

    return arrows
```

### Complexity
- **Time**: O(n log n) - Sorting
- **Space**: O(1) - In-place or O(n) for sort

### Edge Cases
- Empty array: Return 0
- Single balloon: Return 1
- All overlapping: Return 1
- No overlapping: Return n

---

## Problem 4: Non-overlapping Intervals (LC #435) - Medium

- [LeetCode](https://leetcode.com/problems/non-overlapping-intervals/)

### Problem Statement
Given an array of intervals `intervals` where `intervals[i] = [starti, endi]`, return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

### Video Explanation
- [NeetCode - Non-overlapping Intervals](https://www.youtube.com/watch?v=nONCGxWoUfM)

### Examples
```
Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
Output: 1
Explanation: Remove [1,3], then [[1,2],[2,3],[3,4]] don't overlap

Input: intervals = [[1,2],[1,2],[1,2]]
Output: 2
Explanation: Keep one [1,2], remove two

Input: intervals = [[1,2],[2,3]]
Output: 0
Explanation: Already non-overlapping
```

### Intuition Development
```
Equivalent to: n - (max non-overlapping intervals)

intervals = [[1,2], [2,3], [3,4], [1,3]]

┌─────────────────────────────────────────────────────────────────┐
│ Sort by END time (greedy for max non-overlapping):              │
│   [1,2], [2,3], [1,3], [3,4]                                    │
│                                                                  │
│ Greedily select non-overlapping:                                │
│   [1,2]: keep, prev_end = 2                                     │
│   [2,3]: 2 ≥ 2, no overlap, keep, prev_end = 3                  │
│   [1,3]: 1 < 3, overlaps! skip                                  │
│   [3,4]: 3 ≥ 3, no overlap, keep, prev_end = 4                  │
│                                                                  │
│ Kept: 3 intervals                                               │
│ Remove: 4 - 3 = 1 interval                                      │
│                                                                  │
│ Why sort by end?                                                 │
│   Choosing interval that ends earliest leaves more room!        │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def eraseOverlapIntervals(intervals: list[list[int]]) -> int:
    """
    Minimum removals for non-overlapping intervals.

    Equivalent to: n - maximum non-overlapping intervals.

    Strategy:
    - Sort by end time
    - Greedily keep intervals that don't overlap

    Time: O(n log n)
    Space: O(1)
    """
    if not intervals:
        return 0

    # Sort by end time
    intervals.sort(key=lambda x: x[1])

    keep = 1
    prev_end = intervals[0][1]

    for start, end in intervals[1:]:
        if start >= prev_end:
            # No overlap, keep this interval
            keep += 1
            prev_end = end

    return len(intervals) - keep
```

### Complexity
- **Time**: O(n log n) - Sorting
- **Space**: O(1) - In-place

### Edge Cases
- Empty: Return 0
- All identical: Remove n-1
- Already non-overlapping: Return 0
- Touching intervals [1,2],[2,3]: Not overlapping

---

## Problem 5: My Calendar I (LC #729) - Medium

- [LeetCode](https://leetcode.com/problems/my-calendar-i/)

### Problem Statement
Implement a `MyCalendar` class to store events. A new event can be added if it does not cause a **double booking**. Double booking occurs when two events have non-empty intersection. The event `[start, end)` is half-open (includes start, excludes end).

### Video Explanation
- [NeetCode - My Calendar I](https://www.youtube.com/watch?v=dUGPzCVMF7Y)

### Examples
```
MyCalendar calendar = new MyCalendar();
calendar.book(10, 20); // returns true
calendar.book(15, 25); // returns false (overlaps [10,20))
calendar.book(20, 30); // returns true ([20,30) doesn't overlap [10,20))
```

### Intuition Development
```
Maintain sorted list, binary search for conflicts!

Bookings: []

┌─────────────────────────────────────────────────────────────────┐
│ book(10, 20):                                                    │
│   No bookings yet → add (10, 20)                                │
│   Bookings: [(10, 20)]                                          │
│                                                                  │
│ book(15, 25):                                                    │
│   Binary search finds position 1                                │
│   Check previous (10, 20): 20 > 15? Yes! Overlap                │
│   Return false                                                   │
│                                                                  │
│ book(20, 30):                                                    │
│   Binary search finds position 1                                │
│   Check previous (10, 20): 20 > 20? No!                         │
│   Check next: none                                               │
│   Insert at position 1                                          │
│   Bookings: [(10, 20), (20, 30)]                                │
│   Return true                                                    │
│                                                                  │
│ Two intervals [a,b) and [c,d) overlap if: a < d AND c < b       │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
class MyCalendar:
    """
    Calendar with no double booking.

    Strategy:
    - Store booked intervals in sorted order
    - Binary search for insertion point
    - Check overlap with neighbors

    Time: O(log n) search, O(n) insert
    Space: O(n)
    """

    def __init__(self):
        self.bookings = []  # Sorted list of (start, end)

    def book(self, start: int, end: int) -> bool:
        """Book if no overlap, return success."""
        import bisect

        # Find insertion point
        i = bisect.bisect_left(self.bookings, (start, end))

        # Check overlap with previous interval
        if i > 0 and self.bookings[i - 1][1] > start:
            return False

        # Check overlap with next interval
        if i < len(self.bookings) and self.bookings[i][0] < end:
            return False

        # No overlap, insert
        self.bookings.insert(i, (start, end))
        return True


class MyCalendar_TreeMap:
    """
    Alternative: Using sorted dictionary (TreeMap).
    """
    from sortedcontainers import SortedDict

    def __init__(self):
        self.bookings = SortedDict()

    def book(self, start: int, end: int) -> bool:
        # Find previous and next bookings
        idx = self.bookings.bisect_right(start)

        # Check overlap with previous
        if idx > 0:
            prev_start = self.bookings.keys()[idx - 1]
            if self.bookings[prev_start] > start:
                return False

        # Check overlap with next
        if idx < len(self.bookings):
            next_start = self.bookings.keys()[idx]
            if end > next_start:
                return False

        self.bookings[start] = end
        return True
```

### Complexity
- **Time**: O(log n) search, O(n) insert
- **Space**: O(n) - Stored bookings

### Edge Cases
- First booking: Always succeeds
- Adjacent events: [10,20), [20,30) don't overlap
- Same start/end: [10,10) is empty, always succeeds

---

## Problem 6: My Calendar II (LC #731) - Medium

- [LeetCode](https://leetcode.com/problems/my-calendar-ii/)

### Problem Statement
Implement a `MyCalendarTwo` class to store events. A new event can be added if it does not cause a **triple booking**. Triple booking occurs when three events have a common non-empty intersection.

### Video Explanation
- [NeetCode - My Calendar II](https://www.youtube.com/watch?v=fWUTTp3KRRY)

### Examples
```
MyCalendarTwo calendar = new MyCalendarTwo();
calendar.book(10, 20); // returns true
calendar.book(50, 60); // returns true
calendar.book(10, 40); // returns true (double booking [10,20) ok)
calendar.book(5, 15);  // returns false (would triple book [10,15))
calendar.book(5, 10);  // returns true
calendar.book(25, 55); // returns true (double booking [25,40) ok)
```

### Intuition Development
```
Track single and double bookings separately!

┌─────────────────────────────────────────────────────────────────┐
│ Two lists:                                                       │
│   single: all bookings (may overlap)                            │
│   double: intervals where two bookings overlap                  │
│                                                                  │
│ New booking:                                                     │
│   1. Check if overlaps with any in 'double' → reject (triple)  │
│   2. Find overlaps with 'single' → add overlaps to 'double'    │
│   3. Add to 'single'                                            │
│                                                                  │
│ Example: book(10,20), book(50,60), book(10,40)                  │
│   After book(10,20): single=[(10,20)]                           │
│   After book(50,60): single=[(10,20),(50,60)]                   │
│   After book(10,40):                                            │
│     Overlaps with (10,20) in single                             │
│     Add (10,20) to double                                       │
│     single=[(10,20),(50,60),(10,40)]                            │
│     double=[(10,20)]                                            │
│                                                                  │
│ book(5,15) would overlap [10,20) in double → REJECT             │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
class MyCalendarTwo:
    """
    Calendar allowing double but not triple booking.

    Strategy:
    - Track single bookings and double bookings separately
    - New booking overlapping with double booking → triple → reject

    Time: O(n) per booking
    Space: O(n)
    """

    def __init__(self):
        self.single = []   # Single bookings
        self.double = []   # Double bookings (overlaps)

    def book(self, start: int, end: int) -> bool:
        """Book if won't cause triple booking."""
        # Check if overlaps with any double booking
        for ds, de in self.double:
            if start < de and end > ds:
                return False  # Would cause triple booking

        # Add overlaps with single bookings to double
        for ss, se in self.single:
            if start < se and end > ss:
                # Overlap found, add to double
                self.double.append((max(start, ss), min(end, se)))

        # Add to single bookings
        self.single.append((start, end))
        return True
```

### Complexity
- **Time**: O(n) per booking - Check all single/double
- **Space**: O(n) - Both lists

### Edge Cases
- No overlaps: All succeed
- Full overlap of two: Third fails
- Partial triple: Only the overlapping portion causes failure

---

## Problem 7: My Calendar III (LC #732) - Hard

- [LeetCode](https://leetcode.com/problems/my-calendar-iii/)

### Problem Statement
A `k`-booking happens when `k` events have a non-empty common intersection. You are given some events where each event has a `[start, end)` time range. Return the maximum k-booking between all previous events after each new event is added.

### Video Explanation
- [NeetCode - My Calendar III](https://www.youtube.com/watch?v=4MEVRtCNwuQ)

### Examples
```
MyCalendarThree calendar = new MyCalendarThree();
calendar.book(10, 20); // returns 1
calendar.book(50, 60); // returns 1
calendar.book(10, 40); // returns 2 (double booking at [10,20))
calendar.book(5, 15);  // returns 3 (triple booking at [10,15))
calendar.book(5, 10);  // returns 3
calendar.book(25, 55); // returns 3
```

### Intuition Development
```
Line sweep: Track events (start/end) and sweep!

┌─────────────────────────────────────────────────────────────────┐
│ Use a timeline dictionary:                                      │
│   timeline[time] = delta (+1 for start, -1 for end)            │
│                                                                  │
│ After book(10,20), book(10,40), book(5,15):                     │
│   timeline = {5: +1, 10: +2, 15: -1, 20: -1, 40: -1}            │
│                                                                  │
│ Sweep through sorted times:                                     │
│   time 5:  count = 0 + 1 = 1                                    │
│   time 10: count = 1 + 2 = 3 ★ max                              │
│   time 15: count = 3 - 1 = 2                                    │
│   time 20: count = 2 - 1 = 1                                    │
│   time 40: count = 1 - 1 = 0                                    │
│                                                                  │
│ Maximum k-booking = 3                                           │
│                                                                  │
│ Note: We re-sweep entire timeline after each booking            │
│       Can optimize with segment tree for O(log n) queries       │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
from collections import defaultdict

class MyCalendarThree:
    """
    Track maximum concurrent bookings.

    Strategy: Line sweep with lazy updates.

    Time: O(n) per query
    Space: O(n)
    """

    def __init__(self):
        self.timeline = defaultdict(int)

    def book(self, start: int, end: int) -> int:
        """Add booking and return max concurrent."""
        self.timeline[start] += 1   # Booking starts
        self.timeline[end] -= 1     # Booking ends

        # Sweep through timeline
        max_concurrent = 0
        current = 0

        for time in sorted(self.timeline.keys()):
            current += self.timeline[time]
            max_concurrent = max(max_concurrent, current)

        return max_concurrent
```

### Complexity
- **Time**: O(n log n) per query - Sorting timeline
- **Space**: O(n) - Timeline storage

### Edge Cases
- First booking: Returns 1
- No overlaps: Always returns 1
- All same interval: Returns n

---

## Problem 8: Interval List Intersections (LC #986) - Medium

- [LeetCode](https://leetcode.com/problems/interval-list-intersections/)

### Problem Statement
You are given two lists of **closed** intervals, `firstList` and `secondList`, where each list is pairwise disjoint and in sorted order. Return the intersection of these two interval lists.

### Video Explanation
- [NeetCode - Interval List Intersections](https://www.youtube.com/watch?v=Qh8ZjL1RpLI)

### Examples
```
Input: firstList = [[0,2],[5,10],[13,23],[24,25]]
       secondList = [[1,5],[8,12],[15,24],[25,26]]
Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
Explanation:
  [0,2] ∩ [1,5] = [1,2]
  [5,10] ∩ [1,5] = [5,5]
  [5,10] ∩ [8,12] = [8,10]
  [13,23] ∩ [15,24] = [15,23]
  [24,25] ∩ [15,24] = [24,24]
  [24,25] ∩ [25,26] = [25,25]
```

### Intuition Development
```
Two pointers: One for each list!

firstList:  [0,2]  [5,10]  [13,23]  [24,25]
secondList: [1,5]  [8,12]  [15,24]  [25,26]

┌─────────────────────────────────────────────────────────────────┐
│ For two intervals [a_start, a_end] and [b_start, b_end]:       │
│                                                                  │
│   Intersection exists if: max(a_start, b_start) ≤ min(a_end, b_end)│
│   Intersection = [max(starts), min(ends)]                      │
│                                                                  │
│ Two pointer algorithm:                                          │
│   i=0, j=0:                                                      │
│     [0,2] ∩ [1,5]: start=max(0,1)=1, end=min(2,5)=2            │
│     Add [1,2]                                                    │
│     a_end=2 < b_end=5, advance i                                │
│                                                                  │
│   i=1, j=0:                                                      │
│     [5,10] ∩ [1,5]: start=max(5,1)=5, end=min(10,5)=5          │
│     Add [5,5]                                                    │
│     b_end=5 < a_end=10, advance j                               │
│                                                                  │
│   Continue until one list exhausted...                          │
│                                                                  │
│ Advance pointer with SMALLER end (it can't intersect more)      │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def intervalIntersection(firstList: list[list[int]],
                         secondList: list[list[int]]) -> list[list[int]]:
    """
    Find intersections of two interval lists.

    Strategy:
    - Two pointers, one for each list
    - Find intersection if exists
    - Advance pointer with smaller end

    Time: O(m + n)
    Space: O(1) excluding output
    """
    result = []
    i, j = 0, 0

    while i < len(firstList) and j < len(secondList):
        a_start, a_end = firstList[i]
        b_start, b_end = secondList[j]

        # Find intersection
        start = max(a_start, b_start)
        end = min(a_end, b_end)

        if start <= end:
            result.append([start, end])

        # Advance pointer with smaller end
        if a_end < b_end:
            i += 1
        else:
            j += 1

    return result
```

### Complexity
- **Time**: O(m + n) - Single pass through both lists
- **Space**: O(1) excluding output

### Edge Cases
- Empty list: Return []
- No intersection: Return []
- One list inside other: Return one list's intervals

---

## Problem 9: Remove Covered Intervals (LC #1288) - Medium

- [LeetCode](https://leetcode.com/problems/remove-covered-intervals/)

### Problem Statement
Given a list of intervals, remove all intervals that are covered by another interval in the list. Interval `[a,b)` is covered by interval `[c,d)` if `c <= a` and `b <= d`. Return the number of remaining intervals.

### Video Explanation
- [NeetCode - Remove Covered Intervals](https://www.youtube.com/watch?v=nhvyRwRl0X8)

### Examples
```
Input: intervals = [[1,4],[3,6],[2,8]]
Output: 2
Explanation: [1,4] is covered by [2,8]? No (2 > 1)
            [3,6] is covered by [2,8]? Yes (2 ≤ 3 and 6 ≤ 8)
            Remove [3,6], keep [1,4] and [2,8]

Input: intervals = [[1,4],[2,3]]
Output: 1
Explanation: [2,3] is covered by [1,4]

Input: intervals = [[0,10],[5,12]]
Output: 2
Explanation: Neither covers the other
```

### Intuition Development
```
Sort by start (asc), then by end (desc)!

intervals = [[1,4], [3,6], [2,8]]

┌─────────────────────────────────────────────────────────────────┐
│ Sort by (start asc, end desc):                                  │
│   [1,4], [2,8], [3,6]                                           │
│                                                                  │
│ Why end descending?                                             │
│   If starts are equal, longer interval comes first              │
│   Shorter one with same start is covered by longer!             │
│                                                                  │
│ Track max_end seen:                                             │
│   [1,4]: 4 > 0 (max_end), count++, max_end = 4                  │
│   [2,8]: 8 > 4 (max_end), count++, max_end = 8                  │
│   [3,6]: 6 ≤ 8 (max_end), COVERED! skip                         │
│                                                                  │
│ Count = 2                                                        │
│                                                                  │
│ Key insight: After sorting, if current end ≤ max_end,           │
│ it's covered by a previous interval!                            │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def removeCoveredIntervals(intervals: list[list[int]]) -> int:
    """
    Count intervals not covered by another.

    Strategy:
    - Sort by start (ascending), then by end (descending)
    - Track maximum end seen
    - If current end <= max_end, it's covered

    Time: O(n log n)
    Space: O(1)
    """
    # Sort by start asc, end desc
    intervals.sort(key=lambda x: (x[0], -x[1]))

    count = 0
    max_end = 0

    for start, end in intervals:
        if end > max_end:
            count += 1
            max_end = end
        # else: covered by previous interval

    return count
```

### Complexity
- **Time**: O(n log n) - Sorting
- **Space**: O(1) - In-place or O(n) for sort

### Edge Cases
- Single interval: Return 1
- All covered by one: Return 1
- No covered intervals: Return n
- Same start different end: Longer covers shorter

---

## Problem 10: Data Stream as Disjoint Intervals (LC #352) - Hard

- [LeetCode](https://leetcode.com/problems/data-stream-as-disjoint-intervals/)

### Problem Statement
Given a data stream of non-negative integers, implement a class that summarizes the numbers seen so far as a list of disjoint intervals. Implement `addNum(val)` to add a number and `getIntervals()` to return the current intervals.

### Video Explanation
- [NeetCode - Data Stream as Disjoint Intervals](https://www.youtube.com/watch?v=FLlj9Jg6xYs)

### Examples
```
SummaryRanges sr = new SummaryRanges();
sr.addNum(1);    // intervals: [[1,1]]
sr.addNum(3);    // intervals: [[1,1],[3,3]]
sr.addNum(7);    // intervals: [[1,1],[3,3],[7,7]]
sr.addNum(2);    // intervals: [[1,3],[7,7]] (merge!)
sr.addNum(6);    // intervals: [[1,3],[6,7]]
```

### Intuition Development
```
Maintain sorted intervals, merge on insert!

┌─────────────────────────────────────────────────────────────────┐
│ addNum(val): Find position, check for merges                   │
│                                                                  │
│ Intervals: [[1,1], [3,3]]                                       │
│ addNum(2):                                                      │
│   Binary search: 2 would go between [1,1] and [3,3]            │
│                                                                  │
│   Check merge with previous [1,1]: 1+1=2 = val → merge!        │
│   Check merge with next [3,3]: val+1=3 = 3 → merge!            │
│                                                                  │
│   Result: [[1,3]]                                               │
│                                                                  │
│ addNum(6):                                                      │
│   Intervals: [[1,3], [7,7]]                                     │
│   Binary search: 6 goes between                                 │
│   Previous [1,3]: 3+1=4 ≠ 6, no merge                          │
│   Next [7,7]: 6+1=7 = 7, merge! Extend 7 → [6,7]               │
│                                                                  │
│   Result: [[1,3], [6,7]]                                        │
│                                                                  │
│ Cases:                                                           │
│   1. Merge both: prev_end+1 == val == next_start-1              │
│   2. Merge prev: prev_end+1 == val                              │
│   3. Merge next: val+1 == next_start                            │
│   4. New interval: no adjacency                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
class SummaryRanges:
    """
    Track numbers as disjoint intervals.

    Strategy:
    - Use sorted set for quick neighbor lookup
    - Merge with adjacent intervals when adding

    Time: O(log n) add, O(n) getIntervals
    Space: O(n)
    """

    def __init__(self):
        self.intervals = []  # Sorted list of [start, end]

    def addNum(self, val: int) -> None:
        """Add number, merging with adjacent intervals."""
        import bisect

        # Binary search for position
        i = bisect.bisect_left(self.intervals, [val, val])

        # Check if val is already covered
        if i > 0 and self.intervals[i - 1][1] >= val:
            return
        if i < len(self.intervals) and self.intervals[i][0] <= val:
            return

        # Check merge with previous
        merge_prev = i > 0 and self.intervals[i - 1][1] == val - 1

        # Check merge with next
        merge_next = i < len(self.intervals) and self.intervals[i][0] == val + 1

        if merge_prev and merge_next:
            # Merge both
            self.intervals[i - 1][1] = self.intervals[i][1]
            self.intervals.pop(i)
        elif merge_prev:
            # Extend previous
            self.intervals[i - 1][1] = val
        elif merge_next:
            # Extend next
            self.intervals[i][0] = val
        else:
            # New interval
            self.intervals.insert(i, [val, val])

    def getIntervals(self) -> list[list[int]]:
        """Return current intervals."""
        return self.intervals
```

### Complexity
- **Time**: O(log n) add (binary search), O(n) getIntervals
- **Space**: O(n) - Stored intervals

### Edge Cases
- Duplicate value: Already covered, skip
- Values in order: Growing interval
- Disjoint values: Each becomes its own interval
- Merge chain: Adjacent merges can cascade

---

## Summary: Medium Interval Problems

| # | Problem | Key Technique | Time |
|---|---------|---------------|------|
| 1 | Meeting Rooms II | Heap or sweep line | O(n log n) |
| 2 | Employee Free Time | Merge + find gaps | O(n log n) |
| 3 | Min Arrows | Sort by end, greedy | O(n log n) |
| 4 | Non-overlapping | Max non-overlapping | O(n log n) |
| 5 | Calendar I | Binary search | O(n) |
| 6 | Calendar II | Track single/double | O(n) |
| 7 | Calendar III | Line sweep | O(n) |
| 8 | Interval Intersection | Two pointers | O(m + n) |
| 9 | Remove Covered | Sort + max end | O(n log n) |
| 10 | Disjoint Intervals | Sorted + merge | O(log n) |

---

## Practice More Problems

- [ ] LC #57 - Insert Interval
- [ ] LC #218 - The Skyline Problem
- [ ] LC #715 - Range Module
- [ ] LC #850 - Rectangle Area II
- [ ] LC #1235 - Maximum Profit in Job Scheduling

