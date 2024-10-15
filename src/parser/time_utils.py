import re
from typing import Optional

def time_to_seconds(time_str: str) -> int:
    """Convert time string to seconds."""
    parts = time_str.split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    else:
        return int(parts[0])

def get_clock_time(comment: str) -> Optional[int]:
    """Extract clock time from comment."""
    clock_match = re.search(r'\[%clk ([\d:]+)\]', comment)
    if clock_match:
        clock_time = clock_match.group(1)
        return time_to_seconds(clock_time)
    return None
