"""Print structured probe results and concise errors without Ray session data."""

import json
import sys
from pathlib import Path

for line in Path(sys.argv[1]).read_text(errors="replace").splitlines():
    try:
        record = json.loads(line)
    except ValueError:
        if any(
            marker in line
            for marker in [
                "Exception occurred",
                "ImportError:",
                "NotImplementedError:",
                "AssertionError:",
                "RuntimeError:",
                "TypeError:",
                "ValueError:",
                "timeout:",
                " passed in ",
            ]
        ):
            print(line[:450])
    else:
        print(json.dumps(record, ensure_ascii=False))
