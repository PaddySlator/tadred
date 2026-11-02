import subprocess
from pathlib import Path

NOTICE = (
    "# Modified by Paddy Slator, 2026.\n"
    "# Original source: https://github.com/sbb-gh/tadred\n"
)

result = subprocess.run(
    [
        "git", "diff", "--name-only",
        "--diff-filter=M",
        "upstream/main...HEAD",
        "--", "*.py",
    ],
    capture_output=True,
    text=True,
    check=True,
)

for filename in result.stdout.splitlines():
    path = Path(filename)

    if not path.is_file():
        continue

    content = path.read_text(encoding="utf-8")

    if "Modified by Paddy Slator" in content:
        continue

    lines = content.splitlines(keepends=True)

    # Preserve shebang and Python encoding declarations
    insert_at = 0

    if lines and lines[0].startswith("#!"):
        insert_at = 1

    if insert_at < len(lines):
        import re
        if re.match(r"#.*coding[:=]\s*[-\w.]+", lines[insert_at]):
            insert_at += 1

    lines.insert(insert_at, NOTICE + "\n")

    path.write_text("".join(lines), encoding="utf-8")
    print(f"Added notice: {filename}")