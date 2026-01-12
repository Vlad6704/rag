import re

SQL_START_RE = re.compile(r"^(SELECT|UPDATE|INSERT|DELETE|CREATE|DROP|ALTER|BEGIN|COMMIT|ROLLBACK)\b", re.IGNORECASE)

def is_code_like_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith("$ "):
        return True
    if s.startswith("mydb=>") or s.startswith("mydb=#"):
        return True
    if SQL_START_RE.match(s):
        return True
    if "|" in s:  # table output / psql aligned output
        return True
    if s.startswith("-----") or s.startswith("----") or s.startswith("-----------"):
        return True
    return False

def group_lines_into_units(text: str, max_code_lines: int = 30) -> list[str]:
    """
    Splits text into 'units' while keeping code/table-like lines together,
    but limits code blocks to max_code_lines to avoid huge units.
    """
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    units: list[str] = []

    buf: list[str] = []
    in_code_block = False
    code_lines = 0

    def flush():
        nonlocal buf, in_code_block, code_lines
        if buf:
            units.append("\n".join(buf).strip())
        buf = []
        in_code_block = False
        code_lines = 0

    for ln in lines:
        if not ln.strip():
            flush()
            continue

        if in_code_block:
            buf.append(ln)
            code_lines += 1

            # HARD CAP: split very long code/table blocks
            if code_lines >= max_code_lines:
                flush()
            continue

        # Not in code block
        if is_code_like_line(ln):
            flush()
            in_code_block = True
            buf.append(ln)
            code_lines = 1
        else:
            buf.append(ln)

    flush()
    return [u for u in units if u.strip()]
