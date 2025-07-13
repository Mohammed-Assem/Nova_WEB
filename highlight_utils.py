def point_to_error(src, start, end):
    marker = ' '
    line_start = max(src.rfind('\n', 0, start.idx), 0)
    line_end = src.find('\n', line_start + 1)
    if line_end < 0: line_end = len(src)

    lines = end.line - start.line + 1
    for i in range(lines):
        segment = src[line_start:line_end]
        start_col = start.col if i == 0 else 0
        end_col = end.col if i == lines - 1 else len(segment) - 1

        marker += segment + '\n'
        marker += ' ' * start_col + '^' * (end_col - start_col)

        line_start = line_end
        line_end = src.find('\n', line_start + 1)
        if line_end < 0: line_end = len(src)

    return marker.replace('\t', '')
