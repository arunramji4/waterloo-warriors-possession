def smooth_short_runs(labels, blocks, minimum_frames=50):
    """Shortest run first; absorb into longer neighbor, ties to the earlier one.

    X is a hard boundary. Never bridge halves or abstentions. Exactly 50-frame
    runs survive. A sole short run between boundaries has no neighbor to use.
    """
    output = labels.copy()
    removed = 0
    for first, end in blocks:
        runs, start = [], first
        for frame in range(first+1, end+1):
            if frame == end or labels[frame] != labels[start]:
                runs.append([start, frame, labels[start]])
                start = frame
        while True:
            candidates = []
            for i, (a, b, value) in enumerate(runs):
                if value == 'X' or b-a >= minimum_frames:
                    continue
                neighbors = [j for j in (i-1, i+1) if 0 <= j < len(runs) and runs[j][2] != 'X']
                if neighbors:
                    candidates.append((b-a, a, i, neighbors))
            if not candidates:
                break
            _, _, index, neighbors = min(candidates)
            neighbor = max(neighbors, key=lambda j: (runs[j][1]-runs[j][0], -j))
            runs[index][2] = runs[neighbor][2]
            merged = []
            for a, b, value in runs:
                if merged and merged[-1][2] == value:
                    merged[-1][1] = b
                else:
                    merged.append([a, b, value])
            runs = merged
            removed += 1
        for a, b, value in runs:
            output[a:b] = [value]*(b-a)
    assert all(a == b for a, b in zip(labels, output) if a == 'X')
    return output, removed
