#!/usr/bin/env python3
with open('area.xvg', 'w') as OUT:
    for line in open('box-xy.xvg'):
        if line.startswith(('#','@')):
            continue
        t, x, y = map(float, line.split())
        area = x * y / 64
        print(f'{t:.4f} {area:.4f}', file=OUT)
