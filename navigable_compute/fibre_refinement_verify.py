"""Independent checker of persistent intersection and old-coordinate embedding."""
from .intention_fibre import Fibre, MAX_NODES, MAX_WORK
from .intention_fibre_verify import verify_fibre, _canonical, _substitute
from .fibre_refinement import Refinement, OverlayNode


def verify_refinement(parent, equations, certificate):
    """Check the product of the old graph and the added residual equations."""
    try:
        if type(parent) is not Fibre or not verify_fibre(parent.width, parent.equations, parent):
            return False
        width = parent.width
        if (type(equations) is not tuple or len(equations) > 1024
                or any(type(p) is not tuple or len(p) > 2048
                       or any(type(m) is not int or not 0 <= m < 2**width for m in p)
                       or tuple(sorted(set(p))) != p for p in equations)):
            return False
        rows = _canonical(equations)
        c = certificate
        if (type(c) is not Refinement or c.parent != parent or c.equations != rows
                or type(c.nodes) is not tuple or len(c.nodes) > MAX_NODES
                or type(c.work) is not int or not 0 <= c.work <= MAX_WORK
                or type(c.root) is not int or not -(len(parent.nodes)+1) <= c.root <= len(c.nodes)):
            return False
        seen, keys = {}, set()
        work = 0
        stack = [(c.root, parent.root, rows)]
        while stack:
            ref, p, rows = stack.pop()
            if rows == ((0,),) or parent.count_at(p) == 0:
                if ref != 0:
                    return False
                continue
            if not rows:
                if ref != -p:
                    return False
                continue
            if p < 2 or ref <= 0:
                return False
            key = (p, rows)
            if ref in seen:
                if seen[ref] != key:
                    return False
                continue
            if key in keys:
                return False
            n = c.nodes[ref - 1]
            pn = parent.nodes[p - 2]
            if (type(n) is not OverlayNode or type(n.parent_id) is not int
                    or n.parent_id != p or n.equations != rows
                    or any(type(v) is not int or not -(len(parent.nodes)+1) <= v < ref
                           for v in (n.low, n.high))
                    or type(n.count) is not int or not 0 <= n.count <= pn.count):
                return False
            counts = [parent.count_at(-v) if v <= 0 else c.nodes[v-1].count
                      for v in (n.low, n.high)]
            if any(type(v) is not int for v in counts) or n.count != sum(counts):
                return False
            work += 1 + 2 * sum(1 + len(row) for row in rows)
            if work > MAX_WORK:
                return False
            seen[ref] = key
            keys.add(key)
            k = pn.remaining - 1
            stack.append((n.low, pn.low, _substitute(rows, k, 0)))
            stack.append((n.high, pn.high, _substitute(rows, k, 1)))
        return len(seen) == len(c.nodes) and work == c.work
    except (ValueError, TypeError, IndexError, AttributeError, OverflowError):
        return False
