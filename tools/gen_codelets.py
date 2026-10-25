"""Generates codelets.goose: straight-line DFTs of small sizes on cv4s.

    python tools/gen_codelets.py            (from the package root)

Each codelet `dftN(a: cv4[:])` transforms the N cv4s of a in place, forward
and unnormalized, every lane on its own. The DFT is built symbolically as a
split-radix decomposition (radix 2 and 4 butterflies, sizes that are powers
of two) or, for other sizes, as a mixed-radix Cooley-Tukey decomposition
over radices 2, 3, 4 and 5; constants are folded, multiplications by
trivial twiddles (1, -1, +-i, (+-1 +-i)/sqrt 2) become additions, signs are
pushed outward, equal subexpressions are shared, and a multiplication that
only one addition uses fuses into it (fma4 and its kin, in fft.goose). The result is printed
as one let per operation, which the C compiler schedules and vectorizes.
"""
import cmath
import math
import os
import sys

# Sizes past 8 compile badly: clang's SLP vectorizer leaves much of a long
# straight-line block of struct arithmetic scalar, and spills the rest.
POW2 = [2, 4, 8]
MIXED = [3, 5, 6]


class Dag:
    """Real-valued expression DAG with hash-consing. A value is (node, sign)."""

    def __init__(self):
        self.nodes = []          # (op, args...) ; op in in/add/sub/mul
        self.index = {}
        self.consts = {}         # value -> name

    def node(self, key):
        if key not in self.index:
            self.index[key] = len(self.nodes)
            self.nodes.append(key)
        return self.index[key]

    def inp(self, i, part):
        return (self.node(("in", i, part)), 1)

    def add(self, x, y):
        (a, sa), (b, sb) = x, y
        if a is None:
            return y
        if b is None:
            return x
        if sa == sb:
            lo, hi = min(a, b), max(a, b)
            return (self.node(("add", lo, hi)), sa)
        if sa > 0:
            return (self.node(("sub", a, b)), 1)
        return (self.node(("sub", b, a)), 1)

    def neg(self, x):
        return (x[0], -x[1]) if x[0] is not None else x

    def sub(self, x, y):
        return self.add(x, self.neg(y))

    def mul(self, c, x):
        if x[0] is None or c == 0:
            return (None, 1)
        if c == 1:
            return x
        if c == -1:
            return self.neg(x)
        s = 1 if c > 0 else -1
        c = abs(c)
        key = f"{c:.14e}"
        if key not in self.consts:
            self.consts[key] = (f"k{len(self.consts)}", c)
        return (self.node(("mul", key, x[0])), s * x[1])


ZERO = (None, 1)


def cadd(d, x, y):
    return (d.add(x[0], y[0]), d.add(x[1], y[1]))


def csub(d, x, y):
    return (d.sub(x[0], y[0]), d.sub(x[1], y[1]))


def snap(v):
    """Rounds a twiddle component to an exact 0, +-1 or +-sqrt(1/2)."""
    for e in (0.0, 1.0, -1.0, 0.5, -0.5, math.sqrt(0.5), -math.sqrt(0.5), math.sqrt(0.75), -math.sqrt(0.75)):
        if abs(v - e) < 1e-15:
            return e
    return v


def cmul(d, w, x):
    """x times the constant w."""
    wr, wi = snap(w.real), snap(w.imag)
    re, im = x
    if wi == 0:
        return (d.mul(wr, re), d.mul(wr, im))
    if wr == 0:
        # (re + i im) * i wi = -wi im + i wi re
        return (d.mul(-wi, im), d.mul(wi, re))
    if abs(abs(wr) - abs(wi)) < 1e-15:
        # wr (re - s im) + i wr (im + s re), s = wi / wr
        s = 1 if wi / wr > 0 else -1
        a = d.sub(re, im) if s > 0 else d.add(re, im)
        b = d.add(im, re) if s > 0 else d.sub(im, re)
        return (d.mul(wr, a), d.mul(wr, b))
    return (d.sub(d.mul(wr, re), d.mul(wi, im)), d.add(d.mul(wr, im), d.mul(wi, re)))


def w(k, n):
    return cmath.exp(-2j * math.pi * k / n)


def split_radix(d, x):
    n = len(x)
    if n == 1:
        return list(x)
    if n == 2:
        return [cadd(d, x[0], x[1]), csub(d, x[0], x[1])]
    u = split_radix(d, x[0::2])
    z = split_radix(d, x[1::4])
    zp = split_radix(d, x[3::4])
    out = [None] * n
    q = n // 4
    for k in range(q):
        a = cmul(d, w(k, n), z[k])
        b = cmul(d, w(3 * k, n), zp[k])
        s = cadd(d, a, b)
        t = csub(d, a, b)
        mi_t = (t[1], d.neg(t[0]))           # -i t
        out[k] = cadd(d, u[k], s)
        out[k + 2 * q] = csub(d, u[k], s)
        out[k + q] = cadd(d, u[k + q], mi_t)
        out[k + 3 * q] = csub(d, u[k + q], mi_t)
    return out


def small_dft(d, x):
    """Direct DFT of an odd size, the outputs k and n - k sharing their sums:
    X[k] = A + B and X[n-k] = A - B with A = x0 + sum cos(2 pi jk/n) (x_j +
    x_{n-j}) and B = sum sin(2 pi jk/n) (-i)(x_j - x_{n-j})."""
    n = len(x)
    if n == 1:
        return list(x)
    if n == 2:
        return [cadd(d, x[0], x[1]), csub(d, x[0], x[1])]
    assert n % 2 == 1
    h = (n - 1) // 2
    sp = [cadd(d, x[j], x[n - j]) for j in range(1, h + 1)]
    sm = [csub(d, x[j], x[n - j]) for j in range(1, h + 1)]
    out = [None] * n
    acc = x[0]
    for v in sp:
        acc = cadd(d, acc, v)
    out[0] = acc
    for k in range(1, h + 1):
        a = x[0]
        b = ZERO_C
        for j in range(1, h + 1):
            th = 2 * math.pi * ((j * k) % n) / n
            c, sn = snap(math.cos(th)), snap(math.sin(th))
            a = cadd(d, a, (d.mul(c, sp[j - 1][0]), d.mul(c, sp[j - 1][1])))
            # sin * (-i)(sm) = sin * (sm.im, -sm.re)
            b = cadd(d, b, (d.mul(sn, sm[j - 1][1]), d.mul(-sn, sm[j - 1][0])))
        out[k] = cadd(d, a, b)
        out[n - k] = csub(d, a, b)
    return out


ZERO_C = ((None, 1), (None, 1))


def factors(n):
    fs = []
    for f in (4, 2, 3, 5):
        while n % f == 0 and n > 1:
            fs.append(f)
            n //= f
    assert n == 1, n
    return fs


def mixed(d, x):
    """Cooley-Tukey, decimation in time, radix = first factor."""
    n = len(x)
    if n in (1, 2, 3, 4, 5):
        if n == 4:
            return split_radix(d, x)
        return small_dft(d, x)
    if n & (n - 1) == 0:
        return split_radix(d, x)
    r = factors(n)[0]
    if n % 4 != 0 and n % 2 == 0 and n % 3 != 0 and n % 5 != 0:
        r = 2
    m = n // r
    subs = [mixed(d, x[k::r]) for k in range(r)]
    out = [None] * n
    for k in range(m):
        tw = [subs[0][k]] + [cmul(d, w(j * k, n), subs[j][k]) for j in range(1, r)]
        dft = small_dft(d, tw) if r != 4 else split_radix(d, tw)
        for t in range(r):
            out[k + t * m] = dft[t]
    return out


def emit(n):
    d = Dag()
    x = [(d.inp(i, "re"), d.inp(i, "im")) for i in range(n)]
    y = split_radix(d, x) if n & (n - 1) == 0 else mixed(d, x)
    # Which nodes are used, and print them in creation order (a valid
    # topological order: children are always created first).
    used = set()

    def mark(v):
        stack = [v]
        while stack:
            i = stack.pop()
            if i is None or i in used:
                continue
            used.add(i)
            op = d.nodes[i]
            if op[0] in ("add", "sub"):
                stack += [op[1], op[2]]
            elif op[0] == "mul":
                stack.append(op[2])

    for re, im in y:
        mark(re[0])
        mark(im[0])
    # A multiplication by a constant that only one addition uses fuses into
    # it: fma4 (k x + b), fms4 (k x - b) or fnma4 (b - k x).
    uses = {}
    for i in used:
        op = d.nodes[i]
        kids = [op[1], op[2]] if op[0] in ("add", "sub") else [op[2]] if op[0] == "mul" else []
        for c in kids:
            uses[c] = uses.get(c, 0) + 1
    for re, im in y:
        for v in (re[0], im[0]):
            if v is not None:
                uses[v] = uses.get(v, 0) + 1
    fused = {}                       # add node -> (index of the mul operand, mul node)
    absorbed = set()
    for i in sorted(used):
        op = d.nodes[i]
        if op[0] not in ("add", "sub"):
            continue
        for side in (2, 1):
            c = op[side]
            if d.nodes[c][0] == "mul" and uses.get(c) == 1 and c not in absorbed:
                fused[i] = (side, c)
                absorbed.add(c)
                break
    lines = []
    adds = muls = fmas = 0
    names = {}
    for key, (name, c) in sorted(d.consts.items(), key=lambda kv: kv[1][0]):
        if any(d.nodes[i][0] == "mul" and d.nodes[i][1] == key for i in used):
            lines.append(f"    let {name} = splat({c!r});")
    for i, op in enumerate(d.nodes):
        if i not in used or i in absorbed:
            continue
        if op[0] == "in":
            names[i] = f"a[{op[1]}].{op[2]}"
            continue
        nm = f"t{i}"
        names[i] = nm
        if i in fused:
            side, m = fused[i]
            k, x = d.consts[d.nodes[m][1]][0], names[d.nodes[m][2]]
            other = names[op[3 - side]]
            if op[0] == "add":
                lines.append(f"    let {nm} = fma4({k}, {x}, {other});")
            elif side == 2:
                lines.append(f"    let {nm} = fnma4({k}, {x}, {other});")
            else:
                lines.append(f"    let {nm} = fms4({k}, {x}, {other});")
            fmas += 1
        elif op[0] == "add":
            lines.append(f"    let {nm} = {names[op[1]]} + {names[op[2]]};")
            adds += 1
        elif op[0] == "sub":
            lines.append(f"    let {nm} = {names[op[1]]} - {names[op[2]]};")
            adds += 1
        else:
            lines.append(f"    let {nm} = {d.consts[op[1]][0]} * {names[op[2]]};")
            muls += 1

    def val(v):
        node, sign = v
        if node is None:
            return "splat(0.0)"
        return names[node] if sign > 0 else f"splat(0.0) - {names[node]}"

    # Inputs are read into locals first so the stores below can't clobber them.
    loads = [f"    let p{k} = a[{k}];" for k in range(n)]
    body = "\n".join(lines).replace("a[", "p").replace("].re", ".re").replace("].im", ".im")
    stores = [f"    a[{k}] = cv4 {{ {val(re)}, {val(im)} }};".replace("a[", "a[", 1) for k, (re, im) in enumerate(y)]
    stores = [s.replace("{ i", "{ i") for s in stores]
    out_lines = []
    for s in stores:
        # outputs reference inputs by the local names too
        head, rest = s.split("=", 1)
        rest = rest.replace("a[", "p").replace("].re", ".re").replace("].im", ".im")
        out_lines.append(head + "=" + rest)
    text = (f"// {n} points: {adds} additions and {muls} multiplications per lane.\n"
            f"inline fn dft{n}(a: cv4[:]) {{\n    assert(a.len == {n});\n"
            + "\n".join(loads) + "\n" + body + "\n" + "\n".join(out_lines) + "\n}\n")
    return text, adds, muls, fmas


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sizes = sorted(POW2 + MIXED)
    parts = ["// Generated by tools/gen_codelets.py; do not edit.\n//\n"
             "// Straight-line DFTs on cv4s, forward and unnormalized: dftN(a) transforms the\n"
             "// N cv4s of a in place, each of the four lanes on its own. `codelet(a)` picks\n"
             "// the one for a.len, and `has_codelet(n)` says whether there is one.\n\n"
             "namespace fft;\n"]
    for n in sizes:
        text, adds, muls, fmas = emit(n)
        parts.append("\n" + text)
        print(f"dft{n}: {adds} adds, {muls} muls, {fmas} fmas", file=sys.stderr)
    cases = "\n".join(f"        {n} => dft{n}(a)," for n in sizes)
    parts.append("\nfn has_codelet(n: i64) -> bool {\n    match n {\n"
                 + "\n".join(f"        {n} => true," for n in sizes)
                 + "\n        _ => false,\n    }\n}\n")
    parts.append("\n// The codelet for a.len, which must have one.\ninline fn codelet(a: cv4[:]) {\n    match a.len {\n"
                 + cases + "\n        _ => abort(\"no codelet for this length\"),\n    }\n}\n")
    with open(os.path.join(root, "codelets.goose"), "w", newline="\n") as f:
        f.write("".join(parts))


main()
