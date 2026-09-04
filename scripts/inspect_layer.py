"""Exploratory script: dump the structure of an Enfusion .layer file.

Usage: python scripts/inspect_layer.py sample.layer [--max-depth N]

This is a throwaway inspection tool (Step 0 of CLAUDE.md). It loads the whole
file into a tree, which is fine for small samples but NOT how the real parser
works (that one streams).
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

HEADER_RE = re.compile(
    r'^(?P<grp>\$grp\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)?\s*'
    r'(?::\s*"(?P<prefab>[^"]*)")?\s*(?:"(?P<guid>[^"]*)")?\s*\{$'
)


@dataclass
class Node:
    kind: str                # 'grp' | 'entity' | 'instance' | 'block' | 'root'
    name: str | None = None
    prefab: str | None = None
    props: dict = field(default_factory=dict)
    children: list["Node"] = field(default_factory=list)
    line: int = 0


def parse(path: str) -> Node:
    root = Node("root")
    stack = [root]
    with open(path, encoding="utf-8", errors="replace") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            if line == "}":
                stack.pop()
                continue
            if line.endswith("{"):
                m = HEADER_RE.match(line)
                if not m:
                    raise ValueError(f"{lineno}: unparsed header {line!r}")
                if m.group("grp"):
                    kind = "grp"
                elif m.group("name") is None:
                    kind = "instance"      # anonymous block
                else:
                    kind = "entity" if (m.group("prefab") is not None or m.group("guid") is not None) else "block"
                node = Node(kind, m.group("name"), m.group("prefab"), line=lineno)
                stack[-1].children.append(node)
                stack.append(node)
                continue
            key, _, val = line.partition(" ")
            stack[-1].props[key] = val
    assert len(stack) == 1, f"unbalanced braces, depth {len(stack)-1} at EOF"
    return root


def walk(node: Node, depth=0, path=()):
    for ch in node.children:
        label = f"{ch.kind}:{ch.name or '<anon>'}"
        yield ch, depth, path + (label,)
        yield from walk(ch, depth + 1, path + (label,))


def main():
    path = sys.argv[1]
    root = parse(path)

    print("=== structural paths (kind:name chain -> count) ===")
    counts = Counter()
    for node, depth, p in walk(root):
        counts[" / ".join(p)] += 1
    for p, c in sorted(counts.items()):
        print(f"{c:6d}  {p}")

    print("\n=== representative node per class (first occurrence) ===")
    seen = set()
    for node, depth, p in walk(root):
        key = (node.kind, node.name)
        if key in seen:
            continue
        seen.add(key)
        print(f"- line {node.line}: kind={node.kind} name={node.name} prefab={node.prefab}")
        print(f"    props={node.props}")
        print(f"    n_children={len(node.children)} child kinds={Counter((c.kind, c.name) for c in node.children)}")

    # World coordinate resolution: walk with accumulated parent coords.
    print("\n=== world-space resolution of Tree entities ===")
    trees = []
    rotated_parents = []

    def vec(s):
        return tuple(float(x) for x in s.split())

    def resolve(node, origin=(0.0, 0.0, 0.0), chain=()):
        for ch in node.children:
            local = vec(ch.props["coords"]) if "coords" in ch.props else None
            if "angles" in ch.props and ch.children and any(abs(a) > 1e-9 for a in vec(ch.props["angles"])):
                rotated_parents.append((ch.line, ch.name, ch.props["angles"]))
            if local is not None:
                world = tuple(o + l for o, l in zip(origin, local))
            else:
                world = origin
            # groups: instance blocks under a grp inherit the grp's name/prefab
            if ch.kind == "grp":
                for inst in ch.children:
                    ilocal = vec(inst.props["coords"]) if "coords" in inst.props else (0.0, 0.0, 0.0)
                    iworld = tuple(o + l for o, l in zip(origin, ilocal))
                    if ch.name == "Tree":
                        trees.append((ch.prefab.rsplit("/", 1)[-1], iworld, inst.props.get("angles"), inst.props.get("scale")))
                    resolve(inst, iworld, chain + (ch.name,))
            elif ch.kind == "entity":
                if ch.name == "Tree":
                    trees.append((ch.prefab.rsplit("/", 1)[-1], world, ch.props.get("angles"), ch.props.get("scale")))
                resolve(ch, world, chain + (ch.name,))
            else:  # instance/anon block, named block
                resolve(ch, world, chain)

    resolve(root)
    print(f"tree entities found: {len(trees)}")
    xs = [t[1][0] for t in trees]; ys = [t[1][1] for t in trees]; zs = [t[1][2] for t in trees]
    print(f"world X range: {min(xs):.3f} .. {max(xs):.3f}")
    print(f"world Y range (height): {min(ys):.3f} .. {max(ys):.3f}")
    print(f"world Z range: {min(zs):.3f} .. {max(zs):.3f}")
    print("first 5 trees:")
    for t in trees[:5]:
        print("  ", t)
    print("prefab basename counts:")
    for name, c in Counter(t[0] for t in trees).most_common():
        print(f"  {c:5d}  {name}")
    print(f"parents with children AND non-zero angles: {rotated_parents[:10]} (total {len(rotated_parents)})")


if __name__ == "__main__":
    main()
