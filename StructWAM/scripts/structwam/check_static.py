#!/usr/bin/env python3
"""Read-only checks for the StructWAM changes: syntax, intra-repo imports, config keys.

* every Python file under openwam/ and scripts/structwam/ compiles;
* every ``openwam.*`` name imported by the StructWAM files exists in the repo source (AST only,
  nothing is imported, so heavy optional dependencies do not matter);
* the StructWAM configs parse and reference existing config files.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STRUCT_FILES = [
    "openwam/structure/__init__.py", "openwam/structure/history.py", "openwam/structure/layout.py",
    "openwam/structure/losses.py", "openwam/structure/phase.py", "openwam/structure/routing.py",
    "openwam/structure/sketch.py", "openwam/structure/student.py", "openwam/structure/tokens.py",
    "openwam/model/architectures/tri_system/struct_expert.py",
    "openwam/model/architectures/tri_system/struct_driver.py",
    "openwam/model/architectures/tri_system/struct_system.py",
    "openwam/dataloader/libero_struct.py",
]


def module_names(path: Path) -> set:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return names


def resolve(module: str):
    base = ROOT / Path(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    errors = []
    for path in sorted((ROOT / "openwam").rglob("*.py")) + sorted((ROOT / "scripts/structwam").glob("*.py")):
        try:
            compile(path.read_text(), str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"syntax: {path.relative_to(ROOT)}: {exc}")
    for rel in STRUCT_FILES:
        tree = ast.parse((ROOT / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("openwam"):
                target = resolve(node.module)
                if target is None:
                    errors.append(f"import: {rel}: no module {node.module}")
                    continue
                if target.name == "__init__.py" and resolve(f"{node.module}.{node.names[0].name}"):
                    continue
                defined = module_names(target)
                for alias in node.names:
                    if alias.name not in defined and not resolve(f"{node.module}.{alias.name}"):
                        errors.append(f"import: {rel}: {node.module} has no {alias.name}")
    try:
        import yaml
    except ImportError:
        yaml = None
    if yaml is not None:
        for rel in ("configs/model/tri_system_struct.yaml", "configs/dataloader/libero_struct.yaml",
                    "configs/model/video_backbone/cosmos_predict25_2b_robot.yaml", "configs/train_struct.yaml"):
            data = yaml.safe_load((ROOT / rel).read_text())
            for entry in data.get("defaults", []) or []:
                if entry == "_self_":
                    continue
                if isinstance(entry, str):
                    candidates = [ROOT / "configs" / f"{entry}.yaml"]
                    label = entry
                else:
                    group, name = next(iter(entry.items()))
                    group = group.replace("override ", "")
                    label = f"{group}/{name}"
                    # group configs live next to the including file's group or at the configs root
                    candidates = [ROOT / "configs" / group / f"{name}.yaml",
                                  ROOT / "configs" / "model" / group / f"{name}.yaml"]
                if not any(c.is_file() for c in candidates):
                    errors.append(f"config: {rel}: missing {label}.yaml")
    for line in errors:
        print(line)
    print(f"{len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
