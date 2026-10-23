"""Check that the scripts in code/ differ from the authors' working copies only in comments and docstrings.

For every .py file in the cleaned directory, the module is parsed, every docstring (the leading string
statement of a module, class or function body) is removed, and the resulting ast.dump is compared with that
of the original file of the same name. Comments are not part of the AST, and ast.dump omits line and column
numbers, so two files pass exactly when their executable code is identical.

Usage: python tools/check_identical.py ORIGINAL_CODE_DIR [CLEANED_CODE_DIR]
CLEANED_CODE_DIR defaults to code/ next to this tools/ folder. Exits with status 1 if any file differs or
has no original.
"""
import ast
import os
import sys


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:]
    return tree


def code_dump(path):
    with open(path, encoding="utf-8") as fh:
        return ast.dump(strip_docstrings(ast.parse(fh.read(), filename=path)))


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    orig_dir = argv[1]
    clean_dir = argv[2] if len(argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code")
    files = sorted(f for f in os.listdir(clean_dir) if f.endswith(".py"))
    bad = 0
    for f in files:
        o = os.path.join(orig_dir, f)
        if not os.path.exists(o):
            print(f"MISSING  {f}: no original in {orig_dir}")
            bad += 1
            continue
        same = code_dump(o) == code_dump(os.path.join(clean_dir, f))
        print(f"{'OK      ' if same else 'DIFFERS '} {f}")
        bad += not same
    print(f"{len(files) - bad} of {len(files)} files identical after removing docstrings and comments")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
