"""R2 Red-team attack vector tests."""
from openspace.security import check_code_safety
from openspace.skill_engine.review_gate import check_ast_safety
from openspace.skill_engine.types import SkillLineage, SkillOrigin, SkillRecord


def test_gate(label, py_code):
    record = SkillRecord(
        skill_id="test", name="test", description="test", path="/t/SKILL.md",
        lineage=SkillLineage(
            origin=SkillOrigin.FIXED, generation=1,
            parent_skill_ids=["p"],
            content_snapshot={"SKILL.md": "name: test\n", "handler.py": py_code}
        )
    )
    result = check_ast_safety(record)
    print(f"  {label}: verdict={result.verdict}, detail={result.detail[:140]}")
    return result


print("=" * 60)
print("BYPASS VECTOR TESTING")
print("=" * 60)

# 1. Aliased module: import os as o; o.system()
print("\n--- 1. Aliased module name ---")
test_gate("import os as o; o.system()", "import os as o\no.system('cmd')\n")

# 2. Double-layer: exec(compile(..., 'exec'))
print("\n--- 2. exec(compile(..., 'exec')) ---")
test_gate("exec+compile", "exec(compile('import os', '', 'exec'))\n")

# 3. Renamed import: from os import system as x; x()
print("\n--- 3. from os import system as x; x() ---")
test_gate("aliased bare name", "from os import system as x\nx('id')\n")

# 4. globals() dispatch
print("\n--- 4. globals()['eval']('...') ---")
test_gate("globals dispatch", "globals()['eval']('1')\n")

# 5. __builtins__.__import__
print("\n--- 5. __builtins__.__import__ ---")
test_gate("builtins import", "import builtins\nbuiltins.__import__('os')\n")

# 6. getattr on non-module (should NOT block)
print("\n--- 6. getattr on data object (should pass) ---")
test_gate("getattr on data", "class C: pass\nc = C()\ngetattr(c, 'x')\n")

# 7. lambda + __import__
print("\n--- 7. lambda + __import__() ---")
test_gate("lambda __import__", "(lambda: __import__('os'))()\n")

# 8. compile with 'exec' mode
print("\n--- 8. compile(..., 'exec') ---")
test_gate("compile exec mode", "compile('x=1', '', 'exec')\n")

# 9. compile with 'eval' mode (should not trigger)
print("\n--- 9. compile(..., 'eval') should pass ---")
test_gate("compile eval mode", "compile('1+1', '', 'eval')\n")

# 10. os.path.join (benign, should pass)
print("\n--- 10. os.path.join (benign) ---")
test_gate("os.path.join", "import os\nos.path.join('a', 'b')\n")

# 11. Unicode confusable: fullwidth 'e' in eval
print("\n--- 11. Unicode confusable eval ---")
try:
    test_gate("fullwidth eval", "\uff45val('1+1')\n")
except Exception as e:
    print(f"  Error (expected?): {e}")

# 12. String concatenation to build dangerous call
print("\n--- 12. getattr(os, 'sys'+'tem')('id') ---")
test_gate("getattr concat", "import os\ngetattr(os, 'sys'+'tem')('id')\n")

# 13. type() constructor trick
print("\n--- 13. type metaclass with __init_subclass__ ---")
test_gate("type constructor", "type('X', (), {'__init_subclass__': lambda **kw: None})\n")

# 14. Nested exec in exec
print("\n--- 14. exec('exec(...)') ---")
test_gate("nested exec", "exec('exec(chr(49))')\n")

# 15. importlib.import_module bare call
print("\n--- 15. importlib.import_module ---")
test_gate("importlib", "import importlib\nimportlib.import_module('os')\n")

# 16. os.* wildcard attribute access (should be caught by import rule)
print("\n--- 16. from os import * ---")
test_gate("from os import *", "from os import *\n")

# 17. Chained getattr: getattr(getattr(...), 'system')
print("\n--- 17. chained getattr ---")
test_gate("chained getattr", "import os\ngetattr(getattr(os, 'path'), 'join')('a','b')\n")

# 18. sys.modules manipulation
print("\n--- 18. sys.modules['os'] ---")
test_gate("sys.modules", "import sys\nsys.modules['os'].system('id')\n")

print("\n" + "=" * 60)
print("DONE")
