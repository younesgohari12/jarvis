"""Learned selection of bounded algorithms; syntax and isolated behavior checks.

Only source strings owned by this module are executed, never user-supplied code.
General tracing/debugging/refactoring remains with the established v19 pipeline.
"""
from __future__ import annotations
import ast
import re
import subprocess
import sys
from functools import lru_cache
from jarvis.agent.semantic_models_v20 import LearnedModel

RECIPES={
 'sum_values':('def solve(values):\n    return sum(values)\n','assert solve([2, -3, 5]) == 4\nassert solve([]) == 0'),
 'sort_values':('def solve(values):\n    return sorted(values)\n','assert solve([3, -1, 3]) == [-1, 3, 3]\nassert solve([]) == []'),
 'filter_even':('def solve(values):\n    return [x for x in values if x % 2 == 0]\n','assert solve([-2, 0, 3, 4]) == [-2, 0, 4]\nassert solve([]) == []'),
 'filter_positive':('def solve(values):\n    return [x for x in values if x > 0]\n','assert solve([-3, 0, 2, 5]) == [2, 5]\nassert solve([]) == []'),
 'max_value':('def solve(values):\n    if not values:\n        raise ValueError("empty input")\n    return max(values)\n','assert solve([-5, -2, -8]) == -2\nassert solve([0]) == 0'),
 'deduplicate':('def solve(values):\n    result = []\n    for value in values:\n        if value not in result:\n            result.append(value)\n    return result\n','assert solve([3, 1, 3, 2]) == [3, 1, 2]\nassert solve([[1], [1]]) == [[1]]'),
 'reverse_string':('def solve(text):\n    return text[::-1]\n','assert solve("abc") == "cba"\nassert solve("") == ""'),
 'palindrome':('def solve(text):\n    clean = "".join(c.casefold() for c in text if c.isalnum())\n    return clean == clean[::-1]\n','assert solve("Never odd or even")\nassert not solve("hello")'),
 'factorial':('def solve(n):\n    if not isinstance(n, int) or n < 0:\n        raise ValueError("nonnegative integer required")\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n','assert solve(0) == 1\nassert solve(6) == 720'),
 'average':('def solve(values):\n    if not values:\n        raise ValueError("empty input")\n    return sum(values) / len(values)\n','assert solve([2, 4, 9]) == 5\nassert solve([-3, 1]) == -1'),
}

@lru_cache(maxsize=16)
def verified_recipe(label):
    if label not in RECIPES: return None
    source,tests=RECIPES[label]
    try:
        ast.parse(source)
        # Fixed, bounded data only. Isolated interpreter with no shell, imports or I/O.
        result=subprocess.run([sys.executable,'-I','-S','-c',source+'\n'+tests],capture_output=True,text=True,timeout=3)
        if result.returncode!=0: return None
    except (SyntaxError,OSError,subprocess.TimeoutExpired): return None
    return source

class CodeIntelligenceV20:
    def __init__(self): self.model=LearnedModel('code_intent'); self.last=None
    def solve(self,text):
        self.last=None
        if re.search(r'```|\bprint\s*\(|\b(?:def|class)\s+\w+|\b(?:for|while)\b[^;\n]*:|(?:^|\n)\s*import\s+|\b\w+\s*=',text): return None
        scores=self.model.predict(text)
        if not scores or scores[0][1]<.65 or scores[0][0] not in RECIPES: return None
        source=verified_recipe(scores[0][0])
        if source is None: return None
        self.last={'label':scores[0][0],'confidence':scores[0][1],'checkpoint':self.model.sha256,'checks':['python_syntax','isolated_behavior_tests']}
        return '```python\n'+source+'```'
