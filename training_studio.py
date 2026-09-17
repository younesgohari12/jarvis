from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.lab.app import JarvisLab  # noqa: E402


def main() -> int:
    root = tk.Tk()
    JarvisLab(root, ROOT)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
