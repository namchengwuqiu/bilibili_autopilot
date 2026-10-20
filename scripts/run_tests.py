"""从任意仓库目录名运行现有业务测试，不访问真实平台账号。"""

from pathlib import Path

import importlib.util
import sys
import unittest


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    name = "local_bilibili_autopilot"
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py",
                                                submodule_search_locations=[str(root)])
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载插件测试包")
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
