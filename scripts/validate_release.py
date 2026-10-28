"""校验发布元数据，并用明确的文件清单生成不含本地账号数据的插件包。"""

from pathlib import Path

import argparse
import hashlib
import json
import re
import struct
import zipfile


PLUGIN_ID = "local.bilibili-autopilot"
FILES = (
    "__init__.py", "_manifest.json", "plugin.py", "client.py", "config.py", "engine.py",
    "models.py", "sharing.py", "state.py", "webui.py", "webui.json", "icon.png",
    "README.md", "CHANGELOG.md", "LICENSE", "requirements.txt", "pyproject.toml",
    "config.example.toml", "docs/PUBLISHING.md", "docs/MARKETPLACE_SUBMISSION.md", "docs/ICON.md",
    "docs/OPTIMIZATION.md", "docs/share_samples.json", "docs/share_prompt_v063.json",
    "scripts/preview_shares.py", "scripts/run_tests.py", "scripts/validate_release.py",
    "tests/test_autopilot.py", "tests/test_client_plugin.py", "tests/test_recovery.py", "tests/test_replies.py",
    "tests/test_sharing.py", "tests/test_webui.py", "tests/test_handoff.py", "tests/test_candidates.py",
)


def validate(root: Path, tag: str | None) -> tuple[str, str]:
    for name in FILES:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"缺少发布文件或文件为符号链接：{name}")
    manifest = json.loads((root / "_manifest.json").read_text(encoding="utf-8"))
    if manifest["manifest_version"] != 2 or manifest["id"] != PLUGIN_ID:
        raise ValueError("manifest 必须为 v2，且插件 ID 必须保持不变")
    version = manifest["version"]
    if re.fullmatch(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)", version) is None:
        raise ValueError("版本号必须是 X.Y.Z")
    if tag is not None and tag not in {version, f"v{version}"}:
        raise ValueError("Tag 必须是 manifest 的 X.Y.Z 或 vX.Y.Z")
    repo = manifest["urls"]["repository"]
    if re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) is None or repo.endswith(".git"):
        raise ValueError("仓库必须填写公开 GitHub 仓库的 HTTPS 地址，不带 .git")
    if not manifest["author"]["name"] or not manifest["author"]["url"].startswith("https://github.com/"):
        raise ValueError("缺少真实作者信息")
    if manifest["license"] != "MIT" or "MIT License" not in (root / "LICENSE").read_text(encoding="utf-8"):
        raise ValueError("清单与许可证不一致")
    project = (root / "pyproject.toml").read_text(encoding="utf-8")
    if re.search(rf'^version\s*=\s*"{re.escape(version)}"\s*$', project, re.MULTILINE) is None:
        raise ValueError("pyproject.toml 版本与 manifest 不一致")
    if f"**Version:** {version}" not in (root / "README.md").read_text(encoding="utf-8"):
        raise ValueError("README 顶部版本与 manifest 不一致")
    icon = manifest.get("display", {}).get("icon", {})
    if icon.get("type") != "local" or icon.get("value") != "icon.png":
        raise ValueError("必须声明本地图标 icon.png")
    png = (root / "icon.png").read_bytes()
    if len(png) > 512 * 1024 or png[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("图标必须是 PNG 且不超过 512 KB")
    width, height = struct.unpack(">II", png[16:24])
    if width != height or not 64 <= width <= 1024:
        raise ValueError("图标应为 64～1024 像素的正方形")
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = re.search(rf"^## v?{re.escape(version)}(?:[ \t].*)?$", changelog, re.MULTILINE)
    if heading is None:
        raise ValueError("CHANGELOG 缺少当前版本")
    rest = changelog[heading.end():]
    next_heading = re.search(r"^## ", rest, re.MULTILINE)
    notes = (rest[:next_heading.start()] if next_heading else rest).strip()
    if not notes:
        raise ValueError("当前版本的更新说明不能为空")
    for name in ("README.md", "_manifest.json", "docs/PUBLISHING.md", "docs/MARKETPLACE_SUBMISSION.md"):
        if "https://localhost" in (root / name).read_text(encoding="utf-8"):
            raise ValueError(f"发布资料仍含本地占位链接：{name}")
    return version, notes + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tag")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--notes", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    version, notes = validate(root, args.tag)
    if args.notes:
        args.notes.parent.mkdir(parents=True, exist_ok=True)
        args.notes.write_text(notes, encoding="utf-8")
    if args.archive:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        # 明确列出公共文件，绝不遍历复制本地配置、数据库或登录态。
        with zipfile.ZipFile(args.archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in FILES:
                archive.write(root / name, name)
        digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
        args.archive.with_suffix(args.archive.suffix + ".sha256").write_text(
            f"{digest}  {args.archive.name}\n", encoding="utf-8")
        print(f"已生成发布包：{args.archive}（SHA256 {digest}）")
    print(f"发布检查通过：{PLUGIN_ID} v{version}")


if __name__ == "__main__":
    main()
