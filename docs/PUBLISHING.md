# 发布到麦麦插件市场

本项目沿用 Gemini 绘图插件的 Tag → 版本校验 → GitHub Release 流程，另加入自动测试和不含本地配置的 ZIP 附件。

## 仓库与插件身份

- 公开仓库：`https://github.com/namchengwuqiu/bilibili_autopilot`。如果采用其他地址，请同步修改清单、README 和市场登记文案。
- 作者：sakura桜花（`namchengwuqiu`）。
- 插件 ID：`local.bilibili-autopilot`，保持不变。修改 ID 会改变配置、账号数据目录及 WebUI 路由。
- 首个市场准备版本：`0.6.0`。配置格式版本 `plugin.config_version=0.1.0` 与发行版本用途不同，无需随每次发版修改。

## 首次发布

1. 在公开仓库默认分支的根目录提交源码和资源，确保 `_manifest.json`、`plugin.py`、`README.md`、`LICENSE` 都位于根目录。
2. 运行 `python scripts/validate_release.py --tag v0.6.0`。在安装 `requirements.txt` 所列依赖后，运行 `python scripts/run_tests.py`。
3. 确认源码仓库不包含 `config.toml`、`config_back/`、登录凭据、二维码、数据库及运行数据。默认示例仅使用 `config.example.toml`。
4. 创建并推送 `v0.6.0` Tag，GitHub Actions 将执行发布检查和模拟测试，随后创建正式 GitHub Release，并附加插件 ZIP 与 SHA256 文件。
5. 确认 Release 已公开发布，随后按 [市场登记文案](MARKETPLACE_SUBMISSION.md) 在插件中心创建新增插件 Issue。
6. 等待市场验证、同步及审核收录。只有仓库代码或 Tag 不满足首次登记要求，草稿 Release 也不满足要求。

手动生成用于检查的发布附件：

```sh
python scripts/validate_release.py --tag v0.6.0 \
  --archive dist/bilibili-autopilot.zip --notes dist/release-notes.md
```

ZIP 根目录直接包含插件文件，不套多层目录。脚本使用固定公共文件清单，因此不会将正在运行的本地配置或数据目录打包。

## B 站依赖升级检查

`client.py` 为确保签名准备完成后只发送一次评论请求，使用了 `bilibili-api-python` 的私有方法 `Api._prepare_request`。当前依赖固定为 `==17.4.2`。升级该依赖前，应检查私有方法签名、返回值和实际请求行为，并重新运行 `tests/test_client_plugin.py` 中的真实库签名、单次 POST、跟进回复楼层参数、签名期间暂停和未知结果不重试等测试；不要只修改依赖版本号后直接发布。

## 后续发版

1. 同步更新 `_manifest.json` 和 `pyproject.toml` 的版本、README 顶部版本号，并在 CHANGELOG 顶部编写该版本说明。
2. 更新本文件中的发版示例到目标版本，重新运行校验和测试。
3. 创建与清单版本一致的 `vX.Y.Z` 或 `X.Y.Z` Tag 并推送。版本匹配失败、缺少图标或更新日志、测试失败时，工作流不会发布。
4. 如果已有 Tag 的发布任务失败，可修复发布环境后，通过 Actions 的 `Publish plugin release` 手动输入该 Tag 重试。已收录版本不要移动 Tag 或替换代码；修复应使用新版本号。

工作流只使用仓库自身的 `GITHUB_TOKEN`，不需要配置 B 站 Cookie、模型密钥或个人 GitHub Token。Release job 需要仓库允许 Actions 写入 contents 的权限。

官方规则：[插件中心贡献指南](https://github.com/Mai-with-u/plugin-repo/blob/main/CONTRIBUTING.md)。
