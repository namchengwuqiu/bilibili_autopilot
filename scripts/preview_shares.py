"""只生成分享文案对照，不读取平台、运行账号数据库或发送消息。

通过 --adapter module:function 注入 Host 模型调用；适配器 async(prompt, config) 返回正文字符串。
"""

from pathlib import Path
import argparse
import asyncio
import importlib
import importlib.util
import json
import re
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
package_spec = importlib.util.spec_from_file_location(
    "local_bilibili_autopilot", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
package = importlib.util.module_from_spec(package_spec)
sys.modules["local_bilibili_autopilot"] = package
package_spec.loader.exec_module(package)
from local_bilibili_autopilot.config import AutopilotConfig
from local_bilibili_autopilot.models import VideoItem, normalized, parse_share_text, share_prompt
from local_bilibili_autopilot.plugin import BilibiliAutopilotPlugin


def old_prompt(item, context, persona, style):
    path = ROOT / "docs" / "share_prompt_v063.json"
    system = json.loads(path.read_text())
    system += f"\n麦麦人设与聊天风格：{persona[:4000]}\n补充风格：{style}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps({"title": item.title, **context}, ensure_ascii=False)},
    ]


def original_parse(raw):
    # 保留 v0.6.3 原校验，避免新经历拦截把对照组失败掩盖为空白。
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)[:-3].strip()
    obj = json.loads(text)
    if not isinstance(obj, dict) or set(obj) != {"text"} or not isinstance(obj["text"], str):
        raise ValueError("分享文案格式无效")
    text = obj["text"].strip()
    if (not 2 <= len(text) <= 80 or not normalized(text)
            or re.search(r"https?://|www\.|BV[0-9A-Za-z]{10}|@", text, re.I)
            or any(unicodedata.category(c) == "Cc" for c in text)):
        raise ValueError("分享文案长度或内容无效")
    return text


def summarize_calls(calls):
    result = {"calls": len(calls), "elapsed_seconds": round(sum(c["elapsed_seconds"] for c in calls), 3)}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [c[key] for c in calls if isinstance(c.get(key), (int, float))]
        result[key] = sum(values) if values else None
    result["reported_calls"] = calls
    return result


async def main(args):
    module, func = args.adapter.split(":", 1)
    adapter = importlib.import_module(module)
    generate = getattr(adapter, func)
    cfg = AutopilotConfig()
    cfg.share.polish_enabled = True
    if args.model:
        cfg.model.candidates = [args.model]
    if args.persona_file:
        args.persona = args.persona_file.read_text()
    plugin = BilibiliAutopilotPlugin()
    plugin.set_plugin_config(cfg.model_dump())

    async def persona():
        return args.persona

    plugin._persona = persona
    plugin._generate = generate
    samples = json.loads(args.samples.read_text())
    result = []
    for sample in [*samples, *samples[: args.repeats]]:
        item = VideoItem("BV" + str(sample["id"]).zfill(10), sample["id"], sample["title"], "2", "样本作者", 120)
        context = {
            "content_evidence": sample["evidence"],
            "my_comment": sample["comment"],
            "reason": "样本关注点",
            "recent_shares": [row["polished"] for row in result[-4:] if row.get("polished")],
        }
        row = {**sample, "run": sum(r["id"] == sample["id"] for r in result) + 1, "review": "待用户或指定评审者评审"}
        for label, operation in [
            ("original", lambda: generate(old_prompt(item, context, args.persona, ""), cfg)),
            ("prompt_only", lambda: generate(share_prompt(item, context, persona=args.persona, extra_style=""), cfg)),
        ]:
            started = time.monotonic()
            metric_start = len(getattr(adapter, "metrics", []))
            try:
                raw = await asyncio.wait_for(operation(), timeout=cfg.share.total_timeout_seconds)
                row[label + "_raw"] = raw
                row[label] = (original_parse if label == "original" else parse_share_text)(raw)
            except Exception as exc:
                row[label] = ""
                row[label + "_error"] = getattr(exc, "code", type(exc).__name__)
            row[label + "_seconds"] = round(time.monotonic() - started, 3)
            row[label + "_usage"] = summarize_calls(getattr(adapter, "metrics", [])[metric_start:])
        started = time.monotonic()
        metric_start = len(getattr(adapter, "metrics", []))
        trace = {}
        try:
            row["polished"] = await asyncio.wait_for(
                plugin._share_steps(item, context, cfg, trace=trace), timeout=cfg.share.total_timeout_seconds
            )
        except Exception as exc:
            row["polished"] = ""
            row["polished_error"] = getattr(exc, "code", type(exc).__name__)
        row["polished_seconds"] = round(time.monotonic() - started, 3)
        row["polished_trace"] = trace
        row["polished_usage"] = summarize_calls(getattr(adapter, "metrics", [])[metric_start:])
        result.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"样本 {sample['id']} 第{row['run']}次：对照已保存", flush=True)
    report = args.output.with_suffix(".md")
    lines = [
        "# 群分享文案对照预览",
        "",
        f"{len(samples)}条构造内容样本，加{args.repeats}条重复生成；所有输出仅预览。自然度和人设一致性待人工评审。",
        "",
        "原提示词组保留v0.6.3校验，仅改提示组使用新格式校验；完整组另有草稿复核、一次润色和润色复核。空正文表示拦截或生成失败，不能发送；原组输出也不能直接使用。",
        "",
        "|题材与样本|原提示词|仅修改提示词|完整流程草稿|最终文案|结果|",
        "|---|---|---|---|---|---|",
    ]
    for r in result:
        clean = lambda x: x.replace("|", "／").replace("\n", " ")
        lines.append(
            "|"
            + "|".join(
                clean(str(x))
                for x in [
                    r["category"] + " " + r["title"] + " #" + str(r["run"]),
                    r["original"],
                    r["prompt_only"],
                    r["polished_trace"].get("draft", ""),
                    r["polished"],
                    r.get("polished_error") or r["polished_trace"].get("result_mode", ""),
                ]
            )
            + "|"
        )
    lines += ["", "耗时与模型可返回的用量：", "", "|组别|模型调用数|平均耗时（秒/样本）|prompt tokens|completion tokens|total tokens|", "|---|---|---|---|---|---|"]
    for label in ("original", "prompt_only", "polished"):
        calls = [c for r in result for c in r[label + "_usage"]["reported_calls"]]
        usage = summarize_calls(calls)
        avg = round(sum(r[label + "_seconds"] for r in result) / len(result), 3)
        lines.append("|" + "|".join(map(str, [label, usage["calls"], avg, usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]])) + "|")
    lines += ["", "输出为实际配置模型生成，不代表人工评审已通过。详细依据、调用阶段、草稿、复核与回退原因见JSON；用量缺失为None，不估算价格。"]
    report.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--samples", type=Path, default=ROOT / "docs" / "share_samples.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--persona", default="麦麦，随和好奇，熟悉的群里简短自然地聊天")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", default="")
    parser.add_argument("--persona-file", type=Path)
    asyncio.run(main(parser.parse_args()))
