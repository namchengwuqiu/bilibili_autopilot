"""麦麦 SDK v2 入口。定时任务和手动预览共用串行引擎。"""

from pathlib import Path
from typing import Any, Dict, Optional

import asyncio
import base64
import contextlib
import re
import time

import httpx

from maibot_sdk import API, Command, MaiBotPlugin

from .client import BiliClient, LoginRequired, PlatformError
from .config import AutopilotConfig
from .engine import Autopilot, ReadWaiting
from .models import (
    DecisionValidationError,
    ModelResponseError,
    ModelTimeoutError,
    VideoItem,
    local_day_start,
    parse_share_text,
    parse_support,
    polish_prompt,
    share_prompt,
    support_prompt,
)
from .sharing import SHARE_OUTCOMES, VideoSharer
from .state import StateStore
from .webui import local_time, record_row


COMMAND_PATTERN = r"^/bili_auto(?:\s+(?P<action>login|preview|replies|start|pause|status|history|logout|ack)(?:\s+(?P<bvid>BV[0-9A-Za-z]{10}))?)?\s*$"


HELP = (
    "/bili_auto login：私聊扫码登录\n/bili_auto preview：手动浏览一轮，只预览\n"
    "/bili_auto replies：检查别人回复并预览是否接话，不实际发送\n"
    "/bili_auto start：恢复运行，由麦麦自主决定刷不刷（publish_enabled 控制是否发布）\n"
    "/bili_auto pause：暂停\n/bili_auto status：状态\n/bili_auto history：最近记录\n"
    "/bili_auto logout：暂停并清除本插件登录态\n"
    "/bili_auto ack BV号：自行核对未知发送结果后解除阻塞，该视频仍不会重发"
)


class BilibiliAutopilotPlugin(MaiBotPlugin):
    config_model = AutopilotConfig

    def __init__(self) -> None:
        super().__init__()
        self.client: Optional[BiliClient] = None
        self.engine: Optional[Autopilot] = None
        self.sharer: Optional[VideoSharer] = None
        self._share_targets = "未配置群号，不分享"
        self._scheduler: Optional[asyncio.Task] = None
        self._job: Optional[asyncio.Task] = None
        self._wake = asyncio.Event()
        self._login_qr = ""
        self._job_action = ""
        self._last_operation = "尚未执行页面或聊天指令"
        self._last_operation_ok = True

    async def on_load(self) -> None:
        data_dir = Path(self.ctx.paths.data_dir)
        self.client = BiliClient(data_dir / "credential.json")
        self.engine = Autopilot(
            self.client, StateStore(data_dir / "state.sqlite3"), lambda: self.config, self._generate, self._persona
        )
        await self.engine.initialize()
        self.sharer = VideoSharer(
            self.engine.store,
            lambda: self.config,
            self.ctx.chat,
            self.ctx.send,
            self.engine.may_publish,
            self._compose_share,
        )
        self.engine.share_video = self.sharer.share
        self._share_targets = await self.sharer.targets_summary()
        try:
            await self.client.load()
        except Exception as exc:
            await self.engine.pause(f"本插件凭据读取失败：{type(exc).__name__}，请重新登录")
        self._scheduler = asyncio.create_task(self._schedule())
        self.ctx.logger.info("B站自主浏览插件已加载；publish_enabled=%s", self.config.comment.publish_enabled)

    async def on_unload(self) -> None:
        # 保留用户的启停选择；取消发布中的任务会留下 pending，重启后要求核对。
        for task in (self._scheduler, self._job):
            if task is not None:
                task.cancel()
        for task in (self._scheduler, self._job):
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        if self.client is not None:
            await self.client.close()

    async def on_config_update(self, scope: str, config_data: Dict[str, Any], version: str) -> None:
        del scope, config_data, version
        if self.sharer is not None:
            self._share_targets = await self.sharer.targets_summary()
        if self.engine is not None and not self.config.plugin.enabled:
            if not self.engine.paused or self.engine.pause_kind in {"initial", "login", "disabled"}:
                await self.engine.pause("已在配置中禁用", kind="disabled")
        self._wake.set()

    async def _persona(self) -> str:
        profile = await self.ctx.config.get("personality", {})
        if not isinstance(profile, dict):
            raise ValueError("无法读取麦麦人设配置")
        return "\n".join(str(profile.get(key, "")) for key in ("personality", "reply_style"))

    async def _generate(self, prompt: Any, cfg: AutopilotConfig) -> str:
        return await self._generate_candidates(prompt, cfg)

    async def _generate_candidates(self, prompt: Any, cfg: AutopilotConfig) -> str:
        # 沿用 Host 任务的输出额度；思考模型可能先消耗大量 token 才返回 JSON 正文。
        candidates = cfg.model.candidates or [""]
        response_error = None
        stage = "model_generation"
        if isinstance(prompt, list) and prompt:
            match = re.search(r"\[autopilot:([a-z_]+)\]", str(prompt[0].get("content", "")))
            if match:
                stage = match.group(1)
        for index, model in enumerate(candidates):
            started = time.monotonic()
            outcome, reported_usage = "failed", {}
            try:
                # 每个模型单独计时，不能在引擎外层用单模型时限截断整个切换流程。
                kwargs = {"task_name": cfg.model.task, "model_name": model} if model else {"model": cfg.model.task}
                response = await asyncio.wait_for(
                    self.ctx.llm.generate(prompt, **kwargs, timeout_ms=(cfg.model.timeout_seconds + 5) * 1000),
                    timeout=cfg.model.timeout_seconds,
                )
                if isinstance(response, dict):
                    usage = response.get("usage") if isinstance(response.get("usage"), dict) else response
                    reported_usage = {
                        key: usage[key]
                        for key in (
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                            "prompt_cache_hit_tokens",
                            "prompt_cache_miss_tokens",
                        )
                        if isinstance(usage.get(key), (int, float)) and not isinstance(usage.get(key), bool)
                    }
                if isinstance(response, dict) and response.get("success") is False:
                    # Host 的能力接口将底层异常包装为失败响应；仅识别超时，不公开异常原文。
                    if re.search(r"timeout|timed[ _-]?out|超时", str(response.get("error", "")), re.I):
                        raise ModelTimeoutError("模型调用超时")
                result = self._model_text(response)
                outcome = "success"
                return result
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            except (asyncio.TimeoutError, httpx.TimeoutException, ModelTimeoutError):
                outcome = "timeout"
                reason = "调用超时"
            except ModelResponseError as exc:
                response_error = exc
                reason = str(exc)
            except Exception as exc:
                # SDK 的 RPC 超时不是 TimeoutError，而是带 E_TIMEOUT 代码的异常。
                if getattr(exc, "code", None) != "E_TIMEOUT":
                    raise
                outcome = "timeout"
                reason = "调用超时"
            finally:
                if self.engine is not None:
                    await asyncio.to_thread(
                        self.engine.store.model_call,
                        self.client.uid if self.client is not None else "",
                        stage, time.time(), time.monotonic() - started, outcome, reported_usage,
                    )
            if index + 1 < len(candidates):
                self.ctx.logger.warning(
                    "B站模型未返回有效正文：%s（候选 %s/%s）；%s；切换下一个模型",
                    model or cfg.model.task,
                    index + 1,
                    len(candidates),
                    reason,
                )
        if response_error is not None:
            raise ModelResponseError(f"候选模型未返回有效正文：{response_error}；本次跳过，后续有限重查")
        raise ModelTimeoutError("候选模型均超时，本次跳过；自动运行保持开启，下轮继续")

    async def _compose_share(self, uid: str, item: VideoItem, cfg: AutopilotConfig) -> str:
        assert self.engine is not None
        context = await asyncio.to_thread(self.engine.store.share_context, uid, item.bvid)
        return await asyncio.wait_for(self._share_steps(item, context, cfg), timeout=cfg.share.total_timeout_seconds)

    async def _share_steps(
        self, item: VideoItem, context: Dict[str, Any], cfg: AutopilotConfig,
        *, trace: Optional[Dict[str, Any]] = None,
    ) -> str:
        # 仅独立预览工具提供 trace；运行账号不会额外保存模型原文。
        def record(**values: Any) -> None:
            if trace is not None:
                trace.update(values)

        persona = await self._persona()
        async def verify(text: str, original: str = "") -> None:
            raw = await asyncio.wait_for(
                self._generate(support_prompt(text, context.get("content_evidence", {}), draft=original), cfg),
                timeout=cfg.model.review_timeout_seconds,
            )
            review = parse_support(raw)
            record(**{"polish_review" if original else "draft_review": review})
            if not review["supported"]:
                raise DecisionValidationError("unsupported_share", review["reason"])

        draft_context = dict(context)
        draft_attempts = []
        for attempt in range(2):
            prompt = share_prompt(item, draft_context, persona=persona, extra_style=cfg.model.extra_style)
            raw = await self._generate(prompt, cfg)
            record(draft_raw=raw, result_mode="skipped")
            current_attempt: Dict[str, Any] = {"raw": raw}
            draft_attempts.append(current_attempt)
            record(draft_attempts=draft_attempts)
            try:
                draft = parse_share_text(raw)
                record(draft=draft)
                current_attempt["draft"] = draft
                await verify(draft)
            except ValueError as exc:
                feedback = {"code": getattr(exc, "code", "share_format"), "reason": str(exc)[:300]}
                current_attempt["validation_error"] = feedback
                if trace is not None and "draft_review" in trace:
                    current_attempt["review"] = trace.pop("draft_review")
                if attempt == 1:
                    raise DecisionValidationError(feedback["code"], feedback["reason"]) from None
                # 只修正文案一次，仍使用原内容依据并重新复核；不重试消息发送。
                draft_context["validation_feedback"] = feedback
                continue
            if trace is not None:
                current_attempt["review"] = trace.get("draft_review")
            break
        if not cfg.share.polish_enabled:
            record(result_mode="disabled")
            return draft
        if self.config != cfg or (self.engine is not None and not self.engine.may_publish()):
            record(result_mode="cancelled")
            return draft  # 最终发送路径仍会复核暂停和配置。
        polish_cfg = cfg.model_copy(deep=True)
        if cfg.share.polish_model.strip():
            polish_cfg.model.candidates = [cfg.share.polish_model.strip()]

        async def polish() -> str:
            raw = await self._generate(
                polish_prompt(item, context, draft, persona=persona, extra_style=cfg.model.extra_style), polish_cfg
            )
            record(polish_raw=raw)
            text = parse_share_text(raw)
            record(polish_candidate=text)
            await verify(text, draft)
            record(result_mode="polished")
            return text

        try:
            return await asyncio.wait_for(polish(), timeout=cfg.share.polish_timeout_seconds)
        except asyncio.CancelledError:
            raise
        except (ValueError, ModelResponseError, asyncio.TimeoutError) as exc:
            record(result_mode="fallback", fallback_reason=getattr(exc, "code", type(exc).__name__))
            return draft  # 仅回退已通过格式和依据复核的原稿，不发送失败输出。

    @staticmethod
    def _model_text(response: Any) -> str:
        if not isinstance(response, dict) or response.get("success") is False:
            raise ModelResponseError("模型调用失败；请检查麦麦模型服务后重新预览")
        result = response.get("response")
        if not isinstance(result, str) or not result.strip():
            if response.get("reasoning"):
                raise ModelResponseError(
                    "模型只返回了思考内容，没有评论正文；请检查模型任务的最大输出 token 上限后重新预览"
                )
            raise ModelResponseError("模型未返回评论正文；请检查模型服务和任务的最大输出 token 上限后重新预览")
        return result

    async def _schedule(self) -> None:
        try:
            while True:
                self._wake.clear()
                if self.engine is not None:
                    await asyncio.to_thread(self.engine.store.clean_evidence, self.engine.clock())
                if (
                    self.engine is not None
                    and self.engine.paused
                    and self.config.plugin.enabled
                    and self.config.browse.auto_start
                    and self.client is not None
                    and self.client.uid
                    and self.engine.pause_kind in {"initial", "login", "disabled"}
                ):
                    try:
                        await self.engine.start(automatic=True)
                    except ReadWaiting as exc:
                        self.engine.last_result = str(exc)
                    except (LoginRequired, ModelResponseError, PlatformError) as exc:
                        await self.engine.pause(str(exc))
                if self.engine is not None and not self.engine.paused and self.config.plugin.enabled:
                    if self.config.replies.enabled:
                        await self.engine.reply_round()
                    result = await self.engine.round()
                    # 评论正文和账号凭据仅写入插件私有数据，不进入公共日志。
                    self.ctx.logger.info("B站浏览轮次结束：%s", "已暂停" if self.engine.paused else "完成")
                    del result
                try:
                    delay = (
                        await self.engine.next_check_delay()
                        if self.engine is not None
                        else self.config.browse.interval_seconds
                    )
                    await asyncio.wait_for(self._wake.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.ctx.logger.error("B站调度停止：%s", type(exc).__name__)
            if self.engine is not None:
                await self.engine.pause(f"调度停止：{type(exc).__name__}，需要重载插件")

    def _authorized(self, platform: str, user_id: str, is_local_operator: bool) -> bool:
        return is_local_operator is True or (
            bool(platform and user_id) and f"{platform}:{user_id}" in self.config.plugin.admin_users
        )

    async def _respond(self, stream_id: str, text: str, success: bool = True):
        if stream_id:
            await self.ctx.send.text(text, stream_id)
        return success, text, 2

    def _queue_job(self, action: str, stream_id: str = "") -> tuple[bool, str]:
        if self._job is not None and not self._job.done():
            return False, "已有手动任务在运行，请等待结束"
        self._job_action = action
        self._last_operation = f"开始处理 {action}，结果会自动显示"
        self._last_operation_ok = True
        self._job = asyncio.create_task(self._background(action, stream_id))
        return True, self._last_operation

    def _ui_result(self, ok: bool, message: str) -> Dict[str, Any]:
        self._last_operation, self._last_operation_ok = message, ok
        return {"ok": ok, "message": message}

    @API("autopilot_overview", description="读取 B 站管理页状态，不请求视频或触发浏览", version="1")
    async def autopilot_overview(self) -> Dict[str, Any]:
        assert self.engine is not None and self.client is not None
        cfg = self.config
        uid = self.client.uid
        stats = await asyncio.to_thread(
            self.engine.store.stats, uid, local_day_start(self.engine.clock(), cfg.browse.timezone)
        )
        choice = await asyncio.to_thread(self.engine.store.latest_browse_decision, uid)
        reply_last = await asyncio.to_thread(self.engine.store.last_reply_check, uid)
        last_share = await asyncio.to_thread(self.engine.store.latest_share, uid)
        recoveries = await asyncio.to_thread(self.engine.store.recoveries, uid)
        usage = await asyncio.to_thread(
            self.engine.store.model_usage, uid, local_day_start(self.engine.clock(), cfg.browse.timezone)
        )
        busy = self._job is not None and not self._job.done()
        return {
            "account": f"{self.client.name}（UID {uid}）" if uid else "尚未登录，可在本页扫码登录",
            "state": "已暂停"
            if self.engine.paused
            else "自主运行中"
            if cfg.browse.autonomous_enabled
            else "定时运行中",
            "mode": "允许自动评论" if cfg.comment.publish_enabled else "仅预览评论",
            "candidate_pool": await self.engine.candidate_pool_status(),
            "comment_cooldown": (self.engine.comment_cooldown(stats, cfg) or "已满足发送间隔")
            if cfg.comment.publish_enabled
            else "仅预览，不受发送冷却限制",
            "views": f"{stats['views']} / {cfg.browse.videos_per_day}",
            "attempts": f"{stats['attempts']} / {cfg.comment.per_day}",
            "replies": f"{stats['reply_attempts']} / {cfg.replies.per_day}（今日跟进发送尝试）",
            "reply_analyses": f"{stats['reply_analyses']} / {cfg.replies.analyses_per_day}",
            "reply_next_check": (
                local_time(reply_last + cfg.replies.interval_seconds, cfg.browse.timezone)
                if cfg.replies.enabled and reply_last
                else "等待首次检查"
                if cfg.replies.enabled
                else "跟进回复已关闭"
            ),
            "reply_result": self.engine.last_reply_result,
            "share_targets": self._share_targets,
            "share_result": (
                f"{local_time(last_share['ts'], cfg.browse.timezone)} {SHARE_OUTCOMES.get(last_share['status'], last_share['status'])}："
                f"QQ 群 {last_share['group_id']}；{last_share['detail']}"
                if last_share
                else "暂无群分享记录；仅新的视频评论提交成功后分享"
            ),
            "unresolved": "、".join(stats["unresolved"]) or "无",
            "pause_reason": self.engine.pause_reason or "无",
            "choice": ("去浏览：" if choice["should_browse"] else "暂时不刷：") + choice["reason"]
            if choice
            else "尚未产生自主决定",
            "next_check": self.engine.next_check_status(choice, cfg),
            "check_range": f"{(cfg.browse.interval_seconds + 59) // 60}～{cfg.browse.max_check_minutes} 分钟",
            "job": f"执行中：{self._job_action}" if busy else "暂无手动任务",
            "operation": self._last_operation,
            "operation_state": "执行中" if busy else "完成" if self._last_operation_ok else "未完成，请查看结果",
            "last_round": self.engine.last_result,
            "read_recovery": "；".join(
                f"{w['stage']} 第{w['failures']}次失败，{local_time(w['retry_at'], cfg.browse.timezone)}可恢复：{w['reason']}"
                for w in recoveries
            )
            or "无读取恢复等待",
            "model_usage": f"今日 {usage['calls']} 次，累计 {usage['elapsed_seconds']} 秒；用量由 Host 返回时记录",
            "busy": busy or self.engine.lock.locked(),
            "qr": self._login_qr,
            "commands": {
                action: f"/bili_auto {action}"
                for action in ("start", "pause", "preview", "replies", "login", "status", "history")
            },
        }

    @API("autopilot_records", description="分页读取当前 B 站账号的浏览、评论或自主决定记录", version="1")
    async def autopilot_records(self, kind: str = "visits", search: str = "", page: int = 1) -> Dict[str, Any]:
        assert self.engine is not None and self.client is not None
        data = await asyncio.to_thread(self.engine.store.page_records, self.client.uid, kind, search, page)
        return {
            "rows": [record_row(row, kind, self.config.browse.timezone) for row in data["rows"]],
            "pagination": data["pagination"],
            "is_reply": kind in {"replies", "reply_attempts"},
        }

    @API("autopilot_command", description="执行 B 站插件自身的管理指令，不执行系统命令", version="1")
    async def autopilot_command(self, command: str) -> Dict[str, Any]:
        assert self.engine is not None and self.client is not None
        if not isinstance(command, str) or len(command) > 120:
            return self._ui_result(False, "指令无效，请填写 /bili_auto 指令")
        match = re.fullmatch(COMMAND_PATTERN, command.strip())
        if not match:
            return self._ui_result(False, "仅支持本插件的 /bili_auto 指令；不支持系统命令")
        action = match.group("action") or "help"
        if match.group("bvid") and action != "ack":
            return self._ui_result(False, "只有 ack 指令接受 BV 号参数")
        if action in {"logout", "ack"}:
            return self._ui_result(False, "退出登录或核对发送结果请使用页面下方对应按钮")
        if action in {"login", "preview", "replies", "start"}:
            return self._ui_result(*self._queue_job(action))
        if action == "pause":
            await self.engine.pause("管理员手动暂停", kind="manual")
            return self._ui_result(True, "已暂停，尚未发出的评论将被拦住；用恢复运行按钮继续")
        if action == "status":
            return self._ui_result(True, await self.engine.status())
        if action == "history":
            return self._ui_result(True, await asyncio.to_thread(self.engine.store.history, self.client.uid))
        return self._ui_result(True, HELP)

    @API("autopilot_logout", description="暂停并退出本插件的 B 站账号", version="1")
    async def autopilot_logout(self) -> Dict[str, Any]:
        assert self.engine is not None
        await self.engine.pause("管理员退出账号", kind="logout")
        if self._job is not None and not self._job.done():
            self._job.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._job
        return self._ui_result(*self._queue_job("logout"))

    @API("autopilot_ack", description="管理员检查 B 站后标记未知发送结果已核对，不重发评论", version="1")
    async def autopilot_ack(self, bvid: str) -> Dict[str, Any]:
        assert self.engine is not None and self.client is not None
        if not isinstance(bvid, str) or not re.fullmatch(r"BV[0-9A-Za-z]{10}", bvid.strip()):
            return self._ui_result(False, "请填写完整 BV 号，例如 BV1234567890")
        changed = await asyncio.to_thread(self.engine.store.acknowledge, self.client.uid, bvid.strip())
        return self._ui_result(
            changed, "已记为核对完成；该视频不会重发，可恢复运行" if changed else "没有找到该视频的未知发送记录"
        )

    async def _background(self, action: str, stream_id: str) -> None:
        assert self.engine is not None and self.client is not None
        try:
            if action == "preview":
                result = await self.engine.round(preview=True)
            elif action == "replies":
                result = await self.engine.reply_round(preview=True)
            elif action == "start":
                result = await self.engine.start()
                self._wake.set()
            elif action == "login":
                # 显式 pause 即使遇到重新登录也保留；账号异常可通过重新扫码恢复。
                kind = "manual" if self.engine.pause_kind == "manual" else "login"
                await self.engine.pause("正在扫码登录", kind=kind)
                async with self.engine.lock:
                    png, key = await self.client.begin_login()
                    encoded = base64.b64encode(png).decode()
                    self._login_qr = "data:image/png;base64," + encoded
                    if stream_id:
                        await self.ctx.send.image(encoded, stream_id)
                    result = await self.client.poll_login(key)
                if self.engine.pause_kind == "manual":
                    result += "；仍保持管理员手动暂停，使用 start 恢复"
                elif self.config.browse.auto_start:
                    result += "；" + await self.engine.start(automatic=True)
                    self._wake.set()
                else:
                    result += "；当前保持暂停，使用 start 启动"
            elif action == "logout":
                async with self.engine.lock:
                    await self.client.logout()
                result = "已清除本插件登录态；历史去重记录已保留"
            else:
                raise ValueError("未知后台命令")
            self._last_operation = result
            self._last_operation_ok = not result.startswith(("已暂停：", "运行异常：", "跟进回复异常："))
            await self._respond(stream_id, result)
        except asyncio.CancelledError:
            raise
        except ReadWaiting as exc:
            self._last_operation, self._last_operation_ok = str(exc), False
            await self._respond(stream_id, str(exc), False)
        except (LoginRequired, PlatformError) as exc:
            await self.engine.pause(str(exc))
            self._last_operation, self._last_operation_ok = str(exc), False
            await self._respond(stream_id, str(exc), False)
        except Exception as exc:
            message = f"{action} 失败：{type(exc).__name__}。可查看 status 或重新尝试"
            await self.engine.pause(message)
            self._last_operation, self._last_operation_ok = message, False
            self.ctx.logger.warning("B站后台操作失败：%s", type(exc).__name__)
            await self._respond(stream_id, message, False)
        finally:
            if action == "login":
                self._login_qr = ""

    @Command(
        "bili_auto",
        description="管理B站自主浏览与评论",
        pattern=COMMAND_PATTERN,
    )
    async def command(
        self,
        stream_id: str = "",
        platform: str = "",
        user_id: str = "",
        group_id: str = "",
        matched_groups: Optional[Dict[str, str]] = None,
        is_local_operator: bool = False,
        **kwargs: Any,
    ):
        del kwargs
        if not self._authorized(platform, user_id, is_local_operator):
            return await self._respond(
                stream_id, "无权使用此命令；请先在插件配置中设置 admin_users（QQ号或平台:用户号）", False
            )
        if self.engine is None or self.client is None:
            return await self._respond(stream_id, "插件尚未就绪", False)
        action = (matched_groups or {}).get("action") or "help"
        try:
            if action in {"login", "preview", "replies", "start"}:
                if not stream_id:
                    return await self._respond(stream_id, "请在麦麦聊天中执行此命令", False)
                if action == "login" and group_id:
                    return await self._respond(stream_id, "请在私聊中执行扫码登录", False)
                ok, message = self._queue_job(action, stream_id)
                return await self._respond(stream_id, message + ("；结果会发到当前聊天" if ok else ""), ok)
            if action == "pause":
                await self.engine.pause("管理员手动暂停", kind="manual")
                return await self._respond(stream_id, "已暂停，尚未发出的评论将被拦住")
            if action == "logout":
                await self.engine.pause("管理员退出账号", kind="logout")
                if self._job is not None and not self._job.done():
                    self._job.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._job
                self._job = asyncio.create_task(self._background("logout", stream_id))
                return await self._respond(stream_id, "已暂停，当前操作结束后将清除本插件登录态")
            if action == "status":
                return await self._respond(stream_id, await self.engine.status())
            if action == "history":
                result = await asyncio.to_thread(self.engine.store.history, self.client.uid)
                return await self._respond(stream_id, result)
            if action == "ack":
                bvid = (matched_groups or {}).get("bvid") or ""
                if not bvid:
                    return await self._respond(stream_id, "自行核对发送结果后，使用 /bili_auto ack BV号", False)
                changed = await asyncio.to_thread(self.engine.store.acknowledge, self.client.uid, bvid)
                return await self._respond(
                    stream_id,
                    "已记为核对完成；该视频不会重发，可使用 start 继续" if changed else "没有找到该视频的未知发送记录",
                    changed,
                )
            return await self._respond(stream_id, HELP)
        except (LoginRequired, PlatformError) as exc:
            return await self._respond(stream_id, str(exc), False)


def create_plugin() -> BilibiliAutopilotPlugin:
    return BilibiliAutopilotPlugin()
