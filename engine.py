"""串行自主浏览流程；模型仅产生结构化判断，发布控制由程序执行。"""

from dataclasses import asdict
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

import asyncio
import json
import math
import time

import httpx

from .client import (
    BiliClient,
    LoginRequired,
    PlatformError,
    PublishNotSent,
    READ_NETWORK_ERRORS,
    UnknownPublish,
    VideoReadError,
)
from .config import AutopilotConfig
from .models import (
    DecisionValidationError,
    ModelResponseError,
    ModelTimeoutError,
    VideoItem,
    active_time,
    browse_prompt,
    candidate_selection_prompt,
    decision_prompt,
    local_day_start,
    next_local_day,
    normalized,
    parse_browse_decision,
    parse_candidate_selection,
    parse_decision,
    parse_reply_decision,
    parse_support,
    reply_prompt,
    support_prompt,
)
from .state import StateStore


class ReadWaiting(Exception):
    """只读请求等待持久化的下次恢复时间，不改变人工暂停状态。"""


class Autopilot:
    def __init__(
        self,
        client: Any,
        store: StateStore,
        get_config: Callable[[], AutopilotConfig],
        generate: Any,
        get_persona: Any,
        clock: Callable[[], float] = time.time,
        share_video: Optional[Callable[[str, VideoItem], Awaitable[str]]] = None,
        review: Optional[Callable[[Any, AutopilotConfig], Awaitable[str]]] = None,
    ):
        self.client, self.store = client, store
        self.get_config, self.generate, self.get_persona = get_config, generate, get_persona
        self.clock = clock
        self.share_video = share_video
        self.review = review
        self._selected_bvids: List[str] = []
        self.lock = asyncio.Lock()
        self._control_lock = asyncio.Lock()
        self._control_revision = 0
        self.paused = True
        self.pause_reason = "首次运行"
        self.pause_kind = "initial"
        self.last_result = "尚未运行"
        self.last_reply_result = "尚未检查收到的回复"

    async def initialize(self) -> None:
        cfg = self.get_config()
        await asyncio.to_thread(
            self.store.initialize,
            self.clock(),
            cfg.browse.timezone,
            cfg.browse.backlog_max_age_hours,
            cfg.browse.backlog_max_items,
        )
        control = await asyncio.to_thread(self.store.control)
        self.paused = control["paused"] != "0"
        self.pause_reason = control["pause_reason"]
        self.pause_kind = control["pause_kind"]

    async def pause(self, reason: str, kind: str = "error") -> None:
        self._control_revision += 1
        self.paused, self.pause_reason = True, reason
        self.pause_kind = kind
        async with self._control_lock:
            await asyncio.to_thread(self.store.set_pause, True, reason, kind)

    async def start(self, automatic: bool = False) -> str:
        cfg = self.get_config().model_copy(deep=True)
        if not cfg.plugin.enabled:
            return "插件已在配置中禁用"
        if automatic and (not cfg.browse.auto_start or not self.client.uid):
            return "等待管理员启动或登录"
        if self.lock.locked():
            return "当前有任务正在处理，请等待结束后再启动"
        async with self.lock:
            if automatic and self.paused and self.pause_kind not in {"initial", "login", "disabled"}:
                return "当前为手动或异常暂停，请使用 start 恢复"
            revision = self._control_revision
            uid = await self._read("account", self.client.check_account, cfg)
            # 检查账号期间可能收到管理员 pause；自动启动不能覆盖这个更新。
            if automatic and self.paused and self.pause_kind not in {"initial", "login", "disabled"}:
                return "当前为手动或异常暂停，请使用 start 恢复"
            stats = await asyncio.to_thread(self.store.stats, uid, local_day_start(self.clock(), cfg.browse.timezone))
            if stats["unresolved"]:
                await self.pause("存在待核对的评论发送结果")
                return "存在待核对评论：" + ", ".join(stats["unresolved"]) + "；核对后使用 ack BV号"
            async with self._control_lock:
                if revision != self._control_revision or not self.get_config().plugin.enabled:
                    return "启动期间状态已变化，本次启动已取消"
                await asyncio.to_thread(self.store.set_pause, False, "")
                if revision != self._control_revision:
                    return "启动期间已收到暂停，本次启动已取消"
                self.paused, self.pause_reason = False, ""
                self.pause_kind = "running"
        if cfg.browse.autonomous_enabled:
            return "已进入自主模式，由麦麦决定这次刷不刷 B 站；" + (
                "有合适内容时允许发表评论" if cfg.comment.publish_enabled else "评论仅预览，不会发布"
            )
        return "已启动自动发布" if cfg.comment.publish_enabled else "已启动自动预览（不发布评论）"

    def next_check_time(self, choice: Dict[str, Any], cfg: AutopilotConfig) -> float:
        # 已保存的旧决定也服从当前间隔，缩短设置后无需等完原来的几小时。
        return max(
            choice["ts"] + cfg.browse.interval_seconds,
            min(choice["next_check_at"], choice["ts"] + cfg.browse.max_check_minutes * 60),
        )

    def next_check_status(self, choice: Optional[Dict[str, Any]], cfg: AutopilotConfig) -> str:
        if self.paused:
            return "已暂停，恢复后重新检查"
        if not self.active(cfg):
            return "当前不在活动时段，进入活动时段后检查"
        if not cfg.browse.autonomous_enabled:
            return "按固定间隔浏览"
        if choice is None:
            return "等待首次判断"
        when = self.next_check_time(choice, cfg)
        if when <= self.clock():
            return "已到检查时间，等待当前任务完成" if self.lock.locked() else "已到检查时间，等待调度"
        return datetime.fromtimestamp(when, ZoneInfo(cfg.browse.timezone)).strftime("%m-%d %H:%M:%S")

    async def next_check_delay(self) -> float:
        cfg = self.get_config()
        delay = cfg.browse.interval_seconds
        if not self.paused and cfg.browse.autonomous_enabled and self.active(cfg):
            choice = await asyncio.to_thread(self.store.latest_browse_decision, self.client.uid)
            if choice and self.next_check_time(choice, cfg) > self.clock():
                delay = max(1, self.next_check_time(choice, cfg) - self.clock())
        if not self.paused and cfg.replies.enabled and self.client.uid and self.active(cfg):
            last = await asyncio.to_thread(self.store.last_reply_check, self.client.uid)
            delay = min(delay, max(1, last + cfg.replies.interval_seconds - self.clock()))
        if (not self.paused and cfg.comment.publish_enabled and self.client.uid and self.active(cfg)):
            stats = await asyncio.to_thread(self.store.stats, self.client.uid,
                                           local_day_start(self.clock(), cfg.browse.timezone))
            pool = await asyncio.to_thread(self.store.candidates, self.client.uid,
                                          lambda item: not self.filter_item(item, cfg), self.clock(),
                                          cfg.browse.backlog_max_items, ready_only=False)
            if pool and not stats["unresolved"] and stats["views"] < cfg.browse.videos_per_day:
                control = await asyncio.to_thread(self.store.control)
                account_wait = await asyncio.to_thread(self.store.read_wait, self.client.uid, "account")
                due = max(min(row["retry_at"] for row in pool),
                          float(control.get("candidate_check:" + self.client.uid, "0")),
                          stats["last_attempt"] + cfg.comment.min_interval_seconds if stats["last_attempt"] else 0,
                          next_local_day(self.clock(), cfg.browse.timezone) if stats["attempts"] >= cfg.comment.per_day else 0,
                          account_wait["retry_at"] if account_wait else 0)
                delay = min(delay, max(1, due - self.clock()))
        waits = await asyncio.to_thread(self.store.recoveries, self.client.uid)
        waits = [w for w in waits if w["retry_at"] > self.clock() and (w["stage"] != "replies" or cfg.replies.enabled)]
        if waits:
            # 未到期的回复轮次不能把退避等待变成每秒空转。
            delay = min(delay, max(1, min(w["retry_at"] for w in waits) - self.clock()))
        return delay

    async def _read(self, stage: str, operation: Callable[[], Awaitable[Any]], cfg: AutopilotConfig) -> Any:
        uid = self.client.uid
        waiting = await asyncio.to_thread(self.store.read_wait, uid, stage)
        if waiting and waiting["retry_at"] > self.clock():
            when = datetime.fromtimestamp(waiting["retry_at"], ZoneInfo(cfg.browse.timezone)).strftime("%m-%d %H:%M:%S")
            raise ReadWaiting(f"读取阶段 {stage} 等待至 {when}；{waiting['reason']}")
        try:
            result = await operation()
        except asyncio.CancelledError:
            raise
        except (VideoReadError, PlatformError, *READ_NETWORK_ERRORS, httpx.HTTPStatusError) as exc:
            if isinstance(exc, PlatformError):
                retryable = exc.code == -504
            elif isinstance(exc, VideoReadError):
                retryable = exc.retryable
            elif isinstance(exc, httpx.HTTPStatusError):
                code = exc.response.status_code
                if code in {401, 403, 412, 429}:
                    raise PlatformError(stage, code) from None
                retryable = code >= 500 or code == 408
            else:
                retryable = True
            if not retryable:
                raise
            reason = (
                str(exc) if isinstance(exc, (VideoReadError, PlatformError)) else f"临时网络故障：{type(exc).__name__}"
            )
            waiting = await asyncio.to_thread(
                self.store.read_failed, uid, stage, self.clock(), reason, cfg.browse.read_retry_base_seconds
            )
            when = datetime.fromtimestamp(waiting["retry_at"], ZoneInfo(cfg.browse.timezone)).strftime("%m-%d %H:%M:%S")
            raise ReadWaiting(f"读取阶段 {stage} 第{waiting['failures']}次失败，等待至 {when}；{reason}") from None
        await asyncio.to_thread(self.store.read_succeeded, uid, stage)
        return result

    async def reply_round(self, preview: bool = False) -> str:
        if self.lock.locked():
            return "已有浏览、回复检查或登录任务在运行"
        async with self.lock:
            try:
                self.last_reply_result = await self._reply_round(preview)
            except asyncio.CancelledError:
                raise
            except ReadWaiting as exc:
                self.last_reply_result = str(exc)
            except (asyncio.TimeoutError, ModelTimeoutError):
                self.last_reply_result = "模型判断超时，本次跟进跳过；自动运行保持开启，下次检查继续"
            except (LoginRequired, ModelResponseError, PlatformError, PublishNotSent, UnknownPublish) as exc:
                await self.pause(str(exc))
                self.last_reply_result = f"已暂停：{exc}"
            except Exception as exc:
                await self.pause(f"跟进回复异常：{type(exc).__name__}；请检查配置或使用 replies 预览")
                self.last_reply_result = self.pause_reason
            return self.last_reply_result

    async def _reply_round(self, preview: bool) -> str:
        cfg = self.get_config().model_copy(deep=True)
        if not cfg.plugin.enabled or not cfg.replies.enabled:
            return "跟进回复已关闭"
        if not preview and (self.paused or not self.active(cfg)):
            return "已暂停或当前不在活动时段"
        uid = self.client.uid
        if not uid:
            raise LoginRequired("检查收到的回复前请先登录")
        last = await asyncio.to_thread(self.store.last_reply_check, uid)
        if not preview and self.clock() < last + cfg.replies.interval_seconds:
            return self.last_reply_result
        stats = await asyncio.to_thread(self.store.stats, uid, local_day_start(self.clock(), cfg.browse.timezone))
        if stats["unresolved"]:
            await self.pause("存在待核对的评论或跟进回复发送结果")
            return "待核对：" + "、".join(stats["unresolved"])
        own = await asyncio.to_thread(self.store.own_comments, uid)
        if not own:
            await asyncio.to_thread(self.store.set_reply_check, uid, self.clock())
            return "暂无本插件已提交成功的评论，收到回复后才能跟进"
        await self._read("account", self.client.check_account, cfg)
        since = self.clock() - cfg.replies.max_age_hours * 3600
        messages = await self._read("replies", lambda: self.client.incoming_replies(own, since), cfg)
        if self.get_config() != cfg or (not preview and (self.paused or not self.active(cfg))):
            return "读取回复期间已暂停或配置、活动时段变化"
        await asyncio.to_thread(self.store.receive_replies, uid, messages)
        await asyncio.to_thread(self.store.set_reply_check, uid, self.clock())
        mode = "preview" if preview or not cfg.comment.publish_enabled else "publish"
        pending = await asyncio.to_thread(self.store.pending_replies, uid, mode, since, self.clock())
        lines: List[str] = []
        for message in pending:
            if not self.get_config().plugin.enabled or (not preview and (self.paused or not self.active(cfg))):
                break
            stats = await asyncio.to_thread(self.store.stats, uid, local_day_start(self.clock(), cfg.browse.timezone))
            if stats["reply_analyses"] >= cfg.replies.analyses_per_day:
                lines.append("今日收到回复分析额度已满，待处理消息已保留")
                break
            if mode == "publish" and (
                stats["reply_attempts"] >= cfg.replies.per_day
                or (stats["last_attempt"] and self.clock() - stats["last_attempt"] < cfg.replies.min_interval_seconds)
            ):
                lines.append("跟进回复发送额度已满或仍在发送间隔中，待处理消息已保留")
                break
            count = await asyncio.to_thread(self.store.reply_thread_count, uid, message.root_rpid)
            if count >= cfg.replies.per_thread:
                await asyncio.to_thread(
                    self.store.decide_reply,
                    uid,
                    message.source_rpid,
                    mode,
                    self.clock(),
                    "skipped",
                    "",
                    "该讨论已达到跟进次数上限",
                )
                continue
            if any(
                word.strip() and word.casefold() in (message.text + message.root_text).casefold()
                for word in cfg.browse.blocked_keywords
            ):
                await asyncio.to_thread(
                    self.store.decide_reply,
                    uid,
                    message.source_rpid,
                    mode,
                    self.clock(),
                    "skipped",
                    "",
                    "讨论命中屏蔽关键词",
                )
                continue
            history = await asyncio.to_thread(self.store.conversation, uid, message.root_rpid)
            content = await asyncio.to_thread(self.store.content_context, uid, message.bvid, self.clock())
            prompt = reply_prompt(
                message,
                persona=await self.get_persona(),
                extra_style=cfg.model.extra_style,
                max_chars=cfg.comment.max_chars,
                history=history,
            )
            prompt[-1]["content"] = json.dumps(
                {**json.loads(prompt[-1]["content"]), "video_content": content}, ensure_ascii=False
            )
            await asyncio.to_thread(self.store.decide_reply, uid, message.source_rpid, mode, self.clock(), "reading")
            try:
                decision = await self._model_decision(
                    prompt,
                    cfg,
                    lambda raw: parse_reply_decision(raw, cfg.comment.max_chars),
                    preview,
                    lambda d: d.should_reply,
                    lambda d: self._check_support(d.reply, content, cfg),
                )
            except DecisionValidationError as exc:
                reason = f"跟进回复校验失败 [{exc.code}]：{exc}"
                await asyncio.to_thread(
                    self.store.decide_reply,
                    uid,
                    message.source_rpid,
                    mode,
                    self.clock(),
                    "invalid_decision",
                    "",
                    reason,
                )
                lines.append(f"{message.bvid} 收到的回复 {message.source_rpid}：{reason}，等待有限重查")
                continue
            except (asyncio.TimeoutError, ModelResponseError) as exc:
                reason = str(exc) if isinstance(exc, ModelResponseError) else "模型判断超时"
                outcome = (
                    "model_timeout" if isinstance(exc, (asyncio.TimeoutError, ModelTimeoutError)) else "model_error"
                )
                await asyncio.to_thread(
                    self.store.decide_reply, uid, message.source_rpid, mode, self.clock(), outcome, "", reason
                )
                lines.append(f"{message.bvid} 跟进跳过：{reason}；下次检查继续，最多重查两次")
                continue
            except Exception as exc:
                await asyncio.to_thread(
                    self.store.decide_reply,
                    uid,
                    message.source_rpid,
                    mode,
                    self.clock(),
                    "model_error",
                    "",
                    f"模型调用失败：{type(exc).__name__}",
                )
                raise
            outcome = "would_reply" if decision.should_reply else "skipped"
            await asyncio.to_thread(
                self.store.decide_reply,
                uid,
                message.source_rpid,
                mode,
                self.clock(),
                outcome,
                decision.reply,
                decision.reason,
            )
            if not decision.should_reply:
                lines.append(f"{message.bvid} 不接话：{decision.reason}")
                continue
            if mode == "preview":
                lines.append(f"{message.bvid} 预览跟进回复：{decision.reply}\n理由：{decision.reason}")
                continue
            if not self.may_publish() or self.get_config() != cfg:
                await asyncio.to_thread(
                    self.store.decide_reply,
                    uid,
                    message.source_rpid,
                    mode,
                    self.clock(),
                    "cancelled",
                    decision.reply,
                    "生成期间已暂停或配置发生变化",
                )
                break
            now = self.clock()
            reserved = await asyncio.to_thread(
                self.store.reserve_reply,
                uid,
                message,
                decision.reply,
                now,
                local_day_start(now, cfg.browse.timezone),
                cfg.replies.per_day,
                cfg.replies.min_interval_seconds,
                cfg.replies.per_thread,
                cfg.comment.similarity_threshold,
            )
            if reserved != "reserved":
                await asyncio.to_thread(
                    self.store.decide_reply,
                    uid,
                    message.source_rpid,
                    mode,
                    self.clock(),
                    "blocked",
                    decision.reply,
                    reserved,
                )
                lines.append(f"{message.bvid} 跟进回复未发送：{reserved}")
                break
            try:
                rpid = await self.client.publish_reply(
                    message,
                    decision.reply,
                    lambda: self.may_publish() and self.get_config() == cfg and self.get_config().replies.enabled,
                )
            except asyncio.CancelledError:
                raise
            except PlatformError:
                await asyncio.to_thread(self.store.finish_reply, uid, message.source_rpid, "rejected")
                await asyncio.to_thread(
                    self.store.decide_reply,
                    uid,
                    message.source_rpid,
                    mode,
                    self.clock(),
                    "rejected",
                    decision.reply,
                    "平台拒绝跟进回复",
                )
                raise
            except (LoginRequired, PublishNotSent):
                await asyncio.to_thread(self.store.finish_reply, uid, message.source_rpid, "cancelled")
                raise
            except Exception:
                await asyncio.to_thread(self.store.finish_reply, uid, message.source_rpid, "unknown")
                await asyncio.to_thread(
                    self.store.decide_reply,
                    uid,
                    message.source_rpid,
                    mode,
                    self.clock(),
                    "unknown",
                    decision.reply,
                    "发送结果无法确认，等待人工核对",
                )
                raise UnknownPublish(f"{message.bvid} 的跟进回复结果待核对，不会自动重试") from None
            await asyncio.to_thread(self.store.finish_reply, uid, message.source_rpid, "sent", rpid)
            await asyncio.to_thread(
                self.store.decide_reply,
                uid,
                message.source_rpid,
                mode,
                self.clock(),
                "sent",
                decision.reply,
                decision.reason,
            )
            lines.append(f"{message.bvid} 已提交跟进回复（rpid={rpid}）：{decision.reply}")
            break
        return "\n".join(lines) or "本次没有需要处理的新回复"

    async def _want_to_browse(
        self, uid: str, cfg: AutopilotConfig, stats: Dict[str, Any], persona: str, candidates: List[VideoItem]
    ) -> bool:
        choice = await asyncio.to_thread(self.store.latest_browse_decision, uid)
        if choice and self.next_check_time(choice, cfg) > self.clock():
            self.last_result = "尚未到下次自主浏览检查时间"
            return False
        recent = await asyncio.to_thread(self.store.browse_context, uid)
        min_minutes = math.ceil(cfg.browse.interval_seconds / 60)
        prompt = browse_prompt(
            persona=persona,
            interests=cfg.browse.interests,
            now=self.clock(),
            timezone=cfg.browse.timezone,
            min_minutes=min_minutes,
            max_minutes=cfg.browse.max_check_minutes,
            stats=stats,
            recent=recent,
            candidates=candidates,
            explore_more=cfg.browse.explore_more,
        )
        try:
            raw = await self._generate_model(prompt, cfg)
        except (asyncio.TimeoutError, ModelTimeoutError):
            now = self.clock()
            await asyncio.to_thread(
                self.store.record_browse_decision,
                uid,
                now,
                False,
                "自主浏览判断超时，下次检查继续",
                now + cfg.browse.interval_seconds,
            )
            raise
        except ModelResponseError as exc:
            now = self.clock()
            reason = f"自主浏览模型调用失败：{exc}；下次检查继续"
            await asyncio.to_thread(
                self.store.record_browse_decision, uid, now, False, reason, now + cfg.browse.interval_seconds
            )
            self.last_result = reason
            return False
        try:
            decision = parse_browse_decision(
                raw, min_minutes, cfg.browse.max_check_minutes, candidates, cfg.browse.videos_per_round
            )
        except (ValueError, TypeError):
            # 格式不完整时不擅自去浏览；下次检查时重新判断，不自动修补模型原文。
            now = self.clock()
            reason = "自主浏览判断格式无效，本次不浏览，下次检查时重试"
            await asyncio.to_thread(
                self.store.record_browse_decision, uid, now, False, reason, now + cfg.browse.interval_seconds
            )
            self.last_result = reason
            return False
        now = self.clock()
        await asyncio.to_thread(
            self.store.record_browse_decision,
            uid,
            now,
            decision.should_browse,
            decision.reason,
            now + decision.next_check_minutes * 60,
            decision.selected_bvids,
        )
        self._selected_bvids = decision.selected_bvids
        self.last_result = ("决定去浏览：" if decision.should_browse else "这次暂时不刷：") + decision.reason
        return decision.should_browse

    def active(self, cfg: AutopilotConfig) -> bool:
        return active_time(self.clock(), cfg.browse.timezone, cfg.browse.active_start_hour, cfg.browse.active_end_hour)

    def filter_item(self, item: VideoItem, cfg: AutopilotConfig) -> str:
        if item.up_uid == self.client.uid:
            return "跳过本账号的视频"
        if item.up_uid in cfg.browse.blocked_up_uids:
            return "UP 主在屏蔽列表"
        if item.duration <= 0 or item.duration > cfg.browse.max_duration_seconds:
            return "视频时长不合适"
        text = (item.title + "\n" + item.description).casefold()
        if any(word.strip() and word.casefold() in text for word in cfg.browse.blocked_keywords):
            return "命中屏蔽关键词"
        return ""

    def may_publish(self) -> bool:
        cfg = self.get_config().model_copy(deep=True)
        return cfg.plugin.enabled and cfg.comment.publish_enabled and not self.paused and self.active(cfg)

    async def record(
        self, uid: str, item: VideoItem, mode: str, outcome: str, detail: str, retryable: Optional[bool] = None
    ) -> None:
        # 纯文本状态也保留标题，避免管理页只剩 BV 号。
        try:
            fields = json.loads(detail)
        except ValueError:
            fields = {"reason": detail}
        if not isinstance(fields, dict):
            fields = {"reason": detail}
        fields.setdefault("title", item.title)
        fields.setdefault(
            "video",
            {
                "bvid": item.bvid,
                "aid": item.aid,
                "title": item.title,
                "up_uid": item.up_uid,
                "up_name": item.up_name,
                "duration": item.duration,
            },
        )
        detail = json.dumps(fields, ensure_ascii=False)
        await asyncio.to_thread(self.store.visit, uid, item.bvid, mode, self.clock(), outcome, detail, retryable)

    def comment_cooldown(self, stats: Dict[str, Any], cfg: AutopilotConfig) -> str:
        until = stats["last_attempt"] + cfg.comment.min_interval_seconds
        if not stats["last_attempt"] or self.clock() >= until:
            return ""
        when = datetime.fromtimestamp(until, ZoneInfo(cfg.browse.timezone)).strftime("%m-%d %H:%M:%S")
        return f"评论冷却中，最早 {when} 可再次评论；不影响自主浏览"

    async def candidate_pool_status(self) -> str:
        cfg = self.get_config()
        rows = await asyncio.to_thread(self.store.candidates, self.client.uid,
                                       lambda item: not self.filter_item(item, cfg), self.clock(),
                                       cfg.browse.backlog_max_items, ready_only=False)
        return f"{len(rows)} / {cfg.browse.backlog_max_items}；可发送时独立选片，可选择不评论；候选不保证发送"

    async def round(self, preview: bool = False) -> str:
        if self.lock.locked():
            return "已有浏览或登录任务在运行"
        async with self.lock:
            try:
                self.last_result = await self._round(preview)
            except asyncio.CancelledError:
                raise
            except ReadWaiting as exc:
                self.last_result = str(exc)
            except (asyncio.TimeoutError, ModelTimeoutError):
                self.last_result = "模型判断超时，本轮跳过；自动运行保持开启，下轮继续"
            except (LoginRequired, ModelResponseError, PlatformError, PublishNotSent, UnknownPublish) as exc:
                await self.pause(str(exc))
                self.last_result = f"已暂停：{exc}"
            except Exception as exc:
                # 不记录可能包含 Cookie、字幕签名 URL 或请求头的异常全文。
                await self.pause(f"运行异常：{type(exc).__name__}；请检查配置或在预览模式重试")
                self.last_result = self.pause_reason
            return self.last_result

    async def _generate_model(self, prompt: Any, cfg: AutopilotConfig) -> str:
        # 单模型时限由生成器执行；外层只限制整组候选的总等待时间。
        timeout = cfg.model.timeout_seconds * max(1, len(cfg.model.candidates)) + 5
        return await asyncio.wait_for(self.generate(prompt, cfg), timeout=timeout)

    async def _check_support(self, text: str, evidence: Dict[str, Any], cfg: AutopilotConfig) -> Dict[str, Any]:
        prompt = support_prompt(text, evidence)
        try:
            if self.review is None:
                raw = await asyncio.wait_for(
                    self._generate_model(prompt, cfg), timeout=cfg.model.review_timeout_seconds
                )
            else:
                raw = await asyncio.wait_for(self.review(prompt, cfg), timeout=cfg.model.review_timeout_seconds)
        except (asyncio.TimeoutError, ModelTimeoutError):
            raise ModelTimeoutError("正文依据复核超时，未产生发送意图") from None
        result = parse_support(raw)
        if not result["supported"]:
            raise DecisionValidationError("unsupported_body", result["reason"])
        return result

    async def _model_decision(
        self,
        prompt: Any,
        cfg: AutopilotConfig,
        parse: Callable[[str], Any],
        preview: bool,
        should_review: Optional[Callable[[Any], bool]] = None,
        check: Optional[Callable[[Any], Awaitable[Any]]] = None,
    ) -> Any:
        return await asyncio.wait_for(
            self._decision_steps(prompt, cfg, parse, preview, should_review, check),
            timeout=cfg.model.decision_budget_seconds,
        )

    async def _decision_steps(
        self,
        prompt: Any,
        cfg: AutopilotConfig,
        parse: Callable[[str], Any],
        preview: bool,
        should_review: Optional[Callable[[Any], bool]],
        check: Optional[Callable[[Any], Awaitable[Any]]],
    ) -> Any:
        for attempt in range(2):
            raw = await self._generate_model(prompt, cfg)
            try:
                decision = parse(raw)
                if should_review is not None and check is not None and should_review(decision):
                    if self.get_config() != cfg or (not preview and (self.paused or not self.active(cfg))):
                        raise DecisionValidationError("cancelled", "正文复核前已暂停或配置发生变化")
                    await check(decision)
                return decision
            except DecisionValidationError as exc:
                if (
                    attempt
                    or self.get_config() != cfg
                    or not cfg.plugin.enabled
                    or (not preview and (self.paused or not self.active(cfg)))
                ):
                    raise
                # 重新生成一次完整判断，仅告知程序校验原因；不自动修补原文或降低校验要求。
                prompt = [
                    *prompt,
                    {
                        "role": "user",
                        "content": f"上次输出未通过程序校验 [{exc.code}]：{exc}。请重新依据上面的资料生成完整 JSON，"
                        "遵守原有字段与类型要求。内容引用必须来自已提供的摘要或字幕，"
                        "不确定或没有有效依据时选择不评论；只输出 JSON。",
                    },
                ]
        raise AssertionError("模型判断重试次数异常")

    async def _round(self, preview: bool) -> str:
        cfg = self.get_config().model_copy(deep=True)
        if not cfg.plugin.enabled:
            return "插件已禁用"
        if not preview and (self.paused or not self.active(cfg)):
            return "已暂停或当前不在活动时段"
        uid = await self._read("account", self.client.check_account, cfg)
        day_start = local_day_start(self.clock(), cfg.browse.timezone)
        stats = await asyncio.to_thread(self.store.stats, uid, day_start)
        if stats["unresolved"]:
            await self.pause("存在待核对的评论发送结果")
            return "待核对评论：" + ", ".join(stats["unresolved"])
        mode = "publish" if cfg.comment.publish_enabled and not preview else "preview"
        cooling = mode == "publish" and bool(self.comment_cooldown(stats, cfg))
        quota_full = mode == "publish" and stats["attempts"] >= cfg.comment.per_day
        until = max(
            next_local_day(self.clock(), cfg.browse.timezone) if quota_full else 0,
            stats["last_attempt"] + cfg.comment.min_interval_seconds if cooling else 0,
        )
        if until:
            await asyncio.to_thread(self.store.postpone_waits, uid, until)
        remaining = min(cfg.browse.videos_per_round, cfg.browse.videos_per_day - stats["views"])
        if remaining <= 0:
            return "今日视频分析额度已用完"
        persona = await self.get_persona()

        # 冷却仅限制发送；保留的视频等到可发送时再分析，期间继续探索新视频。
        def eligible(item: VideoItem) -> bool:
            return not self.filter_item(item, cfg)

        pool = (await asyncio.to_thread(self.store.candidates, uid, eligible, self.clock(),
                                        cfg.browse.backlog_max_items) if mode == "publish" else [])
        control = await asyncio.to_thread(self.store.control)
        comment_only = bool(pool and not until and float(control.get("candidate_check:" + uid, "0")) <= self.clock())
        items = None
        if comment_only:
            shortlist = pool[:10]
            next_check = self.clock() + cfg.browse.interval_seconds
            # 先保存下一检查时间，取消、超时与重启都不能使选择模型密集重跑。
            await asyncio.to_thread(self.store.set_meta, "candidate_check:" + uid, str(next_check))
            try:
                raw = await asyncio.wait_for(
                    self._generate_model(candidate_selection_prompt(shortlist, persona, self.clock()), cfg),
                    timeout=cfg.model.decision_budget_seconds,
                )
                selected, selection_reason = parse_candidate_selection(raw, shortlist)
            except (ValueError, ModelResponseError, asyncio.TimeoutError) as exc:
                return f"候选选择未完成：{type(exc).__name__}；下次检查再考虑，未尝试发送"
            if self.get_config() != cfg or not self.may_publish():
                return "候选选择期间已暂停或配置、活动时段变化，未尝试发送"
            await asyncio.to_thread(self.store.record_candidate_selection, uid, shortlist, selected,
                                    selection_reason, self.clock(), next_check)
            if not selected:
                return "本次未选中候选，继续保留：" + selection_reason
            items = [row["item"] for row in shortlist if row["bvid"] == selected]
            remaining = 1
        retries = await asyncio.to_thread(self.store.retry_videos, uid, mode, self.clock(), eligible)
        saved = list({item.bvid: item for item in retries}.values())
        self._selected_bvids = []
        if not comment_only and not preview and cfg.browse.autonomous_enabled:
            choice = await asyncio.to_thread(self.store.latest_browse_decision, uid)
            if choice and self.next_check_time(choice, cfg) > self.clock():
                return "尚未到下次自主浏览检查时间"
            # 到期后只先读取推荐元数据。决定去刷之前不读取字幕、调用内容模型或发送评论。
            items = await self._read("recommendations", self.client.recommended, cfg)
            items = saved + [item for item in items if item.bvid not in {v.bvid for v in saved}]
            candidates = []
            for item in items:
                if not self.filter_item(item, cfg) and not await asyncio.to_thread(
                    self.store.seen, uid, item.bvid, mode, include_deferred=not comment_only, now=self.clock()
                ):
                    candidates.append(item)
            # 候选评论独立处理；浏览窗口分给失败重查与新推荐。
            lanes = [
                [
                    v
                    for v in candidates
                    if v.bvid in {x.bvid for x in retries}
                ],
                [v for v in candidates if v.bvid not in {x.bvid for x in saved}],
            ]
            present = [lane for lane in lanes if lane]
            turn_key = "candidate_turn:" + uid
            turn = int((await asyncio.to_thread(self.store.control)).get(turn_key, "0"))
            if present:
                shift = turn % len(present)
                present = present[shift:] + present[:shift]
                candidates = []
                while present and len(candidates) < 10:
                    for lane in present:
                        if lane and len(candidates) < 10:
                            candidates.append(lane.pop(0))
                    present = [lane for lane in present if lane]
                if remaining == 1:
                    # 单项轮次轮流开放一个来源，仍由模型决定是否有兴趣。
                    candidates = candidates[:1]
                await asyncio.to_thread(self.store.set_meta, turn_key, str(turn + 1))
            if self.paused or not self.get_config().plugin.enabled or self.get_config() != cfg or not self.active(cfg):
                return "读取推荐期间已暂停或配置、活动时段变化，本轮不浏览"
            if not await self._want_to_browse(uid, cfg, stats, persona, candidates):
                return self.last_result
            by_id = {item.bvid: item for item in candidates}
            items = [by_id[bvid] for bvid in self._selected_bvids] if self._selected_bvids else candidates
            if self.paused or not self.get_config().plugin.enabled or self.get_config() != cfg or not self.active(cfg):
                return "自主判断期间已暂停或配置、活动时段变化，本轮不浏览"
        if items is None:
            items = await self._read("recommendations", self.client.recommended, cfg)
            items = saved + [item for item in items if item.bvid not in {v.bvid for v in saved}]
        # 有积压时保留新内容机会。单视频轮次交替，其余轮次最多给积压一半位置。
        chosen = list(items)
        saved_ids = {v.bvid for v in saved}
        old = [v for v in chosen if v.bvid in saved_ids]
        fresh = [v for v in chosen if v.bvid not in saved_ids]
        if old and fresh and not self._selected_bvids:
            slots = max(1, remaining // 2)
            if remaining == 1:
                turn_key = "queue_turn:" + uid
                turn = (await asyncio.to_thread(self.store.control)).get(turn_key, "new")
                chosen = old + fresh if turn == "saved" else fresh + old
                await asyncio.to_thread(self.store.set_meta, turn_key, "new" if turn == "saved" else "saved")
            else:
                chosen = old[:slots] + fresh + old[slots:]
        items = chosen
        recent = await asyncio.to_thread(self.store.recent, uid)
        lines: List[str] = []
        commented_item = None
        up_counts: Dict[str, int] = {}
        topic_counts: Dict[str, int] = {}
        for item in items:
            if remaining <= 0:
                break
            if (
                self.get_config() != cfg
                or not self.get_config().plugin.enabled
                or (not preview and (self.paused or not self.active(cfg)))
            ):
                break
            if self.filter_item(item, cfg) or await asyncio.to_thread(
                self.store.seen, uid, item.bvid, mode, include_deferred=not comment_only, now=self.clock()
            ):
                continue
            if up_counts.get(item.up_uid, 0) >= cfg.browse.max_per_up:
                continue
            if item.topic and topic_counts.get(item.topic, 0) >= cfg.browse.max_per_topic:
                continue
            up_counts[item.up_uid] = up_counts.get(item.up_uid, 0) + 1
            if item.topic:
                topic_counts[item.topic] = topic_counts.get(item.topic, 0) + 1
            remaining -= 1
            await self.record(uid, item, mode, "reading", "正在读取视频内容")
            try:
                item, evidence = await self.client.understand(item, cfg.model.content_max_chars)
            except PlatformError as exc:
                await self.record(uid, item, mode, "read_error", str(exc), retryable=exc.code == -504)
                if exc.code == -504:
                    lines.append(f"{item.bvid} 跳过：读取视频内容返回 code=-504，继续处理其他视频")
                    continue
                raise
            except VideoReadError as exc:
                await self.record(uid, item, mode, "read_error", str(exc), retryable=exc.retryable)
                lines.append(f"{item.bvid} 已跳过：{exc}")
                continue
            except (asyncio.TimeoutError, httpx.TimeoutException, httpx.NetworkError) as exc:
                reason = f"读取视频内容超时或网络连接失败：{type(exc).__name__}；已跳过该视频"
                await self.record(uid, item, mode, "read_error", reason)
                lines.append(f"{item.bvid} 跳过：{reason}")
                continue
            except Exception as exc:
                await self.record(uid, item, mode, "read_error", f"读取内容失败：{type(exc).__name__}", retryable=False)
                raise
            reason = self.filter_item(item, self.get_config())
            if self.get_config() != cfg or (not preview and (self.paused or not self.active(cfg))):
                await self.record(uid, item, mode, "cancelled", "读取期间已暂停或配置、活动时段变化")
                break
            if not reason and any(
                word.strip() and word.casefold() in evidence.text.casefold() for word in cfg.browse.blocked_keywords
            ):
                reason = "视频内容命中屏蔽关键词"
            if reason or len(evidence.text.strip()) < 30:
                reason = reason or "缺少足够的摘要或字幕（多 P 视频暂时跳过）"
                await self.record(uid, item, mode, "skipped", reason)
                lines.append(f"{item.bvid} 跳过：{reason}")
                continue
            latest = await asyncio.to_thread(self.store.stats, uid,
                                            local_day_start(self.clock(), cfg.browse.timezone))
            candidate_only = mode == "publish" and (
                bool(self.comment_cooldown(latest, cfg)) or latest["attempts"] >= cfg.comment.per_day
            )
            prompt = decision_prompt(
                item,
                evidence,
                persona=persona,
                interests=cfg.browse.interests,
                extra_style=cfg.model.extra_style,
                max_chars=cfg.comment.max_chars,
                recent=recent,
                candidate_only=candidate_only,
            )
            try:

                async def review_comment(d: Any) -> Any:
                    nonlocal evidence
                    if isinstance(self.client, BiliClient) and evidence.cid > 0 and "摘要" in evidence.source:
                        evidence = await self.client.enrich(
                            item, evidence, d.comment + " " + d.evidence_quote, cfg.model.content_max_chars
                        )
                        prompt[-1]["content"] = json.dumps(
                            {
                                "source": evidence.source,
                                "video_content": evidence.text,
                                "coverage": evidence.coverage,
                                "segments": evidence.segments,
                            },
                            ensure_ascii=False,
                        )
                    if any(
                        word.strip() and word.casefold() in evidence.text.casefold()
                        for word in cfg.browse.blocked_keywords
                    ):
                        raise DecisionValidationError("blocked_content", "补充字幕命中屏蔽关键词")
                    await self._check_support(
                        d.comment,
                        {
                            "source": evidence.source,
                            "text": evidence.text,
                            "segments": evidence.segments,
                            "coverage": evidence.coverage,
                        },
                        cfg,
                    )
                    # 再验证引用与补充内容一致；原摘要已被完整替换时，不能沿用不存在的引用。
                    parse_decision(
                        json.dumps(
                            {
                                "should_comment": d.should_comment,
                                "interest": d.interest,
                                "comment": d.comment,
                                "reason": d.reason,
                                "evidence_quote": d.evidence_quote,
                            }
                        ),
                        evidence,
                        cfg.comment.max_chars,
                    )

                decision = await self._model_decision(
                    prompt,
                    cfg,
                    lambda raw: parse_decision(raw, evidence, cfg.comment.max_chars, candidate_only=candidate_only),
                    preview,
                    lambda d: not candidate_only and d.should_comment and d.interest >= cfg.comment.min_interest,
                    review_comment,
                )
            except DecisionValidationError as exc:
                reason = f"模型输出校验失败 [{exc.code}]：{exc}"
                await self.record(
                    uid,
                    item,
                    mode,
                    "invalid_decision",
                    json.dumps(
                        {
                            "reason": reason,
                            "validation_error": exc.code,
                            "source": evidence.source,
                            "read_warning": evidence.read_warning,
                        },
                        ensure_ascii=False,
                    ),
                )
                lines.append(f"{item.bvid} 跳过：{reason}；等待有限重查")
                continue
            except (asyncio.TimeoutError, ModelTimeoutError) as exc:
                reason = str(exc) if isinstance(exc, ModelTimeoutError) else "评论生成与依据复核总时限已到"
                await self.record(uid, item, mode, "model_timeout", reason)
                lines.append(f"{item.bvid} 模型判断超时：{reason}；本轮跳过，下轮继续，最多重查两次")
                continue
            except VideoReadError as exc:
                await self.record(uid, item, mode, "read_error", f"补充字幕：{exc}", retryable=exc.retryable)
                lines.append(f"{item.bvid} 跳过：补充字幕失败，未产生发送意图")
                continue
            except PlatformError as exc:
                await self.record(uid, item, mode, "read_error", f"补充字幕：{exc}", retryable=exc.code == -504)
                if exc.code == -504:
                    lines.append(f"{item.bvid} 跳过：补充字幕临时失败，未产生发送意图")
                    continue
                raise
            except ModelResponseError as exc:
                await self.record(uid, item, mode, "model_error", str(exc))
                lines.append(f"{item.bvid} 跳过：{exc}；后台保持运行，等待有限重查")
                continue
            except Exception as exc:
                await self.record(uid, item, mode, "model_error", f"模型调用失败：{type(exc).__name__}")
                raise
            detail = json.dumps(
                {
                    "title": item.title,
                    "source": evidence.source,
                    "reason": decision.reason,
                    "interest": decision.interest,
                    "comment": decision.comment,
                    "evidence_quote": decision.evidence_quote,
                    "read_warning": evidence.read_warning,
                },
                ensure_ascii=False,
            )
            if decision.should_comment and decision.interest >= cfg.comment.min_interest:
                quote = decision.evidence_quote[:500]
                canonical_quote = normalized(quote)
                relevant = [r for r in evidence.segments if normalized(r["text"]) and
                            (canonical_quote in normalized(r["text"]) or normalized(r["text"]) in canonical_quote)]
                segments = [{**r, "text": r["text"][:200]} for r in relevant[:3]]
                await asyncio.to_thread(
                    self.store.cache_evidence,
                    uid,
                    item.bvid,
                    {"quote": quote, "segments": segments, "source": evidence.source, "coverage": evidence.coverage},
                    self.clock(),
                    cfg.model.evidence_retention_days,
                )
            if not decision.should_comment or decision.interest < cfg.comment.min_interest:
                await self.record(uid, item, mode, "skipped", detail)
                lines.append(f"{item.bvid} 看过不评论：{decision.reason}")
                continue
            if mode == "preview":
                await self.record(uid, item, mode, "would_comment", detail)
                lines.append(f"{item.bvid} 预览评论：{decision.comment}\n理由：{decision.reason}")
                continue
            if candidate_only:
                if self.get_config() != cfg or not self.may_publish():
                    await self.record(uid, item, mode, "cancelled", "候选判断期间已暂停或配置、活动时段变化")
                    break
                latest = await asyncio.to_thread(self.store.stats, uid,
                                                local_day_start(self.clock(), cfg.browse.timezone))
                quota_full = latest["attempts"] >= cfg.comment.per_day
                until = max(next_local_day(self.clock(), cfg.browse.timezone) if quota_full else 0,
                            latest["last_attempt"] + cfg.comment.min_interval_seconds)
                fields = json.loads(detail)
                fields.update(video=asdict(item),
                              decision_reason=decision.reason, comment="", candidate_only=True,
                              reason="daily_limit" if quota_full else self.comment_cooldown(latest, cfg))
                await asyncio.to_thread(self.store.wait_video, uid, item, self.clock(),
                                        json.dumps(fields, ensure_ascii=False), until,
                                        cfg.browse.backlog_max_age_hours, cfg.browse.backlog_max_items,
                                        "quota_wait" if quota_full else "deferred")
                lines.append(f"{item.bvid} 候选保留中：{fields['reason']}；尚未生成评论正文")
                continue
            # 所有长耗时操作结束后重新读取配置，暂停/关闭发布会在这里生效。
            if not self.may_publish():
                await self.record(uid, item, mode, "cancelled", "运行中已暂停、关闭发布或离开活动时段")
                break
            current = self.get_config()
            if current != cfg:
                await self.record(uid, item, mode, "cancelled", "生成评论期间配置已变化，下轮重新判断")
                break
            if (
                self.filter_item(item, current)
                or len(decision.comment) > current.comment.max_chars
                or decision.interest < current.comment.min_interest
            ):
                await self.record(uid, item, mode, "cancelled", "运行中筛选配置发生变化")
                continue
            now = self.clock()
            reserved = await asyncio.to_thread(
                self.store.reserve,
                uid,
                item.bvid,
                decision.comment,
                now,
                local_day_start(now, current.browse.timezone),
                current.comment.per_day,
                current.comment.min_interval_seconds,
                current.comment.similarity_threshold,
            )
            if reserved != "reserved":
                fields = json.loads(detail)
                fields.update(block_reason=reserved, decision_reason=fields["reason"], reason=reserved)
                outcome = "blocked"
                if reserved in {"cooldown", "daily_limit"}:
                    outcome = "deferred" if reserved == "cooldown" else "quota_wait"
                    fields["video"] = {
                        "bvid": item.bvid,
                        "aid": item.aid,
                        "title": item.title,
                        "up_uid": item.up_uid,
                        "up_name": item.up_name,
                        "duration": item.duration,
                    }
                    latest = await asyncio.to_thread(self.store.stats, uid, day_start)
                    retry_at = max(
                        next_local_day(now, current.browse.timezone)
                        if latest["attempts"] >= current.comment.per_day
                        else 0,
                        latest["last_attempt"] + current.comment.min_interval_seconds,
                    )
                    fields["reason"] = (
                        "daily_limit" if reserved == "daily_limit" else self.comment_cooldown(latest, current)
                    )
                    await asyncio.to_thread(
                        self.store.wait_video,
                        uid,
                        item,
                        now,
                        json.dumps(fields, ensure_ascii=False),
                        retry_at,
                        current.browse.backlog_max_age_hours,
                        current.browse.backlog_max_items,
                        outcome,
                    )
                else:
                    await self.record(uid, item, mode, outcome, json.dumps(fields, ensure_ascii=False))
                lines.append(f"{item.bvid} 未发布：{fields['reason']}")
                if reserved in {"cooldown", "daily_limit"}:
                    cooling = True
                    continue
                break
            if not self.may_publish():
                await asyncio.to_thread(self.store.finish, uid, item.bvid, "cancelled")
                await self.record(uid, item, mode, "cancelled", "发布前已暂停")
                break
            try:
                rpid = await self.client.publish(
                    item, decision.comment, lambda: self.may_publish() and self.get_config() == cfg
                )
            except asyncio.CancelledError:
                # pending 已经持久化，重新加载时会转为 unknown 并阻止再次发送。
                raise
            except PlatformError:
                await asyncio.to_thread(self.store.finish, uid, item.bvid, "rejected")
                await self.record(uid, item, mode, "rejected", detail)
                raise
            except (LoginRequired, PublishNotSent):
                await asyncio.to_thread(self.store.finish, uid, item.bvid, "cancelled")
                await self.record(uid, item, mode, "cancelled", detail)
                raise
            except Exception:
                await asyncio.to_thread(self.store.finish, uid, item.bvid, "unknown")
                await self.record(uid, item, mode, "unknown", detail)
                raise UnknownPublish(f"{item.bvid} 的评论结果待核对，不会自动重试") from None
            await asyncio.to_thread(self.store.finish, uid, item.bvid, "sent", rpid)
            await self.record(uid, item, mode, "sent", detail)
            lines.append(f"{item.bvid} 已提交评论（rpid={rpid}）：{decision.comment}")
            commented_item = item
            break  # 一轮最多发布一条；平台接收不等同于最终公开展示。
        if commented_item is not None and self.share_video is not None:
            try:
                share_result = await self.share_video(uid, commented_item)
                if share_result:
                    lines.append(share_result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # 群分享失败不能回滚已提交的 B 站评论，更不能让它变成 unknown 后重发。
                lines.append(f"视频评论已提交，群分享处理失败：{type(exc).__name__}")
        return "\n".join(lines) or "本轮没有符合条件的新视频"

    async def status(self) -> str:
        cfg = self.get_config()
        uid = self.client.uid
        stats = await asyncio.to_thread(self.store.stats, uid, local_day_start(self.clock(), cfg.browse.timezone))
        mode = "自动发布" if cfg.comment.publish_enabled else "预览"
        behavior = "自主决定刷不刷" if cfg.browse.autonomous_enabled else "固定间隔浏览"
        control = "暂停" if self.paused else "运行"
        choice = await asyncio.to_thread(self.store.latest_browse_decision, uid)
        autonomy = ""
        waits = await asyncio.to_thread(self.store.recoveries, uid)
        recovery = "\n".join(
            f"读取恢复：{w['stage']}；{w['reason']}；{local_time}"
            for w in waits
            for local_time in [
                datetime.fromtimestamp(w["retry_at"], ZoneInfo(cfg.browse.timezone)).strftime("%m-%d %H:%M:%S")
            ]
        )
        usage = await asyncio.to_thread(self.store.model_usage, uid, local_day_start(self.clock(), cfg.browse.timezone))
        if choice:
            autonomy = (
                f"\n最近自主决定：{'去浏览' if choice['should_browse'] else '暂时不刷'}；{choice['reason']}"
                f"\n下次意愿检查：{self.next_check_status(choice, cfg)}"
            )
        return (f"账号：{self.client.name} / {uid}" if uid else "账号：未登录") + (
            f"\n状态：{control}；模式：{mode}；{behavior}\n今日分析：{stats['views']}/{cfg.browse.videos_per_day}"
            f"；发送尝试：{stats['attempts']}/{cfg.comment.per_day}"
            f"\n跟进回复：{'开启' if cfg.replies.enabled else '关闭'}；今日发送尝试：{stats['reply_attempts']}/{cfg.replies.per_day}"
            f"；今日分析：{stats['reply_analyses']}/{cfg.replies.analyses_per_day}"
            f"\n评论发送间隔：{self.comment_cooldown(stats, cfg) or '已满足发送间隔'}"
            f"\n评论候选：{await self.candidate_pool_status()}"
            f"\n待核对：{', '.join(stats['unresolved']) or '无'}\n暂停原因：{self.pause_reason or '无'}"
            f"{autonomy}\n{recovery}\n今日模型调用：{usage['calls']}；累计 {usage['elapsed_seconds']} 秒"
            f"\n最近结果：{self.last_result[:1200]}"
        )
