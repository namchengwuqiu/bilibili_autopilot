"""评论成功后的群分享；仅解析真实已有群会话，发送意图先落盘。"""

from typing import Any, Awaitable, Callable, Dict, Optional

import asyncio
import json
import time

from .config import AutopilotConfig
from .models import DecisionValidationError, VideoItem
from .state import StateStore


SHARE_OUTCOMES = {"pending": "群分享发送中", "sent": "已分享至群聊", "failed": "群分享失败",
                  "unknown": "群分享结果待核对", "cancelled": "群分享已取消", "partial": "仅视频卡片已发送"}


def card_receipt_outcome(message: Any, card: str, group_id: str, message_id: str) -> str:
    """确认回读的是目标群的原消息，且卡片类型与跳转地址仍然正确。"""
    if (not isinstance(message, dict) or message.get("message_type") != "group"
            or str(message.get("group_id")) != group_id
            or str(message.get("message_id")) != message_id):
        return "unknown"
    segments = message.get("message")
    if not isinstance(segments, list):
        return "unknown"
    if any(isinstance(segment, dict) and segment.get("type") == "text"
           and isinstance(segment.get("data"), dict)
           and "发送者版本过低" in str(segment["data"].get("text", "")) for segment in segments):
        return "failed"
    expected = json.loads(card)
    for segment in segments:
        if not isinstance(segment, dict) or not isinstance(segment.get("data"), dict):
            continue
        data = segment["data"]
        if segment.get("type") != "json":
            continue
        raw = data.get("data")
        try:
            received = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            continue
        if not isinstance(received, dict) or any(
                received.get(key) != expected.get(key) for key in ("app", "view")):
            continue
        meta = received.get("meta")
        if not isinstance(meta, dict):
            continue
        for name, fields in expected.get("meta", {}).items():
            actual = meta.get(name)
            if not isinstance(fields, dict) or not isinstance(actual, dict):
                break
            if any(actual.get(key) != fields.get(key) for key in ("jumpUrl", "qqdocurl", "url", "appid", "title", "desc")
                   if key in fields):
                break
        else:
            return "sent"
    return "unknown"


def send_outcome(result: Any) -> tuple[str, str]:
    message_id = str(result.get("message_id") or "").strip() if isinstance(result, dict) else ""
    if result is True or (isinstance(result, dict) and result.get("sent") is True):
        return "sent", message_id
    if result is False or (isinstance(result, dict) and result.get("sent") is False):
        return "failed", ""
    return "unknown", ""


class VideoSharer:
    def __init__(self, store: StateStore, get_config: Callable[[], AutopilotConfig],
                 chat: Any, send: Any, can_send: Callable[[], bool],
                 compose: Callable[[str, VideoItem, AutopilotConfig], Awaitable[str]],
                 clock: Callable[[], float] = time.time,
                 compose_card: Optional[Callable[[VideoItem], Awaitable[str]]] = None,
                 read_card: Optional[Callable[[str, str], Awaitable[Any]]] = None):
        self.store, self.get_config, self.chat, self.send = store, get_config, chat, send
        self.can_send, self.clock = can_send, clock
        self.compose = compose
        self.compose_card = compose_card
        self.read_card = read_card

    async def resolve(self, group_id: str) -> Dict[str, Any]:
        streams = await asyncio.wait_for(self.chat.get_group_streams(platform="qq"), timeout=10)
        # SDK 已将 Host 的 {success, streams} 响应解包为列表。
        if not isinstance(streams, list):
            raise ValueError("无法读取已有 QQ 群会话")
        # 不使用无账号路由的占位会话，也不猜测 hash 或自动创建会话。
        matches = {s["session_id"]: s for s in streams if isinstance(s, dict)
                   and s.get("platform") == "qq" and str(s.get("group_id")) == group_id
                   and s.get("chat_type") == "group" and s.get("session_id") and s.get("account_id")}
        if len(matches) != 1:
            raise ValueError("未找到唯一的真实群会话，请确认麦麦已加入该群并在群内接收过消息")
        return next(iter(matches.values()))

    async def targets_summary(self) -> str:
        targets = tuple(self.get_config().share.qq_group_ids)
        if not targets:
            return "未配置群号，不分享"
        lines = []
        for group_id in targets:
            try:
                stream = await self.resolve(group_id)
                lines.append(f"{stream.get('group_name') or 'QQ 群'}（{group_id}，已找到已有会话）")
            except Exception:
                lines.append(f"QQ 群 {group_id}：未确认唯一的真实会话，分享时会重新检查")
        return "；".join(lines)

    async def share(self, uid: str, item: VideoItem) -> str:
        cfg = self.get_config().model_copy(deep=True)
        targets = tuple(cfg.share.qq_group_ids)
        use_card = cfg.share.miniapp_enabled
        card_label = "小程序卡片"
        video_url = f"https://www.bilibili.com/video/{item.bvid}"
        lines = []
        intro = None
        card = None
        for group_id in targets:
            if not self.can_send() or group_id not in self.get_config().share.qq_group_ids:
                break
            reserved = await asyncio.to_thread(self.store.reserve_share, uid, item.bvid, group_id, self.clock(), item.title)
            if not reserved:
                continue
            post_started = False
            card_sent = False
            phase = "群会话查找"
            status, detail, message_id = "failed", "未发送", ""
            try:
                stream = await self.resolve(group_id)
                if intro is None and self.can_send() and self.get_config() == cfg:
                    phase = "分享文案生成"
                    intro = await self.compose(uid, item, cfg)
                if use_card and card is None and self.can_send() and self.get_config() == cfg:
                    phase = f"{card_label}生成"
                    if self.compose_card is None:
                        raise ValueError("未提供小程序卡片生成服务")
                    card = await asyncio.wait_for(self.compose_card(item), timeout=20)
                    if self.read_card is None:
                        raise ValueError("未提供 QQ 群卡片回读服务")
                if not self.can_send() or group_id not in self.get_config().share.qq_group_ids:
                    status, detail = "cancelled", "发送前已暂停、关闭发布或移除目标群"
                elif self.get_config() != cfg:
                    status, detail = "cancelled", "准备分享期间配置已变化，本次不分享"
                elif not intro:
                    status, detail = "failed", "未取得通过复核的分享感想，不发送"
                elif use_card:
                    phase = f"{card_label}发送"
                    post_started = True
                    result = await asyncio.wait_for(
                        self.send.custom("json", card, stream["session_id"], return_details=True,
                                         processed_plain_text=f"[B站小程序] {item.title}"), timeout=20)
                    status, message_id = send_outcome(result)
                    if status != "sent" or not message_id:
                        # Host 会把 QQ 缺少回执也返回为 sent=false，无法据此确定卡片未送达。
                        status, message_id = "unknown", ""
                        detail = f"{card_label}未取得 QQ 发送回执，结果待核对；未发送分享感想，不会自动重发"
                    else:
                        phase = f"{card_label}群记录核对"
                        message = await asyncio.wait_for(self.read_card(group_id, message_id), timeout=10)
                        status = card_receipt_outcome(message, card, group_id, message_id)
                        post_started = False
                        if status != "sent":
                            detail = (
                                f"{card_label}被 QQ 替换为“发送者版本过低，无法展示内容”" if status == "failed" else
                                f"{card_label}已取得发送回执，但群记录未确认原卡片及目标链接"
                            ) + "；未发送分享感想，不会自动重发"
                            await asyncio.to_thread(self.store.finish_share, uid, item.bvid, group_id, status, detail, message_id)
                            lines.append(f"{SHARE_OUTCOMES[status]}：QQ 群 {group_id}；{detail}")
                            break
                        card_sent = True
                        post_started = False
                        await asyncio.to_thread(self.store.mark_share_card_sent, uid, item.bvid, group_id, message_id)
                        # 卡片确认后再次检查暂停和配置，再发送已通过复核的感想。
                        if not self.can_send() or self.get_config() != cfg:
                            status, detail = "partial", f"{card_label}已发送；发送分享感想前已暂停或配置发生变化，不会自动重发"
                        else:
                            phase = "分享感想发送"
                            post_started = True
                            result = await asyncio.wait_for(
                                self.send.text(intro, stream["session_id"], return_details=True), timeout=20)
                            status, intro_id = send_outcome(result)
                            if status == "sent":
                                message_id = intro_id
                                detail = f"分享文案：{intro}"
                            elif status == "failed":
                                status, detail = "partial", f"{card_label}已发送；分享感想发送失败，不会自动重发"
                            else:
                                detail = f"{card_label}已发送；分享感想发送结果待核对，不会自动重发"
                else:
                    text = f"{intro}\n{video_url}"
                    post_started = True
                    result = await asyncio.wait_for(self.send.text(text, stream["session_id"], return_details=True), timeout=20)
                    if result is True or (isinstance(result, dict) and result.get("sent") is True):
                        status, detail = "sent", f"分享文案：{intro}"
                        message_id = str(result.get("message_id") or "") if isinstance(result, dict) else ""
                    elif result is False or (isinstance(result, dict) and result.get("sent") is False):
                        status, detail = "failed", "群消息发送服务返回失败，不会自动重发"
                    else:
                        status, detail = "unknown", "群消息返回格式无法确认，不会自动重发"
            except asyncio.CancelledError:
                # pending 在重载时转为 unknown，避免中断后重复向群里发消息。
                raise
            except DecisionValidationError as exc:
                status = "unknown" if post_started else "partial" if card_sent else "failed"
                detail = (
                    "群消息可能已发送，结果待核对；不会自动重发" if post_started else
                    f"{phase}校验失败 [{exc.code}]：{str(exc)[:300]}；未发送，不会自动重发"
                )
                if card_sent:
                    detail = f"{card_label}已发送；{phase}校验失败 [{exc.code}]，感想结果待核对，不会自动重发"
            except Exception as exc:
                status = "unknown" if post_started else "partial" if card_sent else "failed"
                detail = ("群消息可能已发送，结果待核对" if post_started else f"{phase}失败，未发送") + f"（{type(exc).__name__}）；不会自动重发"
                if card_sent:
                    detail = f"{card_label}已发送；{phase}失败，" + ("感想结果待核对" if post_started else "未发送感想") + f"（{type(exc).__name__}）；不会自动重发"
            await asyncio.to_thread(self.store.finish_share, uid, item.bvid, group_id, status, detail, message_id)
            lines.append(f"{SHARE_OUTCOMES[status]}：QQ 群 {group_id}；{detail}")
            if status == "failed" and phase in {"分享文案生成", f"{card_label}生成"}:
                break
        return "\n".join(lines)
