"""B 站读取和单次评论发布。写请求不交给带自动重试的通用 API 包装器。"""

from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import asyncio
import json
import os
import tempfile
import time

from bilibili_api import Credential, comment, homepage, session, user, video
from bilibili_api.exceptions import NetworkException, ResponseCodeException
from bilibili_api.utils.network import Api

import httpx
import qrcode

from .models import Evidence, IncomingReply, VideoItem, normalized, recommendations


READ_NETWORK_ERRORS = (asyncio.TimeoutError, httpx.TimeoutException, httpx.NetworkError)
try:
    from curl_cffi.requests.exceptions import ConnectionError as CurlConnectionError, Timeout as CurlTimeout
except ImportError:
    pass  # curl_cffi 是 B 站库的可选客户端，未安装时仍支持 httpx。
else:
    READ_NETWORK_ERRORS += (CurlConnectionError, CurlTimeout)


class LoginRequired(Exception):
    pass


class PlatformError(Exception):
    def __init__(self, operation: str, code: int):
        self.operation = operation
        self.code = code
        super().__init__(f"{operation}被平台拒绝（code={code}）")


class VideoReadError(Exception):
    """视频读取阶段的可公开错误，不包含签名地址或请求头。"""

    def __init__(self, reason: str, retryable: bool):
        self.retryable = retryable
        super().__init__(reason)


class SubtitleText(str):
    """兼容旧的文本接口，同时保留字幕行时间，不缓存完整文件。"""

    def __new__(cls, text: str, segments: List[Dict[str, Any]]):
        value = super().__new__(cls, text)
        value.segments = segments
        return value


def select_subtitles(rows: List[Dict[str, Any]], max_chars: int, topic: str = "") -> Tuple[str, List[Dict[str, Any]]]:
    """在预算内优先取相关片段，兼顾开头、中段与末尾；输出仍按时间排序。"""
    if not rows:
        return "", []
    terms = {normalized(topic[i : i + 2]) for i in range(max(0, len(topic) - 1))}
    terms.discard("")
    score = lambda i: sum(term in normalized(rows[i]["text"]) for term in terms)
    anchors = [0, len(rows) // 2, len(rows) - 1]
    ranked = sorted(range(len(rows)), key=lambda i: (-score(i), i)) if terms else []
    preferred = [i for i in ranked if score(i) > 0]
    order = list(dict.fromkeys([*preferred[:1], *anchors, *preferred[1:], *ranked, *range(len(rows))]))
    selected, used = [], 0
    # 先给三个覆盖位置留空间，避免一条过长字幕独占预算。
    per_segment = max(20, max_chars // 4)
    for i in order:
        if max_chars - used < 10:
            break
        row = rows[i]
        text = row["text"][: min(per_segment, max_chars - used)]
        if not text:
            continue
        selected.append({"source": "subtitle", "from": row["from"], "to": row["to"], "text": text})
        used += len(text) + 1
    selected.sort(key=lambda r: (r["from"], r["to"]))
    return "\n".join(r["text"] for r in selected), selected


class UnknownPublish(Exception):
    """请求可能已发出，但无法确认结果。"""


class PublishNotSent(Exception):
    """已确定评论 POST 尚未调用。"""


def write_private_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".credential-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def make_qr(url: str) -> bytes:
    buffer = BytesIO()
    qrcode.make(url).save(buffer, format="PNG")
    return buffer.getvalue()


def subtitle_url(raw: str) -> str:
    value = "https:" + raw if raw.startswith("//") else raw
    parsed = urlsplit(value)
    host = (parsed.hostname or "").lower()
    allowed = ("bilibili.com", "hdslb.com", "bilivideo.com", "biliapi.net")
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
        or not any(host == suffix or host.endswith("." + suffix) for suffix in allowed)
    ):
        raise ValueError("字幕地址不属于允许的 B 站 HTTPS 域名")
    return value


class BiliClient:
    def __init__(self, credential_path: Path):
        self.credential_path = credential_path
        self.credential: Optional[Credential] = None
        self.uid = ""
        self.name = ""
        # httpx 默认不会重试；禁止跟随评论发布响应的重定向。
        self.http = httpx.AsyncClient(
            timeout=20,
            follow_redirects=False,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.bilibili.com/"},
        )
        self.public_http = httpx.AsyncClient(timeout=20, follow_redirects=False)

    async def load(self) -> None:
        def read() -> Optional[Dict[str, Any]]:
            if not self.credential_path.exists():
                return None
            self.credential_path.chmod(0o600)
            return json.loads(self.credential_path.read_text(encoding="utf-8"))

        payload = await asyncio.to_thread(read)
        if payload is not None:
            self._use_credential(payload)

    def _use_credential(self, payload: Dict[str, Any]) -> None:
        cookies = payload["cookies"]
        if not all(
            isinstance(cookies.get(key), str) and cookies[key] for key in ("SESSDATA", "bili_jct", "DedeUserID")
        ):
            raise LoginRequired("登录凭据不完整，请重新扫码")
        uid = str(int(cookies["DedeUserID"]))
        if uid != str(payload["uid"]) or int(uid) <= 0:
            raise LoginRequired("登录凭据与账号不一致，请重新扫码")
        self.credential = Credential.from_cookies(cookies)
        self.uid, self.name = uid, str(payload.get("name", ""))

    async def close(self) -> None:
        await self.http.aclose()
        await self.public_http.aclose()

    async def logout(self) -> None:
        await asyncio.to_thread(self.credential_path.unlink, missing_ok=True)
        self.credential = None
        self.uid = self.name = ""
        self.http.cookies.clear()

    async def check_account(self) -> str:
        if self.credential is None:
            raise LoginRequired("尚未登录，请在私聊中使用 /bili_auto login")
        info = await self._read_stage("检查登录", user.get_self_info(self.credential))
        if str(info.get("mid", "")) != self.uid:
            raise LoginRequired("当前账号与保存的 UID 不一致，请重新登录")
        return self.uid

    async def begin_login(self) -> Tuple[bytes, str]:
        response = await self.http.get("https://passport.bilibili.com/x/passport-login/web/qrcode/generate")
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise PlatformError("生成登录二维码", int(payload.get("code", -1)))
        data = payload["data"]
        return await asyncio.to_thread(make_qr, data["url"]), data["qrcode_key"]

    async def poll_login(self, key: str) -> str:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            response = await self.http.get(
                "https://passport.bilibili.com/x/passport-login/web/qrcode/poll", params={"qrcode_key": key}
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != 0:
                raise PlatformError("查询扫码状态", int(payload.get("code", -1)))
            data = payload.get("data") or {}
            code = data.get("code")
            if code == 0:
                keep = {"SESSDATA", "bili_jct", "buvid3", "buvid4", "DedeUserID"}
                cookies = {k: v for k, v in response.cookies.items() if k in keep}
                self.http.cookies.clear()
                if not all(cookies.get(k) for k in ("SESSDATA", "bili_jct", "DedeUserID")):
                    raise LoginRequired("扫码成功但缺少发布评论所需凭据，请重新登录")
                credential = Credential.from_cookies(cookies)
                try:
                    info = await asyncio.wait_for(user.get_self_info(credential), timeout=30)
                except ResponseCodeException as exc:
                    raise PlatformError("验证扫码账号", exc.code) from None
                record = {"cookies": cookies, "uid": str(info["mid"]), "name": str(info.get("name", ""))}
                if record["uid"] != cookies["DedeUserID"]:
                    raise LoginRequired("扫码结果 UID 校验失败")
                await asyncio.to_thread(write_private_json, self.credential_path, record)
                self._use_credential(record)
                return f"已登录 {self.name}（UID {self.uid}）"
            if code == 86038:
                raise LoginRequired("二维码已过期，请重新使用 login")
            if code not in (86101, 86090):
                raise PlatformError("扫码登录", int(code or -1))
            await asyncio.sleep(2)
        raise LoginRequired("扫码超时，请重新使用 login")

    async def recommended(self) -> List[VideoItem]:
        payload = await self._read_stage("读取推荐", homepage.get_videos(self.credential))
        return recommendations(payload)

    async def incoming_replies(self, own: Dict[str, Dict[str, Any]], since: float) -> List[IncomingReply]:
        """只接收目标评论 ID、视频 ID、根楼层均与本地已发送记录一致的回复。"""
        if self.credential is None:
            raise LoginRequired("检查回复前需要登录")
        result, seen = [], set()
        cursor_id = cursor_time = None
        for _ in range(3):
            payload = await self._read_stage(
                "读取收到的回复", session.get_replies(self.credential, last_reply_id=cursor_id, reply_time=cursor_time)
            )
            rows = payload.get("items")
            if not isinstance(rows, list):
                raise ValueError("回复通知格式发生变化，缺少 items 列表")
            for row in rows:
                if not isinstance(row, dict):
                    continue
                item, sender = row.get("item"), row.get("user")
                if not isinstance(item, dict) or not isinstance(sender, dict):
                    continue
                try:
                    target = str(int(item.get("target_id", 0)))
                    root = int(item.get("root_id", 0))
                    if target not in own or int(item.get("business_id", 0)) != 1:
                        continue
                    parent = own[target]
                    bvid = parent["bvid"]
                    aid = video.Video(bvid=bvid).get_aid()
                    source = int(item.get("source_id", 0))
                    uid = str(int(sender.get("mid", 0)))
                    timestamp = float(row.get("reply_time", 0))
                    root_rpid = int(parent["root_rpid"])
                    if (
                        int(item.get("subject_id", 0)) != aid
                        or root not in (0, root_rpid)
                        or source <= 0
                        or source == int(target)
                        or int(uid) <= 0
                        or uid == self.uid
                        or not since <= timestamp <= time.time() + 300
                        or source in seen
                    ):
                        continue
                    text = item.get("source_content")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    seen.add(source)
                    result.append(
                        IncomingReply(
                            bvid,
                            aid,
                            root_rpid,
                            source,
                            int(target),
                            uid,
                            str(sender.get("nickname", ""))[:100],
                            text.strip()[:4000],
                            str(parent["root_text"])[:2000],
                            str(parent["text"])[:2000],
                            timestamp,
                        )
                    )
                except (ValueError, TypeError, KeyError):
                    continue
            cursor = payload.get("cursor") or {}
            if not rows or cursor.get("is_end"):
                break
            next_cursor = (cursor.get("id"), cursor.get("time"))
            if next_cursor == (cursor_id, cursor_time) or not all(type(v) is int and v > 0 for v in next_cursor):
                break
            cursor_id, cursor_time = next_cursor
        return sorted(result, key=lambda message: (message.received_at, message.source_rpid))

    async def _read_stage(self, stage: str, operation: Awaitable[Any]) -> Any:
        try:
            return await asyncio.wait_for(operation, timeout=30)
        except ResponseCodeException as exc:
            raise PlatformError(stage, exc.code) from None
        except READ_NETWORK_ERRORS as exc:
            raise VideoReadError(f"{stage}超时或网络连接失败：{type(exc).__name__}", True) from None
        except (NetworkException, httpx.HTTPStatusError) as exc:
            code = exc.status if isinstance(exc, NetworkException) else exc.response.status_code
            if code in {401, 403, 412, 429}:
                raise PlatformError(f"{stage}（HTTP）", code) from None
            raise VideoReadError(f"{stage}失败（HTTP {code}）", code >= 500 or code == 408) from None

    async def understand(self, item: VideoItem, max_chars: int) -> Tuple[VideoItem, Evidence]:
        resource = video.Video(bvid=item.bvid, credential=self.credential)
        info = await self._read_stage("读取视频信息", resource.get_info())
        if int(info["aid"]) != item.aid or info["bvid"] != item.bvid:
            raise ValueError("视频 ID 校验失败")
        item = replace(
            item,
            title=str(info.get("title", ""))[:300],
            description=str(info.get("desc", ""))[:3000],
            duration=int(info.get("duration", 0)),
            up_uid=str(info["owner"]["mid"]),
            up_name=str(info["owner"].get("name", ""))[:100],
            topic=str(info.get("tid") or info.get("tname") or item.topic)[:100],
        )
        if int(info.get("videos", 1)) > 1:
            return item, Evidence("unavailable", "")
        cid = int(info["cid"])
        summary_error = None
        try:
            summary = await self._read_stage("读取 AI 摘要", resource.get_ai_conclusion(cid=cid))
        except (PlatformError, VideoReadError) as exc:
            # 摘要与字幕是独立内容来源；仅在摘要临时不可用时继续读取字幕。
            if not (exc.code == -504 if isinstance(exc, PlatformError) else exc.retryable):
                raise
            summary_error = exc
        else:
            model_result = summary.get("model_result") or {}
            text = model_result.get("summary", "")
            if isinstance(text, str) and len(text.strip()) >= 30:
                return item, Evidence(
                    "B站 AI 摘要（可能存在概括误差）",
                    text.strip()[:max_chars],
                    coverage="仅 AI 摘要，未读取字幕、画面或音频",
                    cid=cid,
                )
        try:
            subtitles = await self._read_stage("读取字幕列表", resource.get_subtitle(cid=cid))
        except (PlatformError, VideoReadError) as exc:
            if summary_error is None:
                raise
            retryable = exc.code == -504 if isinstance(exc, PlatformError) else exc.retryable
            if not retryable:
                raise
            raise VideoReadError(f"{summary_error}；{exc}", True) from None
        entries = (subtitles or {}).get("subtitles") or []
        entries = sorted(entries, key=lambda x: not str(x.get("lan", "")).startswith(("zh", "ai-zh")))
        if not entries:
            if summary_error is not None:
                raise VideoReadError(f"{summary_error}；视频没有可用字幕", True) from None
            return item, Evidence("unavailable", "")
        url = subtitle_url(entries[0]["subtitle_url"])
        text = await self._read_stage("下载字幕文件", self._subtitle_text(url))
        if summary_error is not None and len(text.strip()) < 30:
            raise VideoReadError(f"{summary_error}；字幕不足三十字，无法提供有效内容依据", True) from None
        segments = text.segments if isinstance(text, SubtitleText) else []
        sampled, selected = select_subtitles(segments, max_chars)
        return item, Evidence(
            "字幕（开头、中段和结尾的文本片段）",
            sampled if segments else text.strip()[:max_chars],
            str(summary_error) if summary_error is not None else "",
            selected,
            "字幕文本抽样，未读取画面或音频；未列出的片段不在覆盖范围",
            cid,
        )

    async def enrich(self, item: VideoItem, evidence: Evidence, topic: str, max_chars: int) -> Evidence:
        if evidence.cid <= 0:
            return evidence
        resource = video.Video(bvid=item.bvid, credential=self.credential)
        subtitles = await self._read_stage("补充相关字幕列表", resource.get_subtitle(cid=evidence.cid))
        entries = sorted(
            (subtitles or {}).get("subtitles") or [],
            key=lambda r: not str(r.get("lan", "")).startswith(("zh", "ai-zh")),
        )
        if not entries:
            return evidence
        text = await self._read_stage("补充相关字幕文件", self._subtitle_text(subtitle_url(entries[0]["subtitle_url"])))
        rows = text.segments if isinstance(text, SubtitleText) else []
        # 同一轮字幕已经有相关引用时复用；摘要候选才按评论主题补读，不读取画面或音频。
        summary_budget = min(len(evidence.text), max_chars // 3) if "摘要" in evidence.source else 0
        sampled, selected = select_subtitles(rows, max_chars - summary_budget, topic)
        if not sampled:
            sampled = text.strip()[: max_chars - summary_budget]
        if len(sampled.strip()) < 30:
            return evidence
        combined = ("摘要：" + evidence.text[:summary_budget] + "\n字幕：" if summary_budget else "") + sampled
        return Evidence(
            "AI 摘要与相关字幕" if summary_budget else "相关字幕",
            combined[:max_chars],
            evidence.read_warning,
            selected,
            "字幕按评论主题抽样并覆盖开头、中段、结尾；未读取画面或音频",
            0,
        )

    async def _subtitle_text(self, url: str) -> str:
        chunks = bytearray()
        # 字幕单独使用不含账号 Cookie 的会话；限制响应大小，不跟随重定向。
        async with self.public_http.stream("GET", url) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                chunks.extend(chunk)
                if len(chunks) > 2 * 1024 * 1024:
                    raise ValueError("字幕文件超过 2 MiB")
        payload = json.loads(chunks)
        rows = []
        for row in payload.get("body", []):
            if not isinstance(row, dict) or not isinstance(row.get("content"), str):
                continue
            try:
                start, end = float(row["from"]), float(row["to"])
            except (KeyError, ValueError, TypeError):
                continue
            if not (0 <= start <= end < 86400):
                continue
            rows.append({"from": start, "to": end, "text": row["content"].strip()})
        return SubtitleText("\n".join(r["text"] for r in rows), rows)

    async def publish_reply(self, message: IncomingReply, text: str, can_send: Callable[[], bool]) -> str:
        item = VideoItem(message.bvid, message.aid, "", "", "", 0)
        return await self.publish(item, text, can_send, root=message.root_rpid, parent=message.source_rpid)

    async def publish(
        self,
        item: VideoItem,
        text: str,
        can_send: Callable[[], bool],
        *,
        root: Optional[int] = None,
        parent: Optional[int] = None,
    ) -> str:
        if (root is None) != (parent is None) or (
            root is not None and (type(root) is not int or type(parent) is not int or root <= 0 or parent <= 0)
        ):
            raise PublishNotSent("回复楼层参数不完整或无效")
        if self.credential is None:
            raise LoginRequired("发布前登录态已清除")
        cookies = self.credential.get_cookies()
        if not cookies.get("SESSDATA") or not cookies.get("bili_jct"):
            raise LoginRequired("缺少评论发布凭据")
        # 只复用签名准备，不调用 Api.request 的 WBI 重试循环。
        # _prepare_request 是内部接口，因此固定依赖版本并测试实际签名参数。
        request = Api(**comment.API["comment"]["send"], credential=self.credential)
        data = dict(
            oid=item.aid,
            type=1,
            message=text,
            plat=1,
            gaia_source="main_web",
            statistics=json.dumps({"appId": 100, "platform": 5}),
        )
        if root is not None:
            data.update(root=root, parent=parent)
        request.update_data(**data)
        try:
            prepared = await asyncio.wait_for(request._prepare_request(), timeout=30)
        except ResponseCodeException as exc:
            raise PlatformError("准备评论签名", exc.code) from None
        except Exception as exc:
            raise PublishNotSent(f"评论签名准备失败：{type(exc).__name__}") from None
        # 签名准备可能等待网络；最后一次检查必须在准备结束后，紧挨着 POST。
        if not can_send():
            raise PublishNotSent("发布前已暂停或配置发生变化")
        try:
            outgoing = self.http.build_request(
                "POST",
                prepared["url"],
                params=prepared["params"],
                headers={
                    **prepared["headers"],
                    "Origin": "https://www.bilibili.com",
                    "Referer": f"https://www.bilibili.com/video/{item.bvid}",
                },
                data=prepared["data"],
            )
            outgoing.headers.pop("cookie", None)
            httpx.Cookies(prepared["cookies"]).set_cookie_header(outgoing)
            response = await self.http.send(outgoing, follow_redirects=False)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            raise UnknownPublish("评论请求结果无法确认，需要核对后再继续") from None
        if not isinstance(payload, dict) or type(payload.get("code")) is not int:
            raise UnknownPublish("评论响应格式异常，需要核对后再继续")
        if payload["code"] != 0:
            raise PlatformError("发布评论", payload["code"])
        data = payload.get("data") or {}
        rpid = data.get("rpid") or (data.get("reply") or {}).get("rpid")
        if not str(rpid or "").isdigit() or int(rpid) <= 0:
            raise UnknownPublish("平台返回成功但未给出评论 ID，需要核对后再继续")
        return str(rpid)
