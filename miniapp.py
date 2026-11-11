"""QQ LightAppSvc 中 B 站正式分享模板的请求和回执。"""

from typing import Any, Dict, Iterator

import json


BILI_ARK_COMMAND = "LightAppSvc.mini_app_share.AdaptShareInfo"
BILI_APP_ID = "1109937557"


def _varint(value: int) -> bytes:
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _field(number: int, value: int | str | bytes) -> bytes:
    if isinstance(value, int):
        return _varint(number << 3) + _varint(value)
    data = value.encode("utf-8") if isinstance(value, str) else value
    return _varint(number << 3 | 2) + _varint(len(data)) + data


def build_bili_ark_request(title: str, cover: str, video_url: str, *, jump_path: str, timestamp_ms: int) -> str:
    # 对照 NapCatQQ 的 MiniAppAdaptShareInfoReqBody 与 Bili 模板。
    body = b"".join(_field(number, value) for number, value in (
        (1, _field(2, b"")),
        (2, BILI_APP_ID), (3, "哔哩哔哩"), (4, title), (5, timestamp_ms),
        (6, 1), (7, 1), (8, 0),
        (9, cover), (10, ""), (11, jump_path),
        (12, "https://miniapp.gtimg.cn/public/appicon/51f90239b78a2e4994c11215f4c4ba15_200.jpg"),
        (13, 3), (14, 0), (15, "cfc5f7b05b44b5956502edaecf9d2240"), (16, 0),
        (17, video_url), (18, b""), (19, _field(1, "") + _field(2, "")), (20, ""),
    ))
    return (_field(2, "V1_PC_MINISDK_99.99.99_1_APP_A") + _field(4, body)).hex()


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    result = 0
    for shift in range(0, 64, 7):
        if offset >= len(data):
            break
        byte = data[offset]
        offset += 1
        if shift == 63 and byte > 1:
            break
        result |= (byte & 127) << shift
        if byte < 128:
            return result, offset
    raise ValueError("无效的 QQ 签发回执")


def _fields(data: bytes) -> Iterator[tuple[int, int | bytes]]:
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        number, wire = divmod(tag, 8)
        if not number:
            raise ValueError("无效的 QQ 签发回执")
        if wire == 0:
            value, offset = _read_varint(data, offset)
            yield number, value
            continue
        if wire == 2:
            length, offset = _read_varint(data, offset)
        elif wire in (1, 5):
            length = 8 if wire == 1 else 4
        else:
            raise ValueError("无效的 QQ 签发回执")
        end = offset + length
        if end > len(data):
            raise ValueError("无效的 QQ 签发回执")
        if wire == 2:
            yield number, data[offset:end]
        offset = end


def validate_bili_ark(card: Any, video_url: str) -> None:
    if not isinstance(card, dict) or card.get("app") != "com.tencent.miniapp_01":
        raise ValueError("QQ 返回的不是 B 站小程序卡片")
    meta = card.get("meta")
    detail = meta.get("detail_1") if isinstance(meta, dict) else None
    config = card.get("config")
    if (not isinstance(detail, dict) or str(detail.get("appid")) != BILI_APP_ID
            or not detail.get("url") or detail.get("qqdocurl") != video_url
            or not isinstance(config, dict) or not config.get("token")
            or not all(card.get(key) for key in ("ver", "prompt", "view"))):
        raise ValueError("QQ 签发的 B 站卡片缺少原生地址、网页地址或签名")


def parse_bili_ark_response(payload: Any, video_url: str) -> Dict[str, Any]:
    if not isinstance(payload, str) or not 0 < len(payload) <= 131072:
        raise ValueError("QQ 未返回有效的签发回执")
    response = dict(_fields(bytes.fromhex(payload)))
    if response.get(2, 0) != 0:
        raise ValueError("QQ 未签发 B 站小程序卡片")
    body = response.get(4)
    encoded_ark = dict(_fields(body)).get(2) if isinstance(body, bytes) else None
    if not isinstance(encoded_ark, bytes):
        raise ValueError("QQ 未返回 B 站小程序卡片")
    ark = json.loads(encoded_ark)
    if not isinstance(ark, dict) or ark.get("appName") != "com.tencent.miniapp_01":
        raise ValueError("QQ 返回的不是小程序卡片")
    # 只转换 Ark 外层字段名，保留 QQ 签发的完整内容。
    card = {
        "ver": ark.get("ver"), "prompt": ark.get("prompt"), "config": ark.get("config"), "app": ark["appName"],
        "view": ark.get("appView"), "meta": ark.get("metaData"), "miniappShareOrigin": 3, "miniappOpenRefer": "10002",
    }
    validate_bili_ark(card, video_url)
    return card
