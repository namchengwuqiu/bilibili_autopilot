"""为本插件的 B 站库请求绑定 httpx，避免改变同进程其他插件的客户端。"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

from bilibili_api.clients.HTTPXClient import HTTPXClient
from bilibili_api.utils import network


# 17.4.2 只提供进程级客户端选择。Api 及其 buvid、ticket、WBI 辅助请求
# 均通过 network.get_client 获取传输层，因此用任务上下文绑定本插件的客户端。
# 其他任务继续调用原入口；重载本模块时复用已有上下文，避免叠加包装器。
_context = getattr(network.get_client, "_autopilot_context", None)
if _context is None:
    _context: ContextVar[Optional[HTTPXClient]] = ContextVar("bilibili_autopilot_httpx", default=None)

    def _get_client(original=network.get_client):
        client = _context.get()
        return client if client is not None else original()

    _get_client._autopilot_context = _context
    network.get_client = _get_client


@contextmanager
def api_transport(client: HTTPXClient) -> Iterator[None]:
    token = _context.set(client)
    try:
        yield
    finally:
        _context.reset(token)
