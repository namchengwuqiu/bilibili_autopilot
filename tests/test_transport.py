"""真实 B 站库配合模拟 HTTP：扫码、读取及签名不再进入 curl 证书路径。"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, Mock, patch

import asyncio
import importlib
import ssl
import unittest

from bilibili_api import Credential, get_selected_client
from bilibili_api.utils import network
import httpx

from local_bilibili_autopilot import transport
from local_bilibili_autopilot.client import BiliClient, LoginDiagnosticError, LoginRequired, UnknownPublish
from local_bilibili_autopilot.models import VideoItem


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = TemporaryDirectory(prefix="扫码证书-")
        self.addCleanup(self.temp.cleanup)
        self.client = BiliClient(Path(self.temp.name) / "凭据.json", logger=Mock())
        self.addAsyncCleanup(self.client.close)
        self.selected = get_selected_client()
        self.fallback = self.selected[1]
        self.original_client = network.get_client()
        self.addAsyncCleanup(self.original_client.close)
        # 让原来默认客户端每次调用都抛出故障，证明真实库没有使用它。
        self.old_request = AsyncMock(side_effect=RuntimeError("simulated curl CA file error, code=77"))
        patcher = patch.object(self.original_client, "request", self.old_request)
        patcher.start()
        self.addCleanup(patcher.stop)
        await self.client.http.aclose()
        await self.client.api_http.close()
        self.requests = []
        self.mid = 1
        self.api_error = None
        self.post_error = None

        async def respond(request):
            self.requests.append(request)
            if request.url.path.endswith("/poll"):
                cookies = {"SESSDATA": "test-session", "bili_jct": "test-csrf", "DedeUserID": "1"}
                headers = [("set-cookie", f"{k}={v}; Path=/; Domain=.bilibili.com; Secure")
                           for k, v in cookies.items()]
                return httpx.Response(200, json={"code": 0, "data": {"code": 0}}, headers=headers)
            if self.api_error:
                raise self.api_error
            if request.method == "POST" and self.post_error:
                raise self.post_error
            data = {"mid": self.mid, "name": "模拟账号"} if request.method == "GET" else {"rpid": 9988}
            return httpx.Response(200, json={"code": 0, "data": data})

        self.client.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        self.client.api_http._HTTPXClient__session = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        # 保留真实账号验证与签名逻辑，仅替代与故障无关的动态设备及 ticket 数据。
        for name, result in (("get_buvid", ("test-device-3", "test-device-4")),
                             ("get_bili_ticket", ("test-ticket", "9999999999"))):
            patcher = patch.object(network, name, AsyncMock(return_value=result))
            patcher.start()
            self.addCleanup(patcher.stop)

    def logs(self):
        return "\n".join(call.args[0] % call.args[1:] for call in self.client.logger.mock_calls)

    async def login(self):
        return await self.client.poll_login("test-qr-key")

    async def test_real_account_verification_uses_httpx_and_saves_in_unicode_path(self):
        self.assertIn("已登录", await self.login())
        self.assertTrue(self.client.credential_path.exists())
        self.assertEqual(self.client.uid, "1")
        self.assertIn("stage=验证扫码账号 client=httpx", self.logs())
        self.assertIn("test-session", self.requests[-1].headers["cookie"])
        self.assertEqual(get_selected_client(), self.selected)
        self.old_request.assert_not_awaited()
        self.assertEqual(len(self.client.http.cookies), 0)
        self.assertNotIn("test-session", self.logs())

    async def test_subsequent_account_check_uses_httpx_with_same_uid_check(self):
        await self.login()
        self.assertEqual(await self.client.check_account(), "1")
        self.old_request.assert_not_awaited()
        self.mid = 2
        with self.assertRaises(LoginRequired):
            await self.client.check_account()

    async def test_uid_mismatch_never_persists_credentials(self):
        self.mid = 2
        with self.assertRaises(LoginRequired):
            await self.login()
        self.assertFalse(self.client.credential_path.exists())
        self.assertIsNone(self.client.credential)

    async def test_logout_clears_api_session_cookies(self):
        await self.login()
        cookies = self.client.api_http.get_wrapped_session().cookies
        cookies.set("SESSDATA", "test-session", domain=".bilibili.com")
        await self.client.logout()
        self.assertEqual(len(cookies), 0)
        self.assertIsNone(self.client.credential)

    async def test_ticket_helper_also_uses_local_transport(self):
        # _get_bili_ticket 直接读取 network.get_client，不经过 Api。
        with patch.object(self.client.api_http, "request", AsyncMock()) as request:
            request.return_value = Mock(json=Mock(return_value={"data": {"ticket": "test-ticket"}}))
            async def operation():
                return await network._get_bili_ticket(Credential())
            self.assertEqual(await self.client._read_stage("准备 ticket", operation()), "test-ticket")
        request.assert_awaited_once()
        self.old_request.assert_not_awaited()

    async def test_httpx_certificate_failure_still_stops_login(self):
        self.api_error = httpx.ConnectError("CERTIFICATE_VERIFY_FAILED")
        with self.assertRaises(LoginDiagnosticError) as raised:
            await self.login()
        self.assertIn("certificate_verify_failed", str(raised.exception))
        self.assertFalse(self.client.credential_path.exists())
        self.old_request.assert_not_awaited()

    async def test_signing_uses_local_transport_and_unknown_post_is_not_retried(self):
        self.client.credential = Credential(sessdata="test-session", bili_jct="test-csrf", dedeuserid="1",
                                            buvid3="test-device-3", buvid4="test-device-4")
        item = VideoItem("BV1xx411c7mD", 100, "模拟视频", "", "", "", 0, "")
        async def signing_key(credential):
            await network.get_client().request(method="GET", url="https://api.bilibili.com/mock-signing")
            return "0123456789abcdef0123456789abcdef"
        self.post_error = httpx.ReadTimeout("simulated unknown POST")
        with patch.object(network, "get_wbi_mixin_key", signing_key):
            with self.assertRaises(UnknownPublish):
                await self.client.publish(item, "测试内容", lambda: True)
        self.assertEqual([r.method for r in self.requests], ["GET", "POST"])
        self.old_request.assert_not_awaited()

    async def test_context_is_isolated_across_tasks_and_restored_after_cancellation(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def local():
            with transport.api_transport(self.client.api_http):
                entered.set()
                await release.wait()
                self.assertIs(network.get_client(), self.client.api_http)
        task = asyncio.create_task(local())
        await entered.wait()
        self.assertIsInstance(network.get_client(), self.fallback)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsInstance(network.get_client(), self.fallback)

    async def test_reload_reuses_wrapper_and_nested_context_restores(self):
        wrapper = network.get_client
        importlib.reload(transport)
        self.assertIs(network.get_client, wrapper)
        with transport.api_transport(self.client.api_http):
            with transport.api_transport(self.client.api_http):
                self.assertIs(network.get_client(), self.client.api_http)
            self.assertIs(network.get_client(), self.client.api_http)
        self.assertIsInstance(network.get_client(), self.fallback)

    async def test_api_http_starts_with_certificate_and_hostname_verification_enabled(self):
        with patch("local_bilibili_autopilot.client.HTTPXClient", wraps=transport.HTTPXClient) as factory:
            client = BiliClient(Path(self.temp.name) / "other.json")
        self.addAsyncCleanup(client.close)
        self.assertIs(factory.call_args.kwargs["verify_ssl"], True)
        context = client.api_http.get_wrapped_session()._transport._pool._ssl_context
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
