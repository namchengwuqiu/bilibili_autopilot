"""登录阶段与日志脱敏回归；只使用模拟响应和临时凭据。"""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import asyncio
import os
import ssl
import unittest

from bilibili_api.exceptions import ResponseCodeException
import httpx

from local_bilibili_autopilot.client import (
    BiliClient, LoginDiagnosticError, PlatformError, login_environment, login_error_details,
)
from local_bilibili_autopilot.plugin import BilibiliAutopilotPlugin

try:
    from curl_cffi.requests.exceptions import SSLError as CurlSSLError
except ImportError:
    CurlSSLError = None


PRIVATE = "SESSDATA=private-session; bili_jct=private-csrf; qrcode_key=private-qr; " \
          "https://private-user:private-password@proxy.invalid/path?token=private-token C:\\private\\cert.pem"


class DiagnosticDetailsTests(unittest.TestCase):
    def test_certificate_chain_keeps_codes_and_static_hint_without_sensitive_text(self):
        cause = ssl.SSLCertVerificationError(1, "CERTIFICATE_VERIFY_FAILED " + PRIVATE)
        cause.verify_code = 20
        error = httpx.ConnectError(PRIVATE)
        error.__cause__ = cause
        details = login_error_details(error)
        self.assertIn("httpx.ConnectError", details)
        self.assertIn("ssl.SSLCertVerificationError", details)
        self.assertIn("verify_code=20", details)
        self.assertIn("certificate_verify_failed", details)
        for private in ("SESSDATA", "private-", "qrcode_key", "proxy.invalid", "cert.pem"):
            self.assertNotIn(private, details)

    @unittest.skipUnless(CurlSSLError, "curl_cffi 未安装")
    def test_real_curl_exception_keeps_ca_bundle_code_without_certificate_path(self):
        error = CurlSSLError("error setting certificate verify locations " + PRIVATE, code=77)
        details = login_error_details(error)
        self.assertIn("curl_cffi.requests.exceptions.SSLError", details)
        self.assertIn("code=77", details)
        self.assertIn("ca_bundle_unreadable", details)
        self.assertNotIn("private", details)

    def test_arbitrary_string_attributes_are_not_logged_and_cycles_stop(self):
        error = RuntimeError(PRIVATE)
        error.code = PRIVATE
        error.errno = True
        error.__context__ = error
        self.assertEqual(login_error_details(error), "error=builtins.RuntimeError")

    def test_environment_only_contains_versions_and_system_type(self):
        with patch.dict(os.environ, {"HTTPS_PROXY": PRIVATE, "SSL_CERT_FILE": PRIVATE}):
            environment = login_environment()
        for field in ("plugin=", "python=", "os=", "bilibili-api-python=", "curl_cffi=", "certifi="):
            self.assertIn(field, environment)
        self.assertNotIn("private", environment)
        self.assertNotIn("SSL_CERT_FILE", environment)


class LoginDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.logger = Mock()
        self.client = BiliClient(Path(self.temp.name) / "credential.json", logger=self.logger)
        self.addAsyncCleanup(self.client.close)
        await self.client.http.aclose()
        self.codes = [0]
        self.transport_error = None

        async def transport(request):
            if self.transport_error:
                raise self.transport_error
            code = self.codes.pop(0) if len(self.codes) > 1 else self.codes[0]
            headers = [("set-cookie", f"{key}={value}; Path=/; Domain=.bilibili.com; Secure")
                       for key, value in {"SESSDATA": "private-session", "bili_jct": "private-csrf",
                                          "DedeUserID": "1"}.items()] if code == 0 else []
            return httpx.Response(200, json={"code": 0, "data": {"code": code}}, headers=headers)

        self.client.http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        self.info = AsyncMock(return_value={"mid": 1, "name": "模拟账号"})
        for target, replacement in (("user.get_self_info", self.info),
                                    ("get_selected_client", Mock(return_value=("curl_cffi", object)))):
            patcher = patch("local_bilibili_autopilot.client." + target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def logs(self):
        return "\n".join(call.args[0] % call.args[1:] for call in self.logger.mock_calls)

    def assert_private_absent(self, text):
        for private in ("SESSDATA", "private-", "qrcode_key", "proxy.invalid", "cert.pem", self.temp.name):
            self.assertNotIn(private, text)

    async def test_generation_failure_identifies_stage_and_preserves_safe_ssl_hint(self):
        self.transport_error = httpx.ConnectError("SSL handshake failure " + PRIVATE)
        with self.assertRaises(LoginDiagnosticError) as raised:
            await self.client.begin_login()
        self.assertIn("阶段=生成二维码", str(raised.exception))
        self.assertIn("客户端=httpx", str(raised.exception))
        self.assertIn("tls_handshake_failed", str(raised.exception))
        self.assertIn("python=", self.logs())
        self.assert_private_absent(str(raised.exception) + self.logs())

    async def test_poll_failure_identifies_httpx_without_saving_credentials(self):
        self.transport_error = httpx.ReadTimeout(PRIVATE)
        with self.assertRaises(LoginDiagnosticError) as raised:
            await self.client.poll_login("private-qr")
        self.assertIn("阶段=轮询扫码状态", str(raised.exception))
        self.assertIn("客户端=httpx", str(raised.exception))
        self.info.assert_not_awaited()
        self.assertFalse(self.client.credential_path.exists())
        self.assert_private_absent(str(raised.exception) + self.logs())

    async def test_account_ssl_failure_identifies_selected_client_and_does_not_persist(self):
        self.info.side_effect = ssl.SSLError(1, "WRONG_VERSION_NUMBER " + PRIVATE)
        with self.assertRaises(LoginDiagnosticError) as raised:
            await self.client.poll_login("private-qr")
        self.assertIn("阶段=验证扫码账号", str(raised.exception))
        self.assertIn("客户端=curl_cffi", str(raised.exception))
        self.assertIn("wrong_tls_version", str(raised.exception))
        self.assertFalse(self.client.credential_path.exists())
        self.assertIsNone(self.client.credential)
        self.assertEqual(len(self.client.http.cookies), 0)
        self.logger.warning.assert_called_once()
        self.assert_private_absent(str(raised.exception) + self.logs())

    async def test_success_logs_state_changes_once_and_keeps_existing_login_behavior(self):
        self.codes = [86101, 86101, 86090, 86090, 0]
        with patch("local_bilibili_autopilot.client.asyncio.sleep", AsyncMock()):
            self.assertIn("已登录", await self.client.poll_login("private-qr"))
        logs = self.logs()
        for stage in ("等待扫码", "已扫码，等待确认", "验证扫码账号", "保存登录凭据", "登录完成"):
            self.assertEqual(logs.count("stage=" + stage + " client="), 1)
        self.assertEqual(self.client.uid, "1")
        self.assertTrue(self.client.credential_path.exists())
        self.logger.warning.assert_not_called()
        self.assert_private_absent(logs)

    async def test_persistence_failure_has_local_stage_and_preserves_old_file(self):
        self.client.credential_path.write_text("existing-private-login", encoding="utf-8")
        with patch("local_bilibili_autopilot.client.write_private_json", side_effect=OSError(13, PRIVATE)):
            with self.assertRaises(LoginDiagnosticError) as raised:
                await self.client.poll_login("private-qr")
        self.assertIn("阶段=保存登录凭据", str(raised.exception))
        self.assertIn("errno=13", str(raised.exception))
        self.assertEqual(self.client.credential_path.read_text(), "existing-private-login")
        self.assert_private_absent(str(raised.exception) + self.logs())

    async def test_platform_rejection_retains_platform_code(self):
        self.info.side_effect = ResponseCodeException(-412, PRIVATE)
        with self.assertRaises(PlatformError) as raised:
            await self.client.poll_login("private-qr")
        self.assertEqual(raised.exception.code, -412)
        self.assert_private_absent(str(raised.exception) + self.logs())

    async def test_cancellation_remains_cancellation(self):
        self.info.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.client.poll_login("private-qr")
        self.logger.warning.assert_not_called()

    async def test_background_returns_safe_stage_summary_and_hides_qr(self):
        self.info.side_effect = ssl.SSLError(1, "SSL handshake failure " + PRIVATE)
        plugin = BilibiliAutopilotPlugin()
        plugin.set_plugin_config({})
        plugin.client = self.client
        plugin._ctx = SimpleNamespace(logger=self.logger)
        plugin.engine = SimpleNamespace(pause_kind="initial", pause=AsyncMock(), lock=asyncio.Lock())
        plugin._respond = AsyncMock()
        with patch.object(self.client, "begin_login", AsyncMock(return_value=(b"fake-png", "private-qr"))):
            await plugin._background("login", "")
        self.assertFalse(plugin._last_operation_ok)
        self.assertIn("阶段=验证扫码账号", plugin._last_operation)
        self.assertEqual(plugin._login_qr, "")
        self.assert_private_absent(plugin._last_operation + self.logs())
