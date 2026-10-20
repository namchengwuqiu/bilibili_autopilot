from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs

import asyncio
import json
import time
import unittest

from bilibili_api import Credential, video

import httpx

from local_bilibili_autopilot.client import BiliClient, PlatformError, PublishNotSent, UnknownPublish, subtitle_url
from local_bilibili_autopilot.config import AutopilotConfig
from local_bilibili_autopilot.models import IncomingReply, ModelResponseError, ModelTimeoutError, VideoItem
from local_bilibili_autopilot.plugin import BilibiliAutopilotPlugin
from local_bilibili_autopilot.state import StateStore


ITEM = VideoItem("BV1234567890", 100, "示例", "2", "UP", 60)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.client = BiliClient(Path(self.temp.name) / "credential.json")
        self.addAsyncCleanup(self.client.close)
        self.client.credential = Credential(sessdata="test-only", bili_jct="csrf-test", buvid3="device3", buvid4="device4", dedeuserid="1")
        self.client.uid = "1"
        self.calls = []
        self.status = 200
        self.payload = {"code": 0, "data": {"rpid": 9988}}
        self.failure = None
        async def transport(request):
            self.calls.append(request)
            if self.failure:
                raise self.failure
            return httpx.Response(self.status, json=self.payload)
        await self.client.http.aclose()
        self.client.http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        self.patches = [
            patch("bilibili_api.utils.network.get_wbi_mixin_key", AsyncMock(return_value="0123456789abcdef0123456789abcdef")),
            patch("bilibili_api.utils.network.get_buvid", AsyncMock(return_value=("device3", "device4"))),
            patch("bilibili_api.utils.network.get_bili_ticket", AsyncMock(return_value=("test-ticket", "9999999999"))),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    async def test_real_library_signing_and_single_post(self):
        self.assertEqual(await self.client.publish(ITEM, "测试内容", lambda: True), "9988")
        self.assertEqual(len(self.calls), 1)
        request = self.calls[0]
        self.assertEqual(request.method, "POST")
        self.assertIn("w_rid", request.url.params)
        self.assertIn("wts", request.url.params)
        body = parse_qs(request.content.decode())
        self.assertEqual(body["oid"], ["100"])
        self.assertEqual(body["type"], ["1"])
        self.assertEqual(body["csrf"], ["csrf-test"])
        self.assertNotIn("root", body)
        self.assertNotIn("parent", body)

    async def test_followup_signing_uses_thread_root_and_received_comment_parent(self):
        message=IncomingReply(ITEM.bvid,ITEM.aid,500,900,500,'2','其他用户','回复内容','原评论','原评论',time.time())
        self.assertEqual(await self.client.publish_reply(message,'跟进回复',lambda:True),'9988')
        self.assertEqual(len(self.calls),1)
        body=parse_qs(self.calls[0].content.decode())
        self.assertEqual(body['root'],['500'])
        self.assertEqual(body['parent'],['900'])
        self.assertEqual(body['oid'],['100'])
        self.assertIn('w_rid',self.calls[0].url.params)

    async def test_invalid_reply_parent_never_posts(self):
        for root,parent in [(None,900),(500,None),(0,900),(True,900)]:
            with self.subTest(root=root,parent=parent),self.assertRaises(PublishNotSent):
                await self.client.publish(ITEM,'回复',lambda:True,root=root,parent=parent)
        self.assertEqual(self.calls,[])

    async def test_received_feed_requires_known_target_matching_video_root_and_other_sender(self):
        bvid='BV1xx411c7mD'
        aid=video.Video(bvid=bvid).get_aid()
        now=time.time()
        own={'500':{'bvid':bvid,'root_rpid':500,'root_text':'自己的根评论','text':'自己的根评论'}}
        valid={'id':1,'user':{'mid':2,'nickname':'别人'},'reply_time':now,
               'item':{'business_id':1,'subject_id':aid,'root_id':500,'source_id':900,'target_id':500,'source_content':'认真回复麦麦'}}
        rows=[valid]
        for patch_data in [dict(target_id=999),dict(target_id=0),dict(subject_id=aid+1),dict(root_id=999),
                           dict(business_id=11),dict(source_id=0),dict(source_content='')]:
            rows.append({**valid,'item':{**valid['item'],**patch_data}})
        rows.extend([{**valid,'user':{'mid':1}},{**valid,'reply_time':now-1000000},valid])
        with patch('local_bilibili_autopilot.client.session.get_replies',AsyncMock(return_value={'items':rows,'cursor':{'is_end':True}})) as feed:
            result=await self.client.incoming_replies(own,now-72*3600)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0].source_rpid,900)
        self.assertEqual(result[0].root_text,'自己的根评论')
        self.assertEqual(result[0].sender_uid,'2')
        feed.assert_awaited_once()

    async def test_reply_feed_pagination_uses_documented_cursor_and_stops(self):
        bvid='BV1xx411c7mD';aid=video.Video(bvid=bvid).get_aid();now=time.time()
        own={'600':{'bvid':bvid,'root_rpid':500,'root_text':'根评论','text':'自己跟进过的回复'}}
        row={'user':{'mid':2,'nickname':'别人'},'reply_time':now,
             'item':{'business_id':1,'subject_id':aid,'root_id':500,'source_id':901,'target_id':600,'source_content':'继续回复麦麦'}}
        feed=AsyncMock(side_effect=[{'items':[row],'cursor':{'is_end':False,'id':100,'time':int(now)}},
                                   {'items':[],'cursor':{'is_end':True}}])
        with patch('local_bilibili_autopilot.client.session.get_replies',feed):
            result=await self.client.incoming_replies(own,now-72*3600)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0].target_rpid,600)
        self.assertEqual(feed.await_args.kwargs,{'last_reply_id':100,'reply_time':int(now)})

    async def test_wbi_rejection_is_never_retried(self):
        self.payload = {"code": -403, "message": "denied"}
        with self.assertRaises(PlatformError):
            await self.client.publish(ITEM, "测试内容", lambda: True)
        self.assertEqual(len(self.calls), 1)

    async def test_network_timeout_and_missing_rpid_are_unknown(self):
        self.failure = httpx.ReadTimeout("test timeout")
        with self.assertRaises(UnknownPublish):
            await self.client.publish(ITEM, "测试内容", lambda: True)
        self.assertEqual(len(self.calls), 1)
        self.failure = None
        self.payload = {"code": 0, "data": {}}
        with self.assertRaises(UnknownPublish):
            await self.client.publish(ITEM, "测试内容", lambda: True)

    async def test_pause_while_signing_prevents_post(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def sign(*args):
            entered.set()
            await release.wait()
            return "0123456789abcdef0123456789abcdef"
        enabled = True
        with patch("bilibili_api.utils.network.get_wbi_mixin_key", sign):
            task = asyncio.create_task(self.client.publish(ITEM, "测试内容", lambda: enabled))
            await entered.wait()
            enabled = False
            release.set()
            with self.assertRaises(PublishNotSent):
                await task
        self.assertEqual(self.calls, [])

    async def test_login_persists_private_credentials_and_logout_removes_them(self):
        async def qr_response(request):
            headers = [("set-cookie", f"{key}={value}; Path=/; Domain=.bilibili.com; Secure") for key, value in
                       {"SESSDATA": "test-only", "bili_jct": "csrf-test", "DedeUserID": "1"}.items()]
            return httpx.Response(200, json={"code": 0, "data": {"code": 0}}, headers=headers)
        await self.client.http.aclose()
        self.client.http = httpx.AsyncClient(transport=httpx.MockTransport(qr_response))
        with patch("local_bilibili_autopilot.client.user.get_self_info", AsyncMock(return_value={"mid": 1, "name": "测试"})):
            self.assertIn("已登录", await self.client.poll_login("test-key"))
        self.assertEqual(self.client.credential_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(json.loads(self.client.credential_path.read_text())["uid"], "1")
        self.assertEqual(len(self.client.http.cookies), 0)
        await self.client.logout()
        self.assertFalse(self.client.credential_path.exists())
        self.assertIsNone(self.client.credential)

    def test_subtitle_urls_reject_other_hosts_userinfo_and_http(self):
        for url in ["http://aisubtitle.hdslb.com/x", "https://127.0.0.1/x", "https://hdslb.com.attacker.test/x",
                    "https://user:secret@hdslb.com/x", "https://hdslb.com:8000/x"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                subtitle_url(url)
        self.assertEqual(subtitle_url("//aisubtitle.hdslb.com/x"), "https://aisubtitle.hdslb.com/x")


class PluginTests(unittest.IsolatedAsyncioTestCase):
    async def test_saved_login_starts_scheduler_without_start_command(self):
        with TemporaryDirectory() as folder:
            client = AsyncMock()
            client.uid, client.name = "1", "模拟账号"
            client.check_account.return_value = "1"
            ran = asyncio.Event()
            async def run_round():
                ran.set()
                return "这次暂时不刷"
            plugin = BilibiliAutopilotPlugin()
            plugin.set_plugin_config({})
            plugin._ctx = SimpleNamespace(paths=SimpleNamespace(data_dir=Path(folder)),
                                           chat=AsyncMock(), send=AsyncMock(),
                                           logger=SimpleNamespace(info=lambda *args: None, error=lambda *args: None))
            with patch("local_bilibili_autopilot.plugin.BiliClient", return_value=client), \
                    patch("local_bilibili_autopilot.engine.Autopilot.round", side_effect=run_round):
                await plugin.on_load()
                try:
                    await asyncio.wait_for(ran.wait(), timeout=1)
                    self.assertFalse(plugin.engine.paused)
                    client.check_account.assert_awaited_once()
                    client.publish.assert_not_awaited()
                finally:
                    await plugin.on_unload()

    async def test_saved_manual_pause_survives_scheduler_startup(self):
        with TemporaryDirectory() as folder:
            store = StateStore(Path(folder) / "state.sqlite3")
            store.initialize()
            store.set_pause(True, "管理员手动暂停", kind="manual")
            client = AsyncMock()
            client.uid = "1"
            plugin = BilibiliAutopilotPlugin()
            plugin.set_plugin_config({})
            plugin._ctx = SimpleNamespace(paths=SimpleNamespace(data_dir=Path(folder)),
                                           chat=AsyncMock(), send=AsyncMock(),
                                           logger=SimpleNamespace(info=lambda *args: None))
            with patch("local_bilibili_autopilot.plugin.BiliClient", return_value=client):
                await plugin.on_load()
                try:
                    await asyncio.sleep(0)
                    self.assertTrue(plugin.engine.paused)
                    client.check_account.assert_not_awaited()
                    client.recommended.assert_not_awaited()
                finally:
                    await plugin.on_unload()

    async def test_successful_login_enters_autonomous_mode(self):
        plugin = BilibiliAutopilotPlugin()
        plugin.set_plugin_config({})
        plugin.client = AsyncMock()
        plugin.client.begin_login.return_value = (b"fake qr", "test-key")
        plugin.client.poll_login.return_value = "已登录 模拟账号"
        plugin.engine = SimpleNamespace(lock=asyncio.Lock(), pause_kind="initial", pause=AsyncMock(),
                                        start=AsyncMock(return_value="已进入自主模式"))
        plugin._ctx = SimpleNamespace(send=SimpleNamespace(text=AsyncMock(), image=AsyncMock()))
        await plugin._background("login", "private")
        plugin.engine.start.assert_awaited_once_with(automatic=True)
        self.assertTrue(plugin._wake.is_set())
        self.assertIn("自主模式", plugin._ctx.send.text.await_args.args[0])

    async def test_relogin_does_not_override_manual_pause(self):
        plugin = BilibiliAutopilotPlugin()
        plugin.set_plugin_config({})
        plugin.client = AsyncMock()
        plugin.client.begin_login.return_value = (b"fake qr", "test-key")
        plugin.client.poll_login.return_value = "已登录 模拟账号"
        plugin.engine = SimpleNamespace(lock=asyncio.Lock(), pause_kind="manual", pause=AsyncMock(), start=AsyncMock())
        plugin._ctx = SimpleNamespace(send=SimpleNamespace(text=AsyncMock(), image=AsyncMock()))
        await plugin._background("login", "private")
        plugin.engine.start.assert_not_awaited()
        self.assertIn("手动暂停", plugin._ctx.send.text.await_args.args[0])

    async def test_generation_inherits_host_output_budget(self):
        plugin = BilibiliAutopilotPlugin()
        generate = AsyncMock(return_value={"success": True, "response": '{"should_comment": false}'})
        plugin._ctx = SimpleNamespace(llm=SimpleNamespace(generate=generate))
        cfg = AutopilotConfig()
        result = await plugin._generate("test prompt", cfg)
        self.assertEqual(result, '{"should_comment": false}')
        generate.assert_awaited_once_with("test prompt", model="replyer", timeout_ms=95000)

    async def test_model_timeout_switches_to_a_different_explicit_model(self):
        class RPCTimeout(RuntimeError):
            code = "E_TIMEOUT"

        for failure in [asyncio.TimeoutError(), httpx.ReadTimeout("PRIVATE"),
                        RPCTimeout("PRIVATE"),
                        {"success": False, "error": "APITimeoutError PRIVATE"}]:
            with self.subTest(failure=type(failure).__name__):
                plugin = BilibiliAutopilotPlugin()
                generate = AsyncMock(side_effect=[failure, {"success": True, "response": "valid"}])
                plugin._ctx = SimpleNamespace(llm=SimpleNamespace(generate=generate), logger=SimpleNamespace(warning=lambda *a: None))
                cfg = AutopilotConfig.model_validate({"model": {"candidates": ["first", "second", "first"]}})
                self.assertEqual(await plugin._generate("prompt", cfg), "valid")
                self.assertEqual([c.kwargs["model_name"] for c in generate.await_args_list], ["first", "second"])
                self.assertTrue(all(c.kwargs["task_name"] == "replyer" for c in generate.await_args_list))
                self.assertTrue(all("max_tokens" not in c.kwargs for c in generate.await_args_list))

    async def test_all_model_timeouts_are_bounded_and_private(self):
        plugin = BilibiliAutopilotPlugin()
        generate = AsyncMock(side_effect=asyncio.TimeoutError("PRIVATE"))
        plugin._ctx = SimpleNamespace(llm=SimpleNamespace(generate=generate), logger=SimpleNamespace(warning=lambda *a: None))
        cfg = AutopilotConfig.model_validate({"model": {"candidates": ["first", "second"]}})
        with self.assertRaises(ModelTimeoutError) as raised:
            await plugin._generate("prompt", cfg)
        self.assertEqual(generate.await_count, 2)
        self.assertNotIn("PRIVATE", str(raised.exception))

    async def test_per_model_deadline_allows_second_attempt(self):
        plugin = BilibiliAutopilotPlugin()
        cancelled = asyncio.Event()
        async def generate(prompt, **kwargs):
            if kwargs["model_name"] == "first":
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            return {"success": True, "response": "valid"}
        plugin._ctx = SimpleNamespace(llm=SimpleNamespace(generate=generate), logger=SimpleNamespace(warning=lambda *a: None))
        cfg = AutopilotConfig.model_validate({"model": {"candidates": ["first", "second"]}})
        cfg.model = cfg.model.model_copy(update={"timeout_seconds": .01})
        self.assertEqual(await plugin._generate("prompt", cfg), "valid")
        self.assertTrue(cancelled.is_set())

    async def test_cancellation_does_not_try_another_model(self):
        plugin = BilibiliAutopilotPlugin()
        generate = AsyncMock(side_effect=asyncio.CancelledError)
        plugin._ctx = SimpleNamespace(llm=SimpleNamespace(generate=generate))
        cfg = AutopilotConfig.model_validate({"model": {"candidates": ["first", "second"]}})
        with self.assertRaises(asyncio.CancelledError):
            await plugin._generate("prompt", cfg)
        self.assertEqual(generate.await_count, 1)

    async def test_reasoning_only_response_has_actionable_private_error(self):
        plugin = BilibiliAutopilotPlugin()
        plugin._ctx = SimpleNamespace(llm=SimpleNamespace(generate=AsyncMock(return_value={
            "success": True, "response": " ", "reasoning": "PRIVATE_MODEL_REASONING",
        })))
        with self.assertRaises(ModelResponseError) as raised:
            await plugin._generate("test prompt", AutopilotConfig())
        self.assertIn("最大输出 token", str(raised.exception))
        self.assertIn("没有评论正文", str(raised.exception))
        self.assertNotIn("PRIVATE_MODEL_REASONING", str(raised.exception))

    async def test_failed_and_empty_response_do_not_expose_provider_details(self):
        plugin = BilibiliAutopilotPlugin()
        for payload in [None, {"success": False, "error": "SESSDATA=secret"},
                        {"success": True, "response": ""}]:
            with self.subTest(payload=payload):
                plugin._ctx = SimpleNamespace(llm=SimpleNamespace(generate=AsyncMock(return_value=payload)))
                with self.assertRaises(ModelResponseError) as raised:
                    await plugin._generate("test prompt", AutopilotConfig())
                self.assertNotIn("secret", str(raised.exception))
                self.assertIn("模型", str(raised.exception))

    async def test_plain_qq_admin_roundtrips_and_remains_platform_scoped(self):
        plugin = BilibiliAutopilotPlugin()
        plugin.set_plugin_config({"plugin": {"config_version": "0.1.0", "admin_users": [
            " 12345678 ", "QQ: 12345678", " telegram:98765 ",
        ]}})
        self.assertEqual(plugin.config.plugin.admin_users, ["qq:12345678", "telegram:98765"])
        saved = plugin.get_plugin_config_data()
        self.assertEqual(saved["plugin"]["admin_users"], ["qq:12345678", "telegram:98765"])
        plugin.set_plugin_config(saved)
        self.assertTrue(plugin._authorized("qq", "12345678", False))
        self.assertFalse(plugin._authorized("telegram", "12345678", False))
        self.assertTrue(plugin._authorized("telegram", "98765", False))
        self.assertFalse(plugin._authorized("qq", "98765", False))

    async def test_malformed_admin_entries_still_rejected(self):
        for value in ["", " ", "nickname", "qq:", ":12345678", "qq: "]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                AutopilotConfig.model_validate({"plugin": {"admin_users": [value]}})

    async def test_commands_require_explicit_admin_or_local_operator(self):
        plugin = BilibiliAutopilotPlugin()
        plugin.set_plugin_config({"plugin": {"config_version": "0.1.0", "admin_users": ["qq:42"]}})
        self.assertTrue(plugin._authorized("qq", "42", False))
        self.assertFalse(plugin._authorized("other", "42", False))
        self.assertFalse(plugin._authorized("qq", "99", False))
        plugin.set_plugin_config({})
        self.assertFalse(plugin._authorized("qq", "42", False))
        self.assertTrue(plugin._authorized("", "", True))

    async def test_group_login_does_not_create_qr(self):
        plugin = BilibiliAutopilotPlugin()
        plugin.set_plugin_config({"plugin": {"config_version": "0.1.0", "admin_users": ["qq:42"]}})
        plugin.client, plugin.engine = AsyncMock(), AsyncMock()
        plugin._ctx = SimpleNamespace(send=SimpleNamespace(text=AsyncMock()))
        response = await plugin.command(stream_id="room", platform="qq", user_id="42", group_id="group",
                                        matched_groups={"action": "login"})
        self.assertFalse(response[0])
        plugin.client.begin_login.assert_not_awaited()

    async def test_load_unload_with_real_sdk_and_temporary_data(self):
        with TemporaryDirectory() as folder:
            plugin = BilibiliAutopilotPlugin()
            plugin.set_plugin_config({})
            plugin._ctx = SimpleNamespace(paths=SimpleNamespace(data_dir=Path(folder)),
                                           logger=SimpleNamespace(info=lambda *args: None),
                                           chat=AsyncMock(),
                                           send=SimpleNamespace(text=AsyncMock()))
            await plugin.on_load()
            try:
                self.assertTrue(plugin.engine.paused)
                self.assertFalse(plugin.config.comment.publish_enabled)
                self.assertTrue((Path(folder) / "state.sqlite3").exists())
                self.assertFalse((Path(folder) / "credential.json").exists())
                self.assertTrue(plugin.get_components())
                self.assertTrue(plugin.get_webui_config_schema())
            finally:
                await plugin.on_unload()


if __name__ == "__main__":
    unittest.main()
