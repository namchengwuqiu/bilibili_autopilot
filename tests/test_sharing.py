from pathlib import Path
from dataclasses import replace
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncio
import json
import unittest

from local_bilibili_autopilot.config import AutopilotConfig
from local_bilibili_autopilot.models import DecisionValidationError, ModelTimeoutError, VideoItem, parse_share_text, share_prompt
from local_bilibili_autopilot.plugin import BilibiliAutopilotPlugin
from local_bilibili_autopilot.sharing import VideoSharer
from local_bilibili_autopilot.state import StateStore
from local_bilibili_autopilot.webui import record_row


ITEM = VideoItem("BV1234567890", 100, "游戏音乐：新的主题旋律", "2002", "测试UP", 180)
GROUP = "123456789"
STREAM = {"session_id": "real-session", "platform": "qq", "group_id": GROUP,
          "chat_type": "group", "account_id": "test-bot", "group_name": "测试群"}
INTRO = "这段旋律一变，角色那股劲儿就出来了"
CARD = {"app": "com.tencent.miniapp_01", "meta": {"detail_1": {"desc": ITEM.title, "title": "哔哩哔哩"}}}


class SharingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = StateStore(Path(self.temp.name) / "state.sqlite3")
        self.store.initialize(); self.store.set_pause(False, "")
        self.cfg = AutopilotConfig()
        self.cfg.share.qq_group_ids = [GROUP]
        self.chat, self.send = AsyncMock(), AsyncMock()
        self.chat.get_group_streams.return_value = [STREAM]
        self.send.text.return_value = {"sent": True, "message_id": "msg-1"}
        self.allowed = True
        self.compose = AsyncMock(return_value=INTRO)
        self.sharer = VideoSharer(self.store, lambda: self.cfg, self.chat, self.send, lambda: self.allowed,
                                  self.compose, lambda: 1000)

    def comment_sent(self, status="sent"):
        self.assertEqual(self.store.reserve("1", ITEM.bvid, "测试原评论", 900, 0, 20, 0, .88), "reserved")
        self.store.finish("1", ITEM.bvid, status, "12345" if status == "sent" else "")

    async def test_default_empty_configuration_does_not_lookup_or_send(self):
        self.assertEqual(AutopilotConfig().share.qq_group_ids, [])
        self.cfg.share.qq_group_ids = []
        self.comment_sent()
        self.assertEqual(await self.sharer.share("1", ITEM), "")
        self.chat.get_group_streams.assert_not_awaited()
        self.send.text.assert_not_awaited()
        self.compose.assert_not_awaited()

    def enable_card(self):
        self.cfg.share.miniapp_enabled = True
        self.sharer.compose_card = AsyncMock(return_value=json.dumps(CARD, ensure_ascii=False))
        self.send.custom.return_value = {"sent": True, "message_id": "card-1"}

    async def test_card_is_confirmed_and_saved_before_standalone_intro(self):
        self.enable_card(); self.comment_sent()
        order = []
        async def card(*args, **kwargs):
            order.append("card")
            return {"sent": True, "message_id": "card-1"}
        async def intro(*args, **kwargs):
            order.append("intro")
            row = self.store.latest_share("1")
            self.assertEqual((row["card_sent"], row["card_message_id"]), (1, "card-1"))
            self.assertEqual(row["status"], "pending")
            return {"sent": True, "message_id": "intro-1"}
        self.send.custom.side_effect, self.send.text.side_effect = card, intro
        self.assertIn("已分享至群聊", await self.sharer.share("1", ITEM))
        self.assertEqual(order, ["card", "intro"])
        self.send.custom.assert_awaited_once_with("json", json.dumps(CARD, ensure_ascii=False), "real-session",
                                                return_details=True, processed_plain_text=f"[B站小程序] {ITEM.title}")
        self.send.text.assert_awaited_once_with(INTRO, "real-session", return_details=True)
        row = self.store.latest_share("1")
        self.assertEqual(row["message_id"], "intro-1")
        self.assertIn("小程序卡片已发送", record_row(row, "shares", "Asia/Shanghai")["reason"])
        self.assertEqual(self.store.share_context("1", ITEM.bvid)["recent_shares"], [INTRO])
        self.store.initialize()
        await self.sharer.share("1", ITEM)
        self.assertEqual(order, ["card", "intro"])

    async def test_failed_or_unknown_card_never_sends_intro_or_retries(self):
        self.enable_card(); self.comment_sent()
        self.send.custom.return_value = {"sent": False}
        self.assertIn("未发送分享感想", await self.sharer.share("1", ITEM))
        self.assertEqual(self.store.latest_share("1")["status"], "failed")
        await self.sharer.share("1", ITEM)
        self.send.custom.assert_awaited_once()
        self.send.text.assert_not_awaited()

    async def test_unknown_card_stops_before_intro(self):
        self.enable_card(); self.comment_sent()
        self.send.custom.return_value = {"unexpected": True}
        self.assertIn("结果待核对", await self.sharer.share("1", ITEM))
        self.send.text.assert_not_awaited()
        self.assertEqual(self.store.latest_share("1")["status"], "unknown")

    async def test_card_timeout_stops_before_intro(self):
        self.enable_card(); self.comment_sent()
        self.send.custom.side_effect = asyncio.TimeoutError()
        await self.sharer.share("1", ITEM)
        self.send.text.assert_not_awaited()
        self.assertEqual(self.store.latest_share("1")["status"], "unknown")

    async def test_intro_failure_keeps_card_receipt_and_is_not_replayed(self):
        self.enable_card(); self.comment_sent()
        self.send.text.return_value = {"sent": False}
        self.assertIn("仅小程序卡片已发送", await self.sharer.share("1", ITEM))
        row = self.store.latest_share("1")
        self.assertEqual((row["status"], row["card_message_id"], row["message_id"]), ("partial", "card-1", "card-1"))
        self.store.initialize()
        await self.sharer.share("1", ITEM)
        self.send.custom.assert_awaited_once()
        self.send.text.assert_awaited_once()

    async def test_intro_timeout_keeps_card_receipt_as_unknown(self):
        self.enable_card(); self.comment_sent()
        self.send.text.side_effect = asyncio.TimeoutError()
        self.assertIn("小程序卡片已发送", await self.sharer.share("1", ITEM))
        row = self.store.latest_share("1")
        self.assertEqual((row["status"], row["card_sent"]), ("unknown", 1))
        await self.sharer.share("1", ITEM)
        self.send.custom.assert_awaited_once()
        self.send.text.assert_awaited_once()

    async def test_pause_after_card_stops_intro(self):
        self.enable_card(); self.comment_sent()
        async def pause(*args, **kwargs):
            self.allowed = False
            return {"sent": True, "message_id": "card-1"}
        self.send.custom.side_effect = pause
        await self.sharer.share("1", ITEM)
        self.send.text.assert_not_awaited()
        self.assertEqual(self.store.latest_share("1")["status"], "partial")

    async def test_config_change_after_card_stops_intro(self):
        self.enable_card(); self.comment_sent()
        async def change(*args, **kwargs):
            self.cfg.share.qq_group_ids = []
            return {"sent": True, "message_id": "card-1"}
        self.send.custom.side_effect = change
        await self.sharer.share("1", ITEM)
        self.send.text.assert_not_awaited()
        self.assertEqual(self.store.latest_share("1")["status"], "partial")

    async def test_interruption_after_card_retains_receipt_without_replay(self):
        self.enable_card(); self.comment_sent()
        self.send.text.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.sharer.share("1", ITEM)
        self.store.initialize()
        row = self.store.latest_share("1")
        self.assertEqual((row["status"], row["card_message_id"]), ("unknown", "card-1"))
        self.assertIn("卡片已发送", row["detail"])
        await self.sharer.share("1", ITEM)
        self.send.custom.assert_awaited_once()

    async def test_card_generation_failure_never_sends_messages(self):
        self.enable_card(); self.comment_sent()
        self.sharer.compose_card.side_effect = ValueError("bad card")
        self.assertIn("小程序卡片生成失败", await self.sharer.share("1", ITEM))
        self.send.custom.assert_not_awaited()
        self.send.text.assert_not_awaited()

    async def test_multiple_groups_receive_card_then_intro_with_one_draft(self):
        self.enable_card(); self.comment_sent()
        self.cfg.share.qq_group_ids.append("123456")
        self.chat.get_group_streams.return_value = [STREAM, {**STREAM, "group_id": "123456", "session_id": "second"}]
        order = []
        async def card(kind, data, stream, **kwargs):
            order.append((stream, "card"))
            return {"sent": True, "message_id": f"card-{stream}"}
        async def intro(text, stream, **kwargs):
            order.append((stream, "intro"))
            return {"sent": True, "message_id": f"intro-{stream}"}
        self.send.custom.side_effect, self.send.text.side_effect = card, intro
        await self.sharer.share("1", ITEM)
        self.assertEqual(order, [("real-session", "card"), ("real-session", "intro"),
                                 ("second", "card"), ("second", "intro")])
        self.compose.assert_awaited_once()
        self.sharer.compose_card.assert_awaited_once()

    def test_legacy_share_schema_upgrade_preserves_existing_record(self):
        legacy = StateStore(Path(self.temp.name) / "legacy.sqlite3")
        with legacy.connect() as db:
            db.execute("CREATE TABLE video_shares (uid TEXT,bvid TEXT,group_id TEXT,ts REAL,title TEXT,status TEXT,"
                       "detail TEXT DEFAULT '',message_id TEXT DEFAULT '',PRIMARY KEY(uid,bvid,group_id))")
            db.execute("INSERT INTO video_shares VALUES (?,?,?,?,?,'sent',?,?)",
                       ("1", ITEM.bvid, GROUP, 900, ITEM.title, f"分享文案：{INTRO}", "old-message"))
        legacy.initialize()
        legacy.initialize()
        row = legacy.latest_share("1")
        self.assertEqual((row["status"], row["message_id"], row["detail"]), ("sent", "old-message", f"分享文案：{INTRO}"))
        self.assertEqual((row["card_sent"], row["card_message_id"]), (0, ""))

    async def test_documented_snowluma_and_napcat_card_responses(self):
        plugin = BilibiliAutopilotPlugin()
        item = replace(ITEM, cover="https://i0.hdslb.com/bfs/archive/cover.jpg")
        api = AsyncMock()
        plugin._ctx = SimpleNamespace(api=api)
        expected = {**CARD, "meta": {"detail_1": {**CARD["meta"]["detail_1"],
                    "qqdocurl": f"https://www.bilibili.com/video/{ITEM.bvid}", "scene": 1036}}}
        for data in (CARD, {"ark": json.dumps(CARD)}):
            api.call.return_value = {"status": "ok", "retcode": 0, "data": {"data": data}}
            self.assertEqual(json.loads(await plugin._compose_card(item)), expected)
        api.call.assert_awaited_with("adapter.napcat.account.get_mini_app_ark", version="1",
                                    params={"type": "bili", "title": "哔哩哔哩", "desc": ITEM.title,
                                            "picUrl": item.cover, "jumpUrl": f"https://www.bilibili.com/video/{ITEM.bvid}"})
        for response in ({"status": "failed", "retcode": 1}, {"status": "ok", "retcode": 0, "data": {}},
                         {"status": "ok", "retcode": 0, "data": {"data": {"app": "other", "meta": {}}}},
                         {"status": "ok", "retcode": 0, "data": {"data": {"app": "com.tencent.miniapp_01", "meta": {"detail_1": []}}}}):
            api.call.return_value = response
            with self.assertRaises(ValueError):
                await plugin._compose_card(item)

    async def test_card_web_target_patch_preserves_native_address_and_signature(self):
        plugin = BilibiliAutopilotPlugin()
        item = replace(ITEM, cover="https://i0.hdslb.com/bfs/archive/cover.jpg")
        card = {**CARD, "config": {"token": "original-signature", "ctime": 1000},
                "miniappShareOrigin": 3, "miniappOpenRefer": "10002",
                "meta": {"detail_1": {**CARD["meta"]["detail_1"], "appid": "1109937557", "scene": 0,
                                      "qqdocurl": "", "url": "m.q.qq.com/a/s/original-native-address"}}}
        original = json.loads(json.dumps(card))
        api = AsyncMock()
        api.call.return_value = {"status": "ok", "retcode": 0, "data": {"data": card}}
        plugin._ctx = SimpleNamespace(api=api)
        patched = json.loads(await plugin._compose_card(item))
        expected = json.loads(json.dumps(original))
        expected["meta"]["detail_1"].update(qqdocurl=f"https://www.bilibili.com/video/{ITEM.bvid}", scene=1036)
        self.assertEqual(patched, expected)
        self.assertEqual(card, original)
        self.assertNotIn("verType", patched["meta"]["detail_1"])
        self.assertNotIn("versionId", patched["meta"]["detail_1"])

    async def test_card_preserves_existing_web_target_and_nonzero_scene(self):
        plugin = BilibiliAutopilotPlugin()
        item = replace(ITEM, cover="https://i0.hdslb.com/bfs/archive/cover.jpg")
        card = {**CARD, "meta": {"detail_1": {**CARD["meta"]["detail_1"], "scene": 1037,
                                             "qqdocurl": "https://b23.tv/original-share"}}}
        api = AsyncMock()
        api.call.return_value = {"status": "ok", "retcode": 0, "data": {"data": {"ark": json.dumps(card)}}}
        plugin._ctx = SimpleNamespace(api=api)
        self.assertEqual(json.loads(await plugin._compose_card(item)), card)

    async def test_success_uses_real_existing_group_and_deduplicates_after_restart(self):
        self.comment_sent()
        # 相同群号的无账号占位流不能抢走真实路由。
        self.chat.get_group_streams.return_value.insert(0, {**STREAM,"session_id":"placeholder","account_id":None})
        self.assertIn("已分享至群聊", await self.sharer.share("1", ITEM))
        self.send.text.assert_awaited_once_with(
            f"{INTRO}\nhttps://www.bilibili.com/video/{ITEM.bvid}", "real-session", return_details=True)
        self.store.initialize()
        await self.sharer.share("1", ITEM)
        self.send.text.assert_awaited_once()
        self.compose.assert_awaited_once()
        self.assertEqual(self.store.latest_share("1")["message_id"], "msg-1")
        self.assertIsNone(self.store.latest_share("2"))
        records = self.store.page_records("1", "shares", GROUP, 1)
        row = record_row(records["rows"][0], "shares", "Asia/Shanghai")
        self.assertEqual(row["outcome"], "已分享至群聊")
        self.assertIn(GROUP, row["reason"])
        self.assertEqual(self.store.page_records("2", "shares", "", 1)["rows"], [])

    async def test_unconfirmed_or_untracked_comment_cannot_trigger_share(self):
        await self.sharer.share("1", ITEM)
        self.comment_sent("unknown")
        await self.sharer.share("1", ITEM)
        self.send.text.assert_not_awaited()
        self.compose.assert_not_awaited()
        self.assertIsNone(self.store.latest_share("1"))

    async def test_false_send_result_is_failure_and_never_retried(self):
        self.comment_sent(); self.send.text.return_value = {"sent": False, "message_id": None}
        self.assertIn("群分享失败", await self.sharer.share("1", ITEM))
        await self.sharer.share("1", ITEM)
        self.send.text.assert_awaited_once()
        self.assertEqual(self.store.latest_share("1")["status"], "failed")
        self.assertEqual(self.store.own_comments("1")["12345"]["bvid"], ITEM.bvid)

    async def test_timeout_is_unknown_without_retry_or_pausing_browsing(self):
        self.comment_sent(); self.send.text.side_effect = asyncio.TimeoutError()
        self.assertIn("结果待核对", await self.sharer.share("1", ITEM))
        await self.sharer.share("1", ITEM)
        self.send.text.assert_awaited_once()
        self.assertEqual(self.store.latest_share("1")["status"], "unknown")
        self.assertEqual(self.store.control()["paused"], "0")

    async def test_missing_or_ambiguous_group_is_recorded_without_sending(self):
        for streams in ([], [STREAM, {**STREAM,"session_id":"second","account_id":"other-bot"}],
                        [{**STREAM,"platform":"other"}], [{**STREAM,"chat_type":"private"}]):
            with self.subTest(streams=streams):
                self.chat.get_group_streams.return_value = streams
                with self.assertRaises(ValueError):
                    await self.sharer.resolve(GROUP)
        self.comment_sent()
        self.assertIn("群会话查找失败", await self.sharer.share("1", ITEM))
        self.send.text.assert_not_awaited()
        self.assertEqual(self.store.latest_share("1")["status"], "failed")

    async def test_pause_or_removed_target_during_lookup_cancels_share(self):
        self.comment_sent()
        async def remove(**kwargs):
            self.cfg.share.qq_group_ids = []
            return [STREAM]
        self.chat.get_group_streams.side_effect = remove
        self.assertIn("已取消", await self.sharer.share("1", ITEM))
        self.send.text.assert_not_awaited()

    async def test_concurrent_reservations_only_send_once(self):
        self.comment_sent()
        await asyncio.gather(self.sharer.share("1", ITEM), self.sharer.share("1", ITEM))
        self.send.text.assert_awaited_once()
        self.compose.assert_awaited_once()

    async def test_model_timeout_skips_share_without_pausing_or_reposting_comment(self):
        self.comment_sent()
        self.compose.side_effect = ModelTimeoutError("PRIVATE")
        self.assertIn("分享文案生成失败", await self.sharer.share("1", ITEM))
        self.send.text.assert_not_awaited()
        self.assertEqual(self.store.control()["paused"], "0")
        self.assertEqual(self.store.latest_share("1")["status"], "failed")
        self.assertNotIn("PRIVATE", self.store.latest_share("1")["detail"])
        self.assertIn("12345", self.store.own_comments("1"))

    async def test_pause_during_composition_cancels_send(self):
        self.comment_sent()
        async def compose(*args):
            self.allowed = False
            return INTRO
        self.compose.side_effect = compose
        self.assertIn("已取消", await self.sharer.share("1", ITEM))
        self.send.text.assert_not_awaited()

    async def test_validation_failure_displays_reason_without_sending_or_retrying(self):
        self.comment_sent()
        self.compose.side_effect = DecisionValidationError("unsupported_share", "摘录未提供祠堂地点")
        result = await self.sharer.share("1", ITEM)
        self.assertIn("[unsupported_share]", result)
        self.assertIn("摘录未提供祠堂地点", result)
        await self.sharer.share("1", ITEM)
        self.compose.assert_awaited_once()
        self.send.text.assert_not_awaited()
        self.assertEqual(self.store.latest_share("1")["status"], "failed")

    async def test_configuration_change_during_composition_cancels_send(self):
        self.comment_sent()
        async def compose(*args):
            self.cfg.model.extra_style = "改了配置"
            return INTRO
        self.compose.side_effect = compose
        self.assertIn("已取消", await self.sharer.share("1", ITEM))
        self.send.text.assert_not_awaited()

    async def test_multiple_groups_share_one_composition(self):
        self.comment_sent()
        self.cfg.share.qq_group_ids.append("123456")
        self.chat.get_group_streams.return_value = [STREAM, {**STREAM, "session_id": "other", "group_id": "123456"}]
        await self.sharer.share("1", ITEM)
        self.assertEqual(self.send.text.await_count, 2)
        self.compose.assert_awaited_once()

    async def test_context_contains_confirmed_comment_evidence_and_recent_wording(self):
        self.comment_sent()
        self.store.visit("1", ITEM.bvid, "publish", 900, "sent", json.dumps({
            "reason": "配乐变奏", "evidence_quote": "主题旋律在角色出场后发生变化"}))
        await self.sharer.share("1", ITEM)
        context = self.store.share_context("1", ITEM.bvid)
        self.assertEqual(context["my_comment"], "测试原评论")
        self.assertEqual(context["content_evidence"], "主题旋律在角色出场后发生变化")
        self.assertEqual(context["recent_shares"], [INTRO])
        with self.assertRaises(ValueError):
            self.store.share_context("2", ITEM.bvid)

    def test_share_prompt_uses_persona_and_grounded_details_without_fixed_intro(self):
        prompt = share_prompt(ITEM, {"my_comment": "测试原评论", "content_evidence": "配乐变奏"},
                              persona="随和，爱玩游戏", extra_style="简短")
        self.assertIn("随和，爱玩游戏", prompt[0]["content"])
        self.assertIn("不要每次都说", prompt[0]["content"])
        self.assertEqual(json.loads(prompt[1]["content"])["my_comment"], "测试原评论")

    def test_share_text_accepts_short_chat_and_rejects_links_or_explanations(self):
        self.assertEqual(parse_share_text(json.dumps({"text": INTRO})), INTRO)
        for payload in [{"text": ""}, {"text": "x" * 181}, {"text": "看这个https://example.com"},
                        {"text": "@全体成员"}, {"text": "看看BV1234567890"}, {"text": "分成\n两行"},
                        {"text": INTRO, "reason": "多余解释"}, {"text": 3}]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                parse_share_text(json.dumps(payload))

    def test_share_text_preserves_richer_intro_up_to_limit(self):
        intro = "这个视频讲的是深海生物怎么用发光器官吸引猎物，还提到不同颜色和闪烁频率能传递信号。吸引我的就是同样是发光，却能有不同用途这点，单看这个介绍就挺有意思的，想把这个话题丢来聊聊。"
        self.assertGreater(len(intro), 80)
        self.assertEqual(parse_share_text(json.dumps({"text": intro})), intro)
        self.assertEqual(parse_share_text(json.dumps({"text": "好" * 180})), "好" * 180)

    async def test_real_sdk_unwraps_host_group_stream_response(self):
        from types import SimpleNamespace
        from maibot_sdk.capabilities.chat import ChatCapability
        from maibot_sdk.context import PluginContext
        async def call(capability, **kwargs):
            return PluginContext._normalize_capability_result(capability, {"success": True, "streams": [STREAM]})
        self.sharer.chat = ChatCapability(SimpleNamespace(call_capability=call))
        resolved = await self.sharer.resolve(GROUP)
        self.assertEqual(resolved["session_id"], "real-session")

    async def test_interrupted_share_becomes_unknown_and_is_never_replayed(self):
        self.comment_sent()
        self.assertTrue(self.store.reserve_share("1",ITEM.bvid,GROUP,1000,ITEM.title))
        self.store.initialize()
        self.assertEqual(self.store.latest_share("1")["status"], "unknown")
        await self.sharer.share("1",ITEM)
        self.send.text.assert_not_awaited()

    def test_group_configuration_normalizes_duplicates_and_rejects_invalid_targets(self):
        cfg = AutopilotConfig.model_validate({"share":{"qq_group_ids":[" 123456789 ",GROUP]}})
        self.assertEqual(cfg.share.qq_group_ids,[GROUP])
        for group in ("", "qq:123456789", "１２３４５６７８９", "../12345", "012345"):
            with self.subTest(group=group), self.assertRaises(ValueError):
                AutopilotConfig.model_validate({"share":{"qq_group_ids":[group]}})
