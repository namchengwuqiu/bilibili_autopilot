from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncio
import contextlib
import json
import unittest

from local_bilibili_autopilot.engine import Autopilot
from local_bilibili_autopilot.models import Evidence, IncomingReply, VideoItem
from local_bilibili_autopilot.plugin import BilibiliAutopilotPlugin
from local_bilibili_autopilot.state import StateStore
from local_bilibili_autopilot.webui import record_row


ITEM = VideoItem("BV1234567890", 1, "游戏音乐制作", "2", "测试UP", 120)
EVIDENCE = Evidence("字幕", "视频介绍了游戏配乐如何使用不同的旋律表达角色心境，还展示了主题音乐的变奏方法。")
DECISION = json.dumps({"should_comment": True, "interest": 0.9, "comment": "主题旋律随着角色心境变化，这段处理好有意思。",
                       "reason": "对配乐变奏有兴趣", "evidence_quote": "视频介绍了游戏配乐如何使用不同的旋律"}, ensure_ascii=False)


class WebUITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.plugin = BilibiliAutopilotPlugin()
        self.plugin.set_plugin_config({"plugin": {"config_version": "0.1.0"}, "comment": {"publish_enabled": True}})
        self.client = AsyncMock()
        self.client.uid, self.client.name = "1", "模拟账号"
        self.client.check_account.return_value = "1"
        self.client.recommended.return_value = [ITEM]
        self.client.understand.return_value = ITEM, EVIDENCE
        self.plugin.client = self.client
        self.plugin.engine = Autopilot(self.client, StateStore(Path(self.temp.name) / "state.sqlite3"),
                                        lambda: self.plugin.config, AsyncMock(return_value=DECISION),
                                        AsyncMock(return_value="喜欢游戏与音乐"),review=AsyncMock(return_value=json.dumps({"supported":True,"kind":"factual","reason":"模拟复核通过"})))
        await self.plugin.engine.initialize()
        self.plugin._ctx = SimpleNamespace(send=SimpleNamespace(text=AsyncMock(), image=AsyncMock()),
                                            logger=SimpleNamespace(warning=lambda *a: None))

    async def asyncTearDown(self):
        if self.plugin._job is not None:
            self.plugin._job.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.plugin._job

    async def test_polling_is_read_only_and_current_account_only(self):
        store = self.plugin.engine.store
        store.visit("2", ITEM.bvid, "preview", 1, "skipped", "别的账号的记录")
        overview = await self.plugin.autopilot_overview()
        records = await self.plugin.autopilot_records()
        self.assertEqual(records["rows"], [])
        self.assertIn("UID 1", overview["account"])
        self.client.check_account.assert_not_awaited()
        self.client.recommended.assert_not_awaited()
        self.plugin.engine.generate.assert_not_awaited()
        self.assertIsNone(self.plugin._job)
        self.assertNotIn("credential", json.dumps(overview))

    async def test_page_uses_current_maximum_for_previously_saved_long_wait(self):
        from local_bilibili_autopilot.webui import local_time
        await self.plugin.engine.start()
        now = self.plugin.engine.clock()
        self.plugin.engine.store.record_browse_decision("1", now, False, "休息", now + 7200)
        overview = await self.plugin.autopilot_overview()
        self.assertEqual(overview["next_check"], local_time(now + 1200, "Asia/Shanghai"))
        self.assertEqual(overview["check_range"], "10～20 分钟")

    async def test_paused_page_does_not_show_stale_next_check(self):
        self.plugin.engine.store.record_browse_decision("1", 1, False, "休息", 601)
        self.assertEqual((await self.plugin.autopilot_overview())["next_check"], "已暂停，恢复后重新检查")

    async def test_overdue_check_is_reported_as_waiting_in_page_and_status(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        self.plugin.engine.clock = lambda: datetime(2026, 10, 20, 12, tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
        await self.plugin.engine.start()
        now = self.plugin.engine.clock()
        self.plugin.engine.store.record_browse_decision('1', now - 1800, True, '去浏览', now - 600)
        self.assertEqual((await self.plugin.autopilot_overview())['next_check'], '已到检查时间，等待调度')
        self.assertIn('下次意愿检查：已到检查时间，等待调度', await self.plugin.engine.status())
        async with self.plugin.engine.lock:
            self.assertEqual((await self.plugin.autopilot_overview())['next_check'], '已到检查时间，等待当前任务完成')
        self.client.recommended.assert_not_awaited()
        self.plugin.engine.generate.assert_not_awaited()

    async def test_inactive_or_fixed_mode_does_not_display_old_autonomous_time(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        await self.plugin.engine.start()
        self.plugin.engine.store.record_browse_decision('1', 1, True, '去浏览', 601)
        self.plugin.engine.clock = lambda: datetime(2026, 10, 20, 2, tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
        self.assertEqual((await self.plugin.autopilot_overview())['next_check'], '当前不在活动时段，进入活动时段后检查')
        self.plugin.engine.clock = lambda: datetime(2026, 10, 20, 12, tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
        config = self.plugin.config.model_dump()
        config['browse']['autonomous_enabled'] = False
        self.plugin.set_plugin_config(config)
        self.assertEqual((await self.plugin.autopilot_overview())['next_check'], '按固定间隔浏览')

    async def test_web_preview_finishes_in_background_without_posting_or_chat_messages(self):
        result = await self.plugin.autopilot_command("/bili_auto preview")
        self.assertTrue(result["ok"])
        await self.plugin._job
        overview = await self.plugin.autopilot_overview()
        self.assertIn("预览评论", overview["operation"])
        self.assertIn("主题旋律", overview["operation"])
        self.client.publish.assert_not_awaited()
        self.plugin._ctx.send.text.assert_not_awaited()
        self.assertEqual((await self.plugin.autopilot_records())["rows"][0]["outcome"], "拟发表评论")

    async def test_received_reply_records_are_account_scoped_and_show_original_message(self):
        message=IncomingReply(ITEM.bvid,ITEM.aid,500,900,500,'2','对方','请问你说的是哪一段？','自己的原话','自己的原话',self.plugin.engine.clock())
        store=self.plugin.engine.store
        store.receive_replies('1',[message]);store.receive_replies('2',[message])
        store.decide_reply('1',900,'preview',self.plugin.engine.clock(),'would_reply','就是主题旋律那一段。','自然接话')
        data=await self.plugin.autopilot_records(kind='replies')
        self.assertTrue(data['is_reply'])
        self.assertEqual(data['pagination']['total'],1)
        self.assertEqual(data['rows'][0]['incoming'],'请问你说的是哪一段？')
        self.assertEqual(data['rows'][0]['outcome'],'拟跟进回复')
        self.assertEqual(data['rows'][0]['comment'],'就是主题旋律那一段。')
        self.assertEqual((await self.plugin.autopilot_records(kind='reply_attempts'))['rows'],[])
        self.client.incoming_replies.assert_not_awaited()

    async def test_replies_command_is_preview_only_and_does_not_send_chat_messages(self):
        self.assertTrue((await self.plugin.autopilot_command('/bili_auto replies'))['ok'])
        await self.plugin._job
        self.assertIn('暂无本插件',self.plugin._last_operation)
        self.client.publish_reply.assert_not_awaited()
        self.plugin._ctx.send.text.assert_not_awaited()

    async def test_duplicate_click_does_not_start_second_preview(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def delayed(*args):
            entered.set()
            await release.wait()
            return DECISION
        self.plugin.engine.generate.side_effect = delayed
        await self.plugin.autopilot_command("/bili_auto preview")
        task = self.plugin._job
        await entered.wait()
        self.assertFalse((await self.plugin.autopilot_command("/bili_auto preview"))["ok"])
        self.assertIs(self.plugin._job, task)
        self.assertTrue((await self.plugin.autopilot_overview())["busy"])
        release.set()
        await task
        self.plugin.engine.generate.assert_awaited_once()

    async def test_login_qr_is_shown_only_while_login_job_is_running(self):
        entered, release = asyncio.Event(), asyncio.Event()
        self.client.begin_login.return_value = b"fake png", "test-key"
        async def login(*args):
            entered.set()
            await release.wait()
            return "已登录 模拟账号"
        self.client.poll_login.side_effect = login
        self.assertTrue((await self.plugin.autopilot_command("/bili_auto login"))["ok"])
        await entered.wait()
        self.assertTrue((await self.plugin.autopilot_overview())["qr"].startswith("data:image/png;base64,"))
        self.plugin._ctx.send.image.assert_not_awaited()
        release.set()
        await self.plugin._job
        self.assertEqual((await self.plugin.autopilot_overview())["qr"], "")
        self.assertFalse(self.plugin.engine.paused)

    async def test_page_command_rejects_shell_and_confirmation_bypass(self):
        for command in ["ls", "/bili_auto start; touch /tmp/no", "/bili_auto logout",
                        "/bili_auto ack BV1234567890", "/bili_auto preview BV1234567890"]:
            with self.subTest(command=command):
                self.assertFalse((await self.plugin.autopilot_command(command))["ok"])
        self.assertIsNone(self.plugin._job)
        self.client.logout.assert_not_awaited()
        self.client.publish.assert_not_awaited()

    async def test_page_pause_persists_manual_intent(self):
        self.assertTrue((await self.plugin.autopilot_command("/bili_auto pause"))["ok"])
        self.assertEqual(self.plugin.engine.store.control()["pause_kind"], "manual")

    async def test_ack_is_account_scoped_and_does_not_resume_or_resend(self):
        store = self.plugin.engine.store
        store.set_pause(False, "")
        self.assertEqual(store.reserve("1", ITEM.bvid, "已发送但结果未知的评论", 100, 0, 3, 300, .88), "reserved")
        store.finish("1", ITEM.bvid, "unknown")
        await self.plugin.engine.pause("待核对")
        self.assertFalse((await self.plugin.autopilot_ack("bad"))["ok"])
        self.assertTrue((await self.plugin.autopilot_ack(ITEM.bvid))["ok"])
        self.assertEqual(store.stats("1", 0)["unresolved"], [])
        self.assertTrue(self.plugin.engine.paused)
        self.client.publish.assert_not_awaited()

    async def test_logout_runs_without_sending_chat_messages(self):
        result = await self.plugin.autopilot_logout()
        self.assertTrue(result["ok"])
        await self.plugin._job
        self.client.logout.assert_awaited_once()
        self.assertTrue(self.plugin.engine.paused)
        self.assertEqual(self.plugin.engine.pause_kind, "logout")
        self.plugin._ctx.send.text.assert_not_awaited()

    async def test_pagination_and_search_treat_wildcards_as_literal_text(self):
        store = self.plugin.engine.store
        for i in range(23):
            store.visit("1", f"BV{i:010}", "preview", i, "skipped",
                        json.dumps({"title": "100%_实测" if i == 0 else f"测试视频{i}", "reason": "路过"}, ensure_ascii=False))
        data = await self.plugin.autopilot_records(page=999)
        self.assertEqual(data["pagination"], {"page": 2, "pages": 2, "total": 23})
        self.assertEqual(len(data["rows"]), 3)
        self.assertEqual((await self.plugin.autopilot_records(search="%_"))["pagination"]["total"], 1)
        self.assertEqual((await self.plugin.autopilot_records(search="' OR 1=1 --"))["rows"], [])
        with self.assertRaises(ValueError):
            await self.plugin.autopilot_records(kind="meta")

    def test_record_labels_do_not_claim_drafts_were_sent(self):
        row = {"ts": 1, "bvid": ITEM.bvid, "mode": "preview", "outcome": "would_comment",
               "detail": json.dumps({"title": "视频", "comment": "拟写评论", "interest": .9, "reason": "感兴趣"})}
        self.assertEqual(record_row(row, "visits", "Asia/Shanghai")["outcome"], "拟发表评论")
        row.update(status="sent", text="实际提交的评论", rpid="9")
        self.assertEqual(record_row(row, "attempts", "Asia/Shanghai")["rpid"], "9")

    def test_deferred_record_keeps_title_draft_and_explains_wait(self):
        row = {'ts': 1, 'bvid': ITEM.bvid, 'mode': 'publish', 'outcome': 'deferred',
               'detail': json.dumps({'title': ITEM.title, 'comment': '待重新判断的评论', 'reason': '评论冷却中，最早12:45可再次评论'})}
        result = record_row(row, 'visits', 'Asia/Shanghai')
        self.assertEqual(result['outcome'], '候选保留中（评论冷却）')
        self.assertEqual(result['title'], ITEM.title)
        self.assertEqual(result['comment'], '待重新判断的评论')
        self.assertIn('12:45', result['reason'])
        row.update(outcome='blocked', detail=json.dumps({'title': ITEM.title, 'reason': 'daily_limit'}))
        self.assertEqual(record_row(row, 'visits', 'Asia/Shanghai')['reason'], '今日评论额度已满')


if __name__ == "__main__":
    unittest.main()
