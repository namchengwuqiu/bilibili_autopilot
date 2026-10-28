"""候选池与评论机会调度；平台、模型及发送均为替身，不连接真实账号。"""

from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import asyncio
import json
import unittest

from local_bilibili_autopilot.config import AutopilotConfig
from local_bilibili_autopilot.engine import Autopilot
from local_bilibili_autopilot.models import DecisionValidationError, Evidence, VideoItem, parse_decision
from local_bilibili_autopilot.state import StateStore
from local_bilibili_autopilot.webui import record_row


ITEM = VideoItem("BV1234567890", 100, "深海的光", "2", "UP", 180)
EVIDENCE = Evidence("字幕", "深海生物利用发光器官吸引猎物，不同颜色与闪烁频率也能传递信号，这是海洋生态的一种适应方式。")


def answer(**changes):
    return json.dumps({"should_comment": True, "interest": .9, "comment": "这光还能传信，挺有意思",
                       "reason": "对发光方式感兴趣", "evidence_quote": "深海生物利用发光器官吸引猎物",
                       **changes}, ensure_ascii=False)


class CandidateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.now = datetime(2026, 10, 28, 12, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
        self.cfg = AutopilotConfig()
        self.cfg.comment.publish_enabled = True
        self.cfg.browse.autonomous_enabled = False
        self.cfg.browse.active_start_hour = self.cfg.browse.active_end_hour = 0
        self.cfg.replies.enabled = False
        self.client = AsyncMock()
        self.client.uid, self.client.name = "1", "模拟账号"
        self.client.check_account.return_value = "1"
        self.client.recommended.return_value = [ITEM]
        self.client.understand.return_value = ITEM, EVIDENCE
        self.client.publish.return_value = "500"
        self.selected = ITEM.bvid
        self.want_comment = True

        async def model(prompt, cfg):
            system = prompt[0]["content"]
            if "[autopilot:select_candidate]" in system:
                return json.dumps({"selected_bvid": self.selected, "reason": "结合兴趣重新考虑"})
            if "[autopilot:candidate]" in system:
                return answer(comment="")
            return answer(should_comment=self.want_comment, comment="这光还能传信，挺有意思" if self.want_comment else "")

        self.generate = AsyncMock(side_effect=model)
        self.review = AsyncMock(return_value=json.dumps({"supported": True, "kind": "factual", "reason": "有依据"}))
        self.store = StateStore(Path(self.temp.name) / "state.sqlite3")
        self.engine = Autopilot(self.client, self.store, lambda: self.cfg, self.generate,
                                AsyncMock(return_value="好奇"), lambda: self.now, review=self.review)
        await self.engine.initialize()
        await self.engine.start()

    def cooling(self):
        self.store.reserve("1", "BV0987654321", "已有评论", self.now, 0, 12, 0, .88)
        self.store.finish("1", "BV0987654321", "sent", "400")

    def save(self, item=ITEM, *, interest=.9, ts=None, retry=None, hours=48, limit=24):
        self.store.wait_video("1", item, self.now if ts is None else ts,
                              json.dumps({"video": asdict(item), "interest": interest, "comment": "",
                                          "reason": "感兴趣", "evidence_quote": "深海生物利用发光器官吸引猎物"}),
                              self.now if retry is None else retry, hours, limit, "deferred")

    async def test_cooling_only_screens_interest_without_draft_review_or_send_intent(self):
        self.cooling()
        self.assertIn("候选保留中", await self.engine.round())
        self.generate.assert_awaited_once()
        self.review.assert_not_awaited()
        self.client.publish.assert_not_awaited()
        rows = self.store.candidates("1", now=self.now, ready_only=False)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["fields"]["comment"], "")
        self.assertEqual(rows[0]["retry_at"], self.now + 2700)
        self.assertEqual(self.store.stats("1", 0)["attempts"], 1)

    async def test_ready_candidate_runs_during_browse_rest_and_without_recommendations(self):
        self.save()
        self.cfg.browse.autonomous_enabled = True
        self.store.record_browse_decision("1", self.now, False, "浏览先休息", self.now + 1200)
        self.assertEqual(await self.engine.next_check_delay(), 1)
        self.assertIn("已提交评论", await self.engine.round())
        self.client.recommended.assert_not_awaited()
        self.client.understand.assert_awaited_once()
        self.review.assert_awaited_once()
        self.client.publish.assert_awaited_once()
        self.assertFalse(self.store.latest_browse_decision("1")["should_browse"])

    async def test_scheduler_wakes_at_cooldown_boundary(self):
        self.cooling()
        self.save(retry=self.now + 2700)
        self.cfg.browse.autonomous_enabled = True
        self.store.record_browse_decision("1", self.now, False, "休息", self.now + 1200)
        self.now += 2690
        self.assertEqual(await self.engine.next_check_delay(), 10)
        self.now += 10
        self.assertIn("已提交评论", await self.engine.round())

    async def test_account_read_backoff_does_not_spin_on_ready_candidates(self):
        self.save()
        self.store.read_failed("1", "account", self.now, "临时连接失败", 60)
        self.assertEqual(await self.engine.next_check_delay(), 60)
        await self.engine.round()
        self.generate.assert_not_awaited()

    async def test_recommendation_backoff_does_not_delay_comment_opportunity(self):
        self.cooling()
        self.save(retry=self.now + 2700)
        self.now += 2690
        self.store.read_failed("1", "recommendations", self.now, "临时连接失败", 60)
        self.assertEqual(await self.engine.next_check_delay(), 10)

    async def test_old_quota_candidate_does_not_reenter_browsing_after_declining(self):
        self.save()
        with self.store.connect() as db:
            db.execute("UPDATE visits SET outcome='quota_wait'")
        self.selected = ""
        await self.engine.round()
        await self.engine.round()
        self.assertEqual(self.generate.await_count, 1)
        self.client.understand.assert_not_awaited()

    async def test_declining_preserves_pool_and_throttle_survives_restart(self):
        self.save()
        self.selected = ""
        self.assertIn("本次未选中", await self.engine.round())
        row = self.store.page_records("1", "visits", "", 1)["rows"][0]
        self.assertEqual(record_row(row, "visits", "Asia/Shanghai")["outcome"], "本次未选中，仍保留候选")
        self.assertEqual(self.store.stats("1", 0)["views"], 1)
        self.client.understand.assert_not_awaited()
        await self.engine.initialize()
        await self.engine.round()
        self.assertEqual(self.generate.await_count, 1)
        self.assertEqual(await self.engine.next_check_delay(), 600)
        self.client.publish.assert_not_awaited()

    async def test_selection_outside_pool_is_rejected_without_reading_or_sending(self):
        self.save()
        self.selected = "BV0000000000"
        self.assertIn("候选选择未完成", await self.engine.round())
        self.client.understand.assert_not_awaited()
        self.client.publish.assert_not_awaited()
        self.assertFalse(self.engine.paused)
        self.assertEqual(await self.engine.next_check_delay(), 600)

    async def test_pause_during_selection_cancels_before_content_read(self):
        self.save()
        async def pause(*args):
            await self.engine.pause("测试暂停", "manual")
            return json.dumps({"selected_bvid": ITEM.bvid, "reason": "想聊"})
        self.generate.side_effect = pause
        self.assertIn("已暂停", await self.engine.round())
        self.client.understand.assert_not_awaited()
        self.client.publish.assert_not_awaited()

    async def test_selected_candidate_can_be_rejected_after_fresh_read(self):
        self.save()
        self.want_comment = False
        self.assertIn("看过不评论", await self.engine.round())
        self.client.understand.assert_awaited_once()
        self.review.assert_not_awaited()
        self.client.publish.assert_not_awaited()
        self.assertEqual(self.store.candidates("1", now=self.now), [])

    async def test_daily_analysis_limit_prevents_candidate_read_and_hot_loop(self):
        self.save()
        self.cfg.browse.videos_per_day = 1
        self.assertIn("分析额度已用完", await self.engine.round())
        self.assertEqual(await self.engine.next_check_delay(), 600)
        self.generate.assert_not_awaited()
        self.client.publish.assert_not_awaited()

    async def test_better_old_candidate_survives_new_lower_interest(self):
        self.save(ts=self.now - 3600, interest=.95, limit=1)
        fresh = replace(ITEM, bvid="BV0000000001")
        self.save(fresh, interest=.60, limit=1)
        self.assertEqual([r["bvid"] for r in self.store.candidates("1", now=self.now)], [ITEM.bvid])
        row = self.store.page_records("1", "visits", fresh.bvid, 1)["rows"][0]
        self.assertEqual(row["outcome"], "evicted")
        self.assertEqual(record_row(row, "visits", "Asia/Shanghai")["outcome"], "候选池已满，择优淘汰")

    async def test_true_expiry_and_legacy_capacity_migration_do_not_change_counts(self):
        self.save(ts=self.now - 7200, hours=1)
        self.assertEqual(self.store.candidates("1", now=self.now), [])
        row = self.store.page_records("1", "visits", "", 1)["rows"][0]
        self.assertEqual(row["outcome"], "expired")
        self.assertEqual(record_row(row, "visits", "Asia/Shanghai")["outcome"], "超过有效期")
        old = replace(ITEM, bvid="BV0000000001")
        self.save(old)
        with self.store.connect() as db:
            db.execute("UPDATE visits SET outcome='expired' WHERE bvid=?", (old.bvid,))
        before = self.store.stats("1", 0)
        self.store.initialize(self.now)
        self.store.initialize(self.now)
        self.assertEqual(self.store.stats("1", 0), before)
        row = self.store.page_records("1", "visits", old.bvid, 1)["rows"][0]
        self.assertEqual(row["outcome"], "evicted")

    async def test_selection_timeout_is_throttled_without_sending(self):
        self.save()
        self.generate.side_effect = asyncio.TimeoutError()
        self.assertIn("候选选择未完成", await self.engine.round())
        self.assertEqual(await self.engine.next_check_delay(), 600)
        self.client.publish.assert_not_awaited()

    async def test_interest_only_parser_rejects_drafts_and_ungrounded_quotes(self):
        for raw in [answer(), answer(comment="", evidence_quote="没有提供的内容")]:
            with self.subTest(raw=raw), self.assertRaises(DecisionValidationError):
                parse_decision(raw, EVIDENCE, 160, candidate_only=True)
        self.assertEqual(parse_decision(answer(comment=""), EVIDENCE, 160, candidate_only=True).comment, "")
