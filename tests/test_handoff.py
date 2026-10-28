"""交接六项要求的行为回归；只使用模拟读取、模型和发送接口。"""

from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo
import asyncio
import json
import unittest
import httpx
from local_bilibili_autopilot.client import LoginRequired, PlatformError, select_subtitles
from local_bilibili_autopilot.config import AutopilotConfig
from local_bilibili_autopilot.engine import Autopilot
from local_bilibili_autopilot.models import (
    DecisionValidationError,
    Evidence,
    IncomingReply,
    next_local_day,
    parse_reply_decision,
    parse_share_text,
    parse_support,
)
from local_bilibili_autopilot.plugin import BilibiliAutopilotPlugin
from local_bilibili_autopilot.sharing import VideoSharer
from local_bilibili_autopilot.state import StateStore
from local_bilibili_autopilot.models import VideoItem

ITEM = VideoItem("BV1234567890", 100, "深海光", "2", "UP", 180)
EVIDENCE = Evidence(
    "字幕", "深海生物利用发光器官吸引猎物，不同颜色与闪烁频率也能传递信号，这是海洋生态的一种适应方式。"
)
NOW = datetime(2026, 10, 22, 12, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()


def answer(**kw):
    data = {
        "should_comment": True,
        "interest": 0.9,
        "comment": "这光还能传信，挺有意思",
        "reason": "对发光方式感兴趣",
        "evidence_quote": "深海生物利用发光器官吸引猎物",
    }
    return json.dumps({**data, **kw}, ensure_ascii=False)


def support(ok=True, kind="factual"):
    return json.dumps(
        {"supported": ok, "kind": kind, "reason": "依据支持" if ok else "正文断言未出现在已读内容中"},
        ensure_ascii=False,
    )


class HandoffEngineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.now = NOW
        self.cfg = AutopilotConfig()
        self.cfg.browse.autonomous_enabled = False
        self.cfg.browse.active_start_hour = self.cfg.browse.active_end_hour = 0
        self.cfg.comment.publish_enabled = True
        self.client = AsyncMock()
        self.client.uid = "1"
        self.client.name = "模拟"
        self.client.check_account.return_value = "1"
        self.client.recommended.return_value = [ITEM]
        self.client.understand.return_value = (ITEM, EVIDENCE)
        self.client.publish.return_value = "500"
        self.generate = AsyncMock(return_value=answer())
        self.review = AsyncMock(return_value=support())
        self.store = StateStore(Path(self.temp.name) / "state.sqlite3")
        self.engine = Autopilot(
            self.client,
            self.store,
            lambda: self.cfg,
            self.generate,
            AsyncMock(return_value="好奇"),
            lambda: self.now,
            review=self.review,
        )
        await self.engine.initialize()
        await self.engine.start()

    def quota(self):
        self.generate.return_value = answer(comment="")
        self.cfg.comment.per_day = 1
        self.store.reserve("1", "BV0987654321", "今天早些时候的其他评论", self.now - 7200, 0, 10, 300, 0.88)
        self.store.finish("1", "BV0987654321", "sent", "400")

    async def test_quota_wait_same_day_restart_and_next_day_fresh_generation(self):
        self.quota()
        self.assertIn("daily_limit", await self.engine.round())
        row = self.store.page_records("1", "visits", "", 1)["rows"][0]
        self.assertEqual(row["outcome"], "quota_wait")
        self.assertEqual(row["retry_at"], next_local_day(self.now, "Asia/Shanghai"))
        await self.engine.round()
        self.assertEqual(self.generate.await_count, 1)
        await self.engine.initialize()
        self.client.recommended.return_value = []
        self.now = next_local_day(self.now, "Asia/Shanghai") + 36000
        self.generate.side_effect = [json.dumps({"selected_bvid": ITEM.bvid, "reason": "还想聊"}),
                                     answer(comment="这发光方式确实挺有趣")]
        self.assertIn("已提交评论", await self.engine.round())
        self.assertEqual(self.generate.await_count, 3)
        self.assertEqual(self.client.publish.await_args.args[1], "这发光方式确实挺有趣")

    async def test_quota_and_cooldown_use_later_deadline(self):
        self.now = next_local_day(self.now, "Asia/Shanghai") - 60
        self.cfg.comment.min_interval_seconds = 86400
        self.quota()
        await self.engine.round()
        row = self.store.page_records("1", "visits", "", 1)["rows"][0]
        self.assertGreater(row["retry_at"], next_local_day(self.now, "Asia/Shanghai"))

    async def test_expired_and_newly_blocked_waits_exit_queue(self):
        self.quota()
        await self.engine.round()
        self.now += 49 * 3600
        await self.engine.round()
        self.assertEqual(self.store.page_records("1", "visits", "", 1)["rows"][0]["outcome"], "expired")
        self.client.publish.assert_not_awaited()

    async def test_network_connect_timeout_and_5xx_persist_wait_and_recover(self):
        for failure in [
            httpx.ConnectError("PRIVATE"),
            asyncio.TimeoutError(),
            httpx.HTTPStatusError(
                "PRIVATE", request=httpx.Request("GET", "https://example.test"), response=httpx.Response(503)
            ),
        ]:
            with self.subTest(failure=type(failure).__name__):
                self.client.recommended.side_effect = failure
                result = await self.engine.round()
                self.assertIn("recommendations", result)
                self.assertNotIn("模型判断超时", result)
                self.assertFalse(self.engine.paused)
                calls = self.client.recommended.await_count
                await self.engine.round()
                self.assertEqual(self.client.recommended.await_count, calls)
                await self.engine.initialize()
                self.now += 60
                self.client.recommended.side_effect = None
                self.client.recommended.return_value = []
                await self.engine.round()
                self.assertEqual(self.store.recoveries("1"), [])

    async def test_continuous_failure_backoff_does_not_busy_loop(self):
        self.client.recommended.side_effect = httpx.ConnectError("PRIVATE")
        for delay in [60, 120, 240, 480, 3600]:
            await self.engine.round()
            w = self.store.read_wait("1", "recommendations")
            self.assertEqual(w["retry_at"] - self.now, delay)
            self.now = w["retry_at"]
        self.assertFalse(self.engine.paused)

    async def test_rejections_and_login_remain_paused(self):
        self.client.recommended.side_effect = PlatformError("读取推荐", 429)
        await self.engine.round()
        self.assertTrue(self.engine.paused)
        self.assertEqual(self.store.recoveries("1"), [])

    async def test_unrelated_assertion_valid_quote_never_reserves(self):
        self.generate.return_value = answer(comment="视频说明鲸鱼能在太空生活")
        self.review.return_value = support(False)
        await self.engine.round()
        self.client.publish.assert_not_awaited()
        self.assertEqual(self.store.stats("1", 0)["attempts"], 0)
        self.assertEqual(self.generate.await_count, 2)
        self.assertIn("unsupported_body", self.store.history("1"))

    async def test_title_only_detail_and_review_failure_never_reserves(self):
        self.generate.return_value = answer(comment="标题说鲸鱼能活一千岁")
        self.review.side_effect = asyncio.TimeoutError()
        result = await self.engine.round()
        self.assertIn("复核超时", result)
        self.assertEqual(self.store.stats("1", 0)["attempts"], 0)

    async def test_link_or_mention_blocks_before_review(self):
        for text in ["看看https://example.com", "@某人", "＠某人", "去example.com看"]:
            self.generate.return_value = answer(comment=text)
            await self.engine.round()
            self.now += 600
        self.client.publish.assert_not_awaited()
        self.review.assert_not_awaited()

    async def test_short_subjective_response_passes(self):
        self.generate.return_value = answer(comment="这个挺有意思")
        self.review.return_value = support(True, "subjective")
        self.assertIn("已提交评论", await self.engine.round())

    async def test_pause_during_review_prevents_reservation(self):
        async def pause(*args):
            await self.engine.pause("人工暂停", kind="manual")
            return support()

        self.review.side_effect = pause
        await self.engine.round()
        self.client.publish.assert_not_awaited()
        self.assertEqual(self.store.stats("1", 0)["attempts"], 0)

    async def test_model_selects_tail_video_first(self):
        tail = replace(ITEM, bvid="BV0987654321", up_uid="3")
        self.cfg.browse.autonomous_enabled = True
        self.client.recommended.return_value = [ITEM, tail]
        self.generate.side_effect = [
            json.dumps(
                {
                    "should_browse": True,
                    "reason": "先看后面这个",
                    "next_check_minutes": 10,
                    "selected_bvids": [tail.bvid],
                }
            ),
            answer(),
        ]
        self.client.understand.return_value = (tail, EVIDENCE)
        await self.engine.round()
        self.assertEqual(self.client.understand.await_args_list[0].args[0], tail)
        self.assertEqual(json.loads(self.store.latest_browse_decision("1")["selected_bvids"]), [tail.bvid])

    async def test_unknown_or_duplicate_selection_reads_nothing(self):
        self.cfg.browse.autonomous_enabled = True
        for ids in [["BV0987654321"], [ITEM.bvid, ITEM.bvid], []]:
            self.generate.return_value = json.dumps(
                {"should_browse": True, "reason": "看看", "next_check_minutes": 10, "selected_bvids": ids}
            )
            await self.engine.round()
            self.now += 600
        self.client.understand.assert_not_awaited()

    async def test_wait_queue_does_not_resend_any_attempt_status(self):
        for i, status in enumerate(["pending", "unknown", "sent", "rejected", "cancelled", "acknowledged"]):
            item = replace(ITEM, bvid="BV" + str(i).zfill(10))
            self.store.wait_video("1", item, self.now, json.dumps({"video": asdict(item)}), self.now, 48, 24)
            with self.store.connect() as db:
                db.execute(
                    "INSERT INTO attempts(uid,bvid,ts,text,text_hash,status) VALUES (?,?,?,?,?,?)",
                    ("1", item.bvid, self.now, status, status, status),
                )
            self.assertNotIn(item, self.store.deferred_videos("1", now=self.now))

    async def test_old_quota_without_metadata_reappears_only_after_midnight(self):
        self.store.visit("1", ITEM.bvid, "publish", self.now, "blocked", json.dumps({"reason": "daily_limit"}))
        await self.engine.initialize()
        self.assertTrue(self.store.seen("1", ITEM.bvid, "publish", now=self.now))
        self.now = next_local_day(self.now, "Asia/Shanghai")
        self.assertFalse(self.store.seen("1", ITEM.bvid, "publish", now=self.now))

    async def test_queue_limit_and_cache_expiry_are_account_scoped(self):
        for i in range(3):
            item = replace(ITEM, bvid="BV" + str(i).zfill(10))
            self.store.wait_video("1", item, self.now + i, json.dumps({"video": asdict(item)}), self.now, 48, 2)
        self.assertEqual(len(self.store.deferred_videos("1", now=self.now)), 2)
        self.store.cache_evidence("1", ITEM.bvid, {"quote": "发光器官"}, self.now, 1)
        self.assertEqual(self.store.content_context("2", ITEM.bvid, self.now), {})
        self.assertEqual(self.store.content_context("1", ITEM.bvid, self.now + 86400), {})

    async def test_newly_blocked_wait_is_cancelled_after_config_change(self):
        self.quota()
        await self.engine.round()
        self.cfg.browse.blocked_up_uids = [ITEM.up_uid]
        self.now += 86400
        await self.engine.round()
        self.assertEqual(self.store.page_records("1", "visits", "", 1)["rows"][0]["outcome"], "cancelled")

    async def test_login_loss_and_http_risk_pause_without_retry(self):
        self.client.check_account.side_effect = LoginRequired("登录失效")
        await self.engine.round()
        self.assertTrue(self.engine.paused)
        self.assertEqual(self.store.recoveries("1"), [])

    async def test_wait_and_recheck_leave_an_opportunity_for_new_content(self):
        self.cfg.browse.videos_per_round = 1
        self.store.wait_video("1", ITEM, self.now, json.dumps({"video": asdict(ITEM)}), self.now + 600, 48, 24)
        fresh = replace(ITEM, bvid="BV0987654321", up_uid="3")
        self.client.recommended.return_value = [fresh]
        self.client.understand.side_effect = lambda item, *args: (item, EVIDENCE)
        self.generate.return_value = answer(should_comment=False, comment="", evidence_quote="")
        await self.engine.round()
        self.assertEqual(self.client.understand.await_args_list[0].args[0], fresh)
        fresh2 = replace(fresh, bvid="BV1111111111")
        self.client.recommended.return_value = [fresh2]
        self.now += 600
        self.generate.side_effect = [json.dumps({"selected_bvid": ITEM.bvid, "reason": "这条还想聊"}),
                                     answer(should_comment=False, comment="", evidence_quote="")]
        await self.engine.round()
        self.assertEqual(self.client.understand.await_args_list[1].args[0], ITEM)

    async def test_same_up_limit_keeps_other_up_opportunity(self):
        self.cfg.browse.videos_per_round = 3
        same = [replace(ITEM, bvid="BV" + str(i).zfill(10)) for i in range(3)]
        other = replace(ITEM, bvid="BV9999999999", up_uid="3")
        self.client.recommended.return_value = [*same, other]
        self.client.understand.side_effect = lambda item, *args: (item, EVIDENCE)
        self.generate.return_value = answer(should_comment=False, comment="", evidence_quote="")
        await self.engine.round()
        self.assertEqual(self.client.understand.await_count, 3)
        self.assertEqual(self.client.understand.await_args_list[-1].args[0], other)

    async def test_reply_feed_network_wait_recovers_without_model_timeout_label(self):
        self.cfg.replies.enabled = True
        self.store.reserve("1", ITEM.bvid, "旧评论", self.now - 7200, 0, 10, 300, 0.88)
        self.store.finish("1", ITEM.bvid, "sent", "500")
        self.client.incoming_replies.side_effect = httpx.ConnectError("PRIVATE")
        self.assertIn("replies", await self.engine.reply_round())
        self.assertFalse(self.engine.paused)
        await self.engine.reply_round()
        self.client.incoming_replies.assert_awaited_once()
        self.now += 60
        self.client.incoming_replies.side_effect = None
        self.client.incoming_replies.return_value = []
        await self.engine.reply_round()
        self.assertEqual(self.store.read_wait("1", "replies"), None)

    async def test_factual_followup_without_content_is_rejected(self):
        self.cfg.replies.enabled = True
        self.store.reserve("1", ITEM.bvid, "旧评论", self.now - 7200, 0, 10, 300, 0.88)
        self.store.finish("1", ITEM.bvid, "sent", "500")
        msg = IncomingReply(
            ITEM.bvid, 100, 500, 900, 500, "2", "网友", "鲸鱼真的住在太空吗", "旧评论", "旧评论", self.now
        )
        self.client.incoming_replies.return_value = [msg]
        self.generate.return_value = json.dumps(
            {"should_reply": True, "reply": "视频证明鲸鱼住在太空", "reason": "回应提问"}
        )
        self.review.return_value = support(False)
        await self.engine.reply_round()
        self.client.publish_reply.assert_not_awaited()
        self.assertEqual(self.store.stats("1", 0)["reply_attempts"], 0)

    async def test_configuration_change_during_review_cancels_send(self):
        async def change(*args):
            self.cfg.comment.publish_enabled = False
            return support()

        self.review.side_effect = change
        await self.engine.round()
        self.client.publish.assert_not_awaited()

    async def test_inactive_during_read_never_calls_model(self):
        async def read(*args):
            self.now += 12 * 3600
            return ITEM, EVIDENCE

        self.cfg.browse.active_start_hour = 9
        self.cfg.browse.active_end_hour = 23
        self.client.understand.side_effect = read
        await self.engine.round()
        self.generate.assert_not_awaited()


class HandoffContentTests(unittest.TestCase):
    def test_share_actual_actions_blocked_but_text_reactions_allowed(self):
        for text in ["跟着拍手练三拍换四拍，直接手忙脚乱", "跟着拍手试了两下", "我也去过这里", "照着拍了两下"]:
            with self.subTest(text=text), self.assertRaises(DecisionValidationError):
                parse_share_text(json.dumps({"text": text}, ensure_ascii=False))
        for text in ["三百多级台阶，光听介绍腿就酸了", "三拍和四拍的区别还挺有意思", "换完弦还得重新调音"]:
            self.assertEqual(parse_share_text(json.dumps({"text": text}, ensure_ascii=False)), text)

    def test_relevant_tail_and_timestamps_survive_budget(self):
        rows = [{"from": i * 10, "to": i * 10 + 9, "text": "开头普通说明" * 30} for i in range(100)]
        rows[93]["text"] = "关键结论：深海的闪烁频率也能传递信号"
        text, segments = select_subtitles(rows, 500, "闪烁频率传递信号")
        self.assertIn("关键结论", text)
        self.assertTrue(any(r["from"] == 930 for r in segments))
        self.assertLessEqual(len(text), 500)
        self.assertTrue(any(r["from"] == 0 for r in segments))
        self.assertTrue(any(r["from"] == 990 for r in segments))

    def test_dst_next_midnight_uses_calendar_day(self):
        now = datetime(2026, 3, 8, 0, tzinfo=ZoneInfo("America/New_York")).timestamp()
        self.assertEqual(next_local_day(now, "America/New_York") - now, 23 * 3600)

    def test_reply_links_mentions_and_support_schema_fail_closed(self):
        for text in ["@别人", "https://example.com", "example.org"]:
            with self.assertRaises(DecisionValidationError):
                parse_reply_decision(json.dumps({"should_reply": True, "reply": text, "reason": "接话"}), 160)
        with self.assertRaises(DecisionValidationError):
            parse_support('{"supported":true}')


class SharePolishTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.plugin = BilibiliAutopilotPlugin()
        self.plugin.set_plugin_config({})
        self.plugin._persona = AsyncMock(return_value="随和")
        self.draft = "这树还成景点了"
        self.polished = "一棵树也是景点啊"
        self.context = {
            "content_evidence": "这个景区有一棵单独的树作为景点",
            "my_comment": "原评论",
            "recent_shares": [],
        }

    async def test_polish_success_and_at_most_one_call(self):
        self.plugin._generate = AsyncMock(
            side_effect=[json.dumps({"text": self.draft}), support(), json.dumps({"text": self.polished}), support()]
        )
        trace = {}
        result = await self.plugin._share_steps(ITEM, self.context, self.plugin.config.model_copy(deep=True), trace=trace)
        self.assertEqual(result, self.polished)
        self.assertEqual(self.plugin._generate.await_count, 4)
        self.assertEqual(trace["draft"], self.draft)
        self.assertEqual(trace["polish_candidate"], self.polished)
        self.assertEqual(trace["result_mode"], "polished")

    async def test_unfounded_polish_is_reviewed_and_fallback_recorded(self):
        self.plugin._generate = AsyncMock(side_effect=[
            json.dumps({"text": self.draft}), support(), json.dumps({"text": "树边还有三间小店"}), support(False),
        ])
        trace = {}
        result = await self.plugin._share_steps(ITEM, self.context, self.plugin.config.model_copy(deep=True), trace=trace)
        self.assertEqual(result, self.draft)
        self.assertEqual(self.plugin._generate.await_count, 4)
        self.assertEqual(trace["result_mode"], "fallback")
        self.assertEqual(trace["fallback_reason"], "unsupported_share")
        self.assertFalse(trace["polish_review"]["supported"])

    async def test_each_candidate_call_counts_and_keeps_its_own_usage(self):
        from types import SimpleNamespace
        with TemporaryDirectory() as folder:
            store = StateStore(Path(folder) / "state.sqlite3")
            store.initialize()
            self.plugin.engine = SimpleNamespace(store=store)
            self.plugin.client = SimpleNamespace(uid="1")
            generate = AsyncMock(side_effect=[
                {"success": False, "error": "invalid output", "usage": {"total_tokens": 17}},
                {"success": True, "response": "valid", "usage": {"total_tokens": 23}},
            ])
            self.plugin._ctx = SimpleNamespace(llm=SimpleNamespace(generate=generate), logger=SimpleNamespace(warning=lambda *a: None))
            cfg = AutopilotConfig.model_validate({"model": {"candidates": ["first", "second"]}})
            result = await BilibiliAutopilotPlugin._generate(self.plugin, [{"role": "system", "content": "[autopilot:evidence_review]"}], cfg)
            self.assertEqual(result, "valid")
            self.assertEqual(store.model_usage("1", 0)["calls"], 2)
            self.assertEqual(store.model_usage("1", 0)["reported_usage"], [{"total_tokens": 17}, {"total_tokens": 23}])
            with store.connect() as db:
                self.assertEqual([r[0] for r in db.execute("SELECT outcome FROM model_calls ORDER BY id")], ["failed", "success"])

    async def test_cancelled_model_call_counts_without_fallback(self):
        from types import SimpleNamespace
        with TemporaryDirectory() as folder:
            store = StateStore(Path(folder) / "state.sqlite3")
            store.initialize()
            self.plugin.engine = SimpleNamespace(store=store)
            self.plugin.client = SimpleNamespace(uid="1")
            generate = AsyncMock(side_effect=asyncio.CancelledError)
            self.plugin._ctx = SimpleNamespace(llm=SimpleNamespace(generate=generate))
            with self.assertRaises(asyncio.CancelledError):
                await BilibiliAutopilotPlugin._generate(self.plugin, "prompt", AutopilotConfig())
            self.assertEqual(generate.await_count, 1)
            with store.connect() as db:
                self.assertEqual(db.execute("SELECT outcome FROM model_calls").fetchone()[0], "cancelled")

    async def test_invalid_empty_link_mention_and_unfounded_polish_falls_back(self):
        for raw in [
            "not json",
            json.dumps({"text": ""}),
            json.dumps({"text": "@所有人"}),
            json.dumps({"text": "https://example.com"}),
            json.dumps({"text": "我去年去过这里"}),
        ]:
            self.plugin._generate = AsyncMock(
                side_effect=[json.dumps({"text": self.draft}), support(), raw, support(False)]
            )
            self.assertEqual(
                await self.plugin._share_steps(ITEM, self.context, self.plugin.config.model_copy(deep=True)), self.draft
            )

    async def test_timeout_falls_back_to_verified_draft(self):
        self.plugin._generate = AsyncMock(
            side_effect=[json.dumps({"text": self.draft}), support(), asyncio.TimeoutError()]
        )
        self.assertEqual(
            await self.plugin._share_steps(ITEM, self.context, self.plugin.config.model_copy(deep=True)), self.draft
        )

    async def test_draft_review_failure_never_returns_sendable_text(self):
        self.plugin._generate = AsyncMock(side_effect=[json.dumps({"text": "我以前去过这里"}), support(False)])
        with self.assertRaises(DecisionValidationError):
            await self.plugin._share_steps(ITEM, self.context, self.plugin.config.model_copy(deep=True))

    async def test_disabled_polish_uses_verified_draft(self):
        self.plugin.config.share.polish_enabled = False
        self.plugin._generate = AsyncMock(side_effect=[json.dumps({"text": self.draft}), support()])
        self.assertEqual(
            await self.plugin._share_steps(ITEM, self.context, self.plugin.config.model_copy(deep=True)), self.draft
        )
        self.assertEqual(self.plugin._generate.await_count, 2)

    async def test_rejected_draft_is_rewritten_once_with_specific_feedback_and_reviewed(self):
        self.plugin.config.share.polish_enabled = False
        rejection = {"supported": False, "kind": "factual", "reason": "摘录未提供祠堂地点"}
        self.plugin._generate = AsyncMock(side_effect=[
            json.dumps({"text": "跑去祠堂问公司能不能活三年"}), json.dumps(rejection),
            json.dumps({"text": "这问题压力有点大"}), support(),
        ])
        trace = {}
        result = await self.plugin._share_steps(ITEM, self.context, self.plugin.config.model_copy(deep=True), trace=trace)
        self.assertEqual(result, "这问题压力有点大")
        self.assertEqual(self.plugin._generate.await_count, 4)
        retry_context = json.loads(self.plugin._generate.await_args_list[2].args[0][1]["content"])
        self.assertEqual(retry_context["validation_feedback"]["reason"], rejection["reason"])
        self.assertEqual(retry_context["content_evidence"], self.context["content_evidence"])
        review_context = json.loads(self.plugin._generate.await_args_list[3].args[0][1]["content"])
        self.assertEqual(review_context["body"], result)
        self.assertEqual(len(trace["draft_attempts"]), 2)
        self.assertFalse(trace["draft_attempts"][0]["review"]["supported"])
        self.assertNotIn("validation_feedback", self.context)

    async def test_second_draft_rejection_stops_with_public_reason(self):
        rejection = json.dumps({"supported": False, "kind": "factual", "reason": "摘录缺少地点依据"})
        self.plugin._generate = AsyncMock(side_effect=[
            json.dumps({"text": "祠堂压力大"}), rejection,
            json.dumps({"text": "祖宗压力大"}), rejection,
        ])
        with self.assertRaisesRegex(DecisionValidationError, "摘录缺少地点依据"):
            await self.plugin._share_steps(ITEM, self.context, self.plugin.config.model_copy(deep=True))
        self.assertEqual(self.plugin._generate.await_count, 4)

    async def test_invalid_draft_format_can_be_repaired_but_timeout_is_not_retried(self):
        self.plugin.config.share.polish_enabled = False
        self.plugin._generate = AsyncMock(side_effect=["not json", json.dumps({"text": self.draft}), support()])
        self.assertEqual(
            await self.plugin._share_steps(ITEM, self.context, self.plugin.config.model_copy(deep=True)), self.draft
        )
        self.assertEqual(self.plugin._generate.await_count, 3)
        self.plugin._generate = AsyncMock(side_effect=[json.dumps({"text": self.draft}), asyncio.TimeoutError()])
        with self.assertRaises(asyncio.TimeoutError):
            await self.plugin._share_steps(ITEM, self.context, self.plugin.config.model_copy(deep=True))
        self.assertEqual(self.plugin._generate.await_count, 2)

    async def test_pause_and_config_change_during_polish_cancel_actual_share(self):
        for pause in [True, False]:
            with TemporaryDirectory() as folder:
                plugin = BilibiliAutopilotPlugin()
                plugin.set_plugin_config({})
                plugin.config.comment.publish_enabled = True
                plugin.config.browse.active_start_hour = plugin.config.browse.active_end_hour = 0
                plugin.config.share.qq_group_ids = ["123456"]
                store = StateStore(Path(folder) / "state.sqlite3")
                store.initialize()
                store.set_pause(False, "")
                engine = Autopilot(AsyncMock(), store, lambda: plugin.config, AsyncMock(), AsyncMock())
                engine.paused = False
                plugin.engine = engine
                plugin._persona = AsyncMock(return_value="随和")
                store.reserve("1", ITEM.bvid, "原评论", NOW, 0, 10, 300, 0.88)
                store.finish("1", ITEM.bvid, "sent", "500")
                store.visit("1", ITEM.bvid, "publish", NOW, "sent", json.dumps({"evidence_quote": "一棵树成为景点"}))

                async def generate(prompt, cfg):
                    system = prompt[0]["content"]
                    if "share_polish" in system:
                        if pause:
                            await engine.pause("人工暂停", kind="manual")
                        else:
                            plugin.config.browse.interests = "配置变化"
                        return json.dumps({"text": "一棵树就成景点了"})
                    if "evidence_review" in system:
                        return support()
                    return json.dumps({"text": "这树还成景点了"})

                plugin._generate = AsyncMock(side_effect=generate)
                chat = AsyncMock()
                chat.get_group_streams.return_value = [
                    {
                        "platform": "qq",
                        "group_id": "123456",
                        "chat_type": "group",
                        "session_id": "test-session",
                        "account_id": "test-account",
                    }
                ]
                send = AsyncMock()
                sharer = VideoSharer(
                    store, lambda: plugin.config, chat, send, engine.may_publish, plugin._compose_share
                )
                result = await sharer.share("1", ITEM)
                send.text.assert_not_awaited()
                self.assertIn("已取消", result)
