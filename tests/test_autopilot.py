from datetime import datetime
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import asyncio
import json
import unittest

from local_bilibili_autopilot.client import PlatformError, UnknownPublish
from local_bilibili_autopilot.config import AutopilotConfig
from local_bilibili_autopilot.engine import Autopilot
from local_bilibili_autopilot.models import (Evidence, IncomingReply, ModelResponseError, ModelTimeoutError, VideoItem, active_time, local_day_start,
                                           parse_browse_decision, parse_decision, recommendations)
from local_bilibili_autopilot.state import StateStore


ITEM = VideoItem("BV1234567890", 100, "深海生物的光", "2002", "科普UP", 180)
EVIDENCE = Evidence("字幕", "深海生物利用发光器官吸引猎物，不同颜色与闪烁频率也能在黑暗环境中传递信号，这是海洋生态的一种适应方式。")
NOW = datetime(2026, 10, 19, 12, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()


def decision(**changes):
    fields = {"should_comment": True, "interest": 0.9, "comment": "原来深海的光既能捕猎又能传信，这种适应方式好有意思。",
              "reason": "对深海生态有兴趣，发光用途具体有趣", "evidence_quote": "深海生物利用发光器官吸引猎物"}
    fields.update(changes)
    return json.dumps(fields, ensure_ascii=False)


class EngineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = AutopilotConfig()
        self.cfg.replies.enabled = False
        self.cfg.browse.autonomous_enabled = False
        self.cfg.browse.max_check_minutes = 180
        self.cfg.browse.interval_seconds = 1800
        self.cfg.comment.publish_enabled = True
        self.cfg.browse.active_start_hour = self.cfg.browse.active_end_hour = 0
        self.now = NOW
        self.client = AsyncMock()
        self.client.uid, self.client.name = "1001", "模拟账号"
        self.client.check_account.return_value = "1001"
        self.client.recommended.return_value = [ITEM]
        self.client.understand.return_value = ITEM, EVIDENCE
        self.client.publish.return_value = "9988"
        self.generate = AsyncMock(return_value=decision())
        self.persona = AsyncMock(return_value="好奇、喜欢海洋生物")
        self.store = StateStore(Path(self.temp.name) / "state.sqlite3")
        self.engine = Autopilot(self.client, self.store, lambda: self.cfg, self.generate, self.persona, lambda: self.now)
        await self.engine.initialize()

    async def test_initial_state_does_not_browse_or_publish(self):
        self.assertTrue(self.engine.paused)
        await self.engine.round()
        self.client.recommended.assert_not_awaited()
        self.client.publish.assert_not_awaited()

    async def test_explicit_preview_never_publishes_with_publish_switch_on(self):
        result = await self.engine.round(preview=True)
        self.assertIn("预览评论", result)
        self.client.publish.assert_not_awaited()
        self.assertEqual(self.store.stats("1001", 0)["attempts"], 0)
        self.assertTrue(self.engine.paused)

    async def test_scheduled_preview_when_switch_off(self):
        self.cfg.comment.publish_enabled = False
        await self.engine.start()
        self.assertIn("预览评论", await self.engine.round())
        self.client.publish.assert_not_awaited()

    async def test_share_callback_runs_after_success_is_persisted(self):
        async def share(uid, item):
            self.assertEqual(self.store.own_comments(uid)["9988"]["bvid"], item.bvid)
            return "已分享至群聊"
        self.engine.share_video = AsyncMock(side_effect=share)
        await self.engine.start()
        self.assertIn("已分享至群聊", await self.engine.round())
        self.engine.share_video.assert_awaited_once_with(self.client.uid, ITEM)

    async def test_preview_and_skipped_comments_do_not_share(self):
        self.engine.share_video = AsyncMock(return_value="已分享至群聊")
        await self.engine.round(preview=True)
        await self.engine.start()
        self.generate.return_value = decision(should_comment=False, interest=.8, comment="", evidence_quote="")
        await self.engine.round()
        self.engine.share_video.assert_not_awaited()

    async def test_share_failure_does_not_pause_or_undo_successful_comment(self):
        self.engine.share_video = AsyncMock(side_effect=RuntimeError("test"))
        await self.engine.start()
        self.assertIn("群分享处理失败", await self.engine.round())
        self.assertFalse(self.engine.paused)
        self.assertEqual(self.store.own_comments(self.client.uid)["9988"]["bvid"], ITEM.bvid)

    async def test_preview_does_not_consume_live_video_opportunity(self):
        await self.engine.round(preview=True)
        await self.engine.start()
        result = await self.engine.round()
        self.assertIn("已提交评论", result)
        self.client.publish.assert_awaited_once()
        self.assertEqual(self.store.stats("1001", 0)["attempts"], 1)

    async def test_no_repeated_comment_after_restart(self):
        await self.engine.start()
        await self.engine.round()
        self.now += 7200
        restored = Autopilot(self.client, self.store, lambda: self.cfg, self.generate, self.persona, lambda: self.now)
        await restored.initialize()
        await restored.round()
        self.client.publish.assert_awaited_once()

    async def test_missing_content_and_no_interest_do_not_publish(self):
        await self.engine.start()
        self.client.understand.return_value = ITEM, Evidence("unavailable", "")
        await self.engine.round()
        self.generate.assert_not_awaited()
        self.client.publish.assert_not_awaited()

    async def test_video_read_minus_504_skips_only_that_video_and_continues(self):
        bad = replace(ITEM, bvid="BV0987654321")
        self.client.recommended.return_value = [bad, ITEM]
        self.client.understand.side_effect = [PlatformError("读取视频内容", -504), (ITEM, EVIDENCE)]
        await self.engine.start()
        result = await self.engine.round()
        self.assertIn("code=-504", result)
        self.assertIn("已提交评论", result)
        self.assertFalse(self.engine.paused)
        self.client.publish.assert_awaited_once()
        records = self.store.page_records(self.client.uid, "visits", bad.bvid, 1)["rows"]
        self.assertEqual(records[0]["outcome"], "read_error")
        self.assertIn("code=-504", records[0]["detail"])

    async def test_read_timeout_skips_without_publishing_or_pausing(self):
        self.client.understand.side_effect = asyncio.TimeoutError()
        await self.engine.start()
        self.assertIn("已跳过", await self.engine.round())
        self.assertFalse(self.engine.paused)
        self.generate.assert_not_awaited()
        self.client.publish.assert_not_awaited()

    async def test_other_platform_read_errors_still_pause(self):
        self.client.understand.side_effect = PlatformError("读取视频内容", -412)
        await self.engine.start()
        self.assertIn("已暂停", await self.engine.round())
        self.assertTrue(self.engine.paused)
        self.client.publish.assert_not_awaited()

    async def test_model_can_choose_not_to_comment(self):
        await self.engine.start()
        self.generate.return_value = decision(should_comment=False, comment="", evidence_quote="", interest=0.2)
        result = await self.engine.round()
        self.assertIn("看过不评论", result)
        self.client.publish.assert_not_awaited()

    async def test_invalid_model_output_fails_closed(self):
        await self.engine.start()
        self.generate.return_value = decision(should_comment="true")
        await self.engine.round()
        self.client.publish.assert_not_awaited()

    async def test_pause_during_generation_prevents_publish(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def delayed(*args):
            entered.set()
            await release.wait()
            return decision()
        self.generate.side_effect = delayed
        await self.engine.start()
        task = asyncio.create_task(self.engine.round())
        await entered.wait()
        await self.engine.pause("用户暂停")
        release.set()
        await task
        self.client.publish.assert_not_awaited()

    async def test_config_change_during_generation_invalidates_decision(self):
        async def change(*args):
            self.cfg.browse.interests = "现在只想看音乐"
            return decision()
        self.generate.side_effect = change
        await self.engine.start()
        await self.engine.round()
        self.client.publish.assert_not_awaited()

    async def test_unknown_result_pauses_and_does_not_retry(self):
        self.engine.share_video = AsyncMock()
        self.client.publish.side_effect = UnknownPublish("timeout")
        await self.engine.start()
        result = await self.engine.round()
        self.assertIn("待核对", result)
        self.assertTrue(self.engine.paused)
        self.assertEqual(self.store.stats("1001", 0)["unresolved"], [ITEM.bvid])
        self.assertIn("待核对", await self.engine.start())
        await self.engine.round(preview=True)
        self.client.publish.assert_awaited_once()
        self.engine.share_video.assert_not_awaited()
        self.assertTrue(self.store.acknowledge("1001", ITEM.bvid))
        self.now += 7200
        await self.engine.start()
        await self.engine.round()
        self.client.publish.assert_awaited_once()

    async def test_daily_view_limit_also_bounds_manual_previews(self):
        self.cfg.browse.videos_per_day = 1
        await self.engine.round(preview=True)
        self.assertIn("额度已用完", await self.engine.round(preview=True))
        self.client.recommended.assert_awaited_once()

    async def test_background_and_manual_rounds_are_serial(self):
        async with self.engine.lock:
            self.assertIn("已有", await self.engine.round(preview=True))
        self.client.recommended.assert_not_awaited()

    async def test_blocked_up_never_reaches_model(self):
        self.cfg.browse.blocked_up_uids = [ITEM.up_uid]
        await self.engine.round(preview=True)
        self.generate.assert_not_awaited()

    async def test_content_keyword_blocks_even_if_title_does_not_match(self):
        self.cfg.browse.blocked_keywords = ["发光器官"]
        await self.engine.round(preview=True)
        self.generate.assert_not_awaited()

    async def test_model_timeout_keeps_browsing_enabled_for_next_round(self):
        await self.engine.start()
        self.generate.side_effect = ModelTimeoutError("候选模型均超时")
        self.assertIn("超时", await self.engine.round())
        self.assertFalse(self.engine.paused)
        self.assertEqual(self.store.control()["paused"], "0")
        self.client.publish.assert_not_awaited()
        self.client.recommended.return_value = [replace(ITEM, bvid="BV1234567891")]
        self.generate.side_effect = None
        await self.engine.round()
        self.client.publish.assert_awaited_once()

    async def test_browse_choice_timeout_records_next_check_without_pausing(self):
        await self.engine.start()
        self.cfg.browse.autonomous_enabled = True
        self.generate.side_effect = asyncio.TimeoutError
        self.assertIn("超时", await self.engine.round())
        self.assertFalse(self.engine.paused)
        self.assertEqual(await self.engine.next_check_delay(), self.cfg.browse.interval_seconds)
        self.client.understand.assert_not_awaited()
        self.client.publish.assert_not_awaited()

    async def test_model_failure_pauses_with_no_leaked_error_details(self):
        self.generate.side_effect = RuntimeError("SESSDATA=secret")
        await self.engine.round(preview=True)
        self.assertTrue(self.engine.paused)
        self.assertNotIn("secret", self.engine.last_result)
        self.client.publish.assert_not_awaited()

    async def test_model_response_error_pauses_with_actionable_reason_and_no_attempt(self):
        message = "模型只返回了思考内容，没有评论正文；请检查模型任务的最大输出 token 上限后重新预览"
        self.generate.side_effect = ModelResponseError(message)
        await self.engine.start()
        result = await self.engine.round()
        self.assertTrue(self.engine.paused)
        self.assertEqual(result, "已暂停：" + message)
        self.assertEqual(self.store.control()["pause_reason"], message)
        self.assertIn(message, self.store.history(self.client.uid))
        self.assertEqual(self.store.stats(self.client.uid, 0)["attempts"], 0)
        self.client.publish.assert_not_awaited()

    async def test_truncated_json_is_never_published(self):
        self.generate.return_value = '{"should_comment": true, "interest": 0.9, "comment": "未完成'
        await self.engine.start()
        self.assertIn("模型输出校验失败", await self.engine.round())
        self.assertEqual(self.store.stats(self.client.uid, 0)["attempts"], 0)
        self.client.publish.assert_not_awaited()

    async def test_autonomous_rest_only_reads_recommendation_metadata_and_honors_deadline(self):
        self.cfg.browse.autonomous_enabled = True
        self.generate.return_value = json.dumps({"should_browse": False, "reason": "刚刷过，晚些再看看有没有感兴趣的内容",
                                                 "next_check_minutes": 60})
        await self.engine.start()
        self.assertIn("暂时不刷", await self.engine.round())
        self.assertFalse(self.engine.paused)
        self.client.recommended.assert_awaited_once()
        self.client.understand.assert_not_awaited()
        self.client.publish.assert_not_awaited()
        self.assertEqual(self.store.stats(self.client.uid, 0)["views"], 0)
        self.assertEqual(await self.engine.next_check_delay(), 3600)
        self.assertIn("晚些再看看", await self.engine.status())
        self.assertIn("暂时不刷", self.store.history(self.client.uid))
        # 重启或重复唤醒不会跳过麦麦自己的休息决定。
        await self.engine.initialize()
        self.assertIn("尚未到", await self.engine.round())
        self.generate.assert_awaited_once()
        self.client.recommended.assert_awaited_once()
        self.now += 3601
        await self.engine.round()
        self.assertEqual(self.generate.await_count, 2)

    async def test_autonomous_browsing_and_commenting_are_separate_decisions(self):
        self.cfg.browse.autonomous_enabled = True
        self.generate.side_effect = [json.dumps({"should_browse": True, "reason": "想找点海洋科普看看",
                                                 "next_check_minutes": 45}), decision()]
        await self.engine.start()
        self.assertIn("已提交评论", await self.engine.round())
        self.assertEqual(self.generate.await_count, 2)
        self.client.recommended.assert_awaited_once()
        self.client.publish.assert_awaited_once()
        self.assertEqual(await self.engine.next_check_delay(), 2700)

    async def test_preview_bypasses_autonomous_desire_and_never_publishes(self):
        self.cfg.browse.autonomous_enabled = True
        self.assertIn("预览评论", await self.engine.round(preview=True))
        self.generate.assert_awaited_once()
        self.assertIsNone(self.store.latest_browse_decision(self.client.uid))
        self.client.publish.assert_not_awaited()

    async def test_pause_during_desire_prevents_reading_content_or_publishing(self):
        self.cfg.browse.autonomous_enabled = True
        async def stop(*args):
            await self.engine.pause("管理员手动暂停", kind="manual")
            return json.dumps({"should_browse": True, "reason": "想看科普", "next_check_minutes": 30})
        self.generate.side_effect = stop
        await self.engine.start()
        self.assertIn("不浏览", await self.engine.round())
        self.client.recommended.assert_awaited_once()
        self.client.understand.assert_not_awaited()
        self.client.publish.assert_not_awaited()
        self.assertEqual(self.store.control()["pause_kind"], "manual")

    async def test_invalid_desire_is_not_repaired_into_permission_to_browse(self):
        self.cfg.browse.autonomous_enabled = True
        self.generate.return_value = '{"should_browse":true'
        await self.engine.start()
        self.assertIn("格式无效", await self.engine.round())
        self.client.recommended.assert_awaited_once()
        self.client.understand.assert_not_awaited()
        self.assertEqual(await self.engine.next_check_delay(), 1800)

    async def test_first_login_auto_starts_without_command(self):
        self.cfg.browse.autonomous_enabled = True
        self.assertIn("自主模式", await self.engine.start(automatic=True))
        self.assertFalse(self.engine.paused)
        self.assertEqual(self.store.control()["pause_kind"], "running")

    async def test_auto_start_never_resumes_manual_or_error_pause_after_restart(self):
        for kind in ("manual", "error", "logout"):
            with self.subTest(kind=kind):
                await self.engine.pause("保持暂停", kind=kind)
                await self.engine.initialize()
                self.client.check_account.reset_mock()
                await self.engine.start(automatic=True)
                self.assertTrue(self.engine.paused)
                self.client.check_account.assert_not_awaited()

    async def test_auto_start_requires_enabled_setting_and_saved_account(self):
        self.cfg.browse.auto_start = False
        await self.engine.start(automatic=True)
        self.assertTrue(self.engine.paused)
        self.cfg.browse.auto_start = True
        self.client.uid = ""
        await self.engine.start(automatic=True)
        self.client.check_account.assert_not_awaited()

    async def test_manual_pause_during_account_check_cancels_auto_start(self):
        async def check():
            await self.engine.pause("管理员手动暂停", kind="manual")
            return self.client.uid
        self.client.check_account.side_effect = check
        await self.engine.start(automatic=True)
        self.assertTrue(self.engine.paused)
        self.assertEqual(self.store.control()["pause_kind"], "manual")

    async def test_comment_quota_does_not_stop_browsing(self):
        self.cfg.comment.per_day = 1
        await self.engine.start()
        self.assertEqual(self.store.reserve(self.client.uid, "BV0987654321", "不重复的已有评论", self.now,
                                            0, 1, 3600, .88), "reserved")
        self.store.finish(self.client.uid, "BV0987654321", "sent", "existing")
        self.now += 3601  # 单独验证每日限额，不受发送冷却影响。
        self.assertIn("daily_limit", await self.engine.round())
        self.client.recommended.assert_awaited_once()
        self.generate.assert_awaited_once()
        self.client.publish.assert_not_awaited()

    async def test_cooldown_is_checked_before_browse_models_and_recovers_at_boundary(self):
        await self.engine.start()
        self.store.reserve(self.client.uid, 'BV0987654321', '不重复的已有评论', self.now, 0, 12, 0, .88)
        self.store.finish(self.client.uid, 'BV0987654321', 'sent', '1')
        self.cfg.browse.autonomous_enabled = True
        result = await self.engine.round()
        self.assertIn('12:45:00', result)
        self.assertIn('冷却中', result)
        self.client.recommended.assert_not_awaited()
        self.client.understand.assert_not_awaited()
        self.generate.assert_not_awaited()
        self.assertFalse(self.engine.paused)
        self.assertFalse(self.store.seen(self.client.uid, ITEM.bvid, 'publish'))
        self.cfg.browse.autonomous_enabled = False
        self.now += self.cfg.comment.min_interval_seconds
        self.assertIn('已提交评论', await self.engine.round())
        self.client.publish.assert_awaited_once()

    async def test_manual_preview_can_run_during_comment_cooldown(self):
        await self.engine.start()
        self.store.reserve(self.client.uid, 'BV0987654321', '不重复的已有评论', self.now, 0, 12, 0, .88)
        self.store.finish(self.client.uid, 'BV0987654321', 'sent', '1')
        self.assertIn('预览评论', await self.engine.round(preview=True))
        self.generate.assert_awaited_once()
        self.client.publish.assert_not_awaited()

    async def test_recent_followup_also_delays_new_video_comment(self):
        await self.engine.start()
        self.store.reserve(self.client.uid, 'BV0987654321', '已有根评论', self.now-7200, 0, 12, 0, .88)
        self.store.finish(self.client.uid, 'BV0987654321', 'sent', '500')
        message = IncomingReply('BV0987654321', 100, 500, 900, 500, '2', '对方', '继续讨论',
                                '已有根评论', '已有根评论', self.now)
        self.assertEqual(self.store.reserve_reply(self.client.uid, message, '跟进讨论的回复', self.now,
                                                  0, 8, 900, 3, .88), 'reserved')
        self.store.finish_reply(self.client.uid, 900, 'sent', '501')
        self.assertIn('12:45:00', await self.engine.round())
        self.generate.assert_not_awaited()
        self.client.publish.assert_not_awaited()

    async def test_concurrent_send_defers_video_preserves_details_and_retries_after_restart(self):
        await self.engine.start()
        async def concurrent_send(*args):
            self.store.reserve(self.client.uid, 'BV0987654321', '其他工作进程刚提交的评论', self.now, 0, 12, 0, .88)
            self.store.finish(self.client.uid, 'BV0987654321', 'sent', '1')
            return decision()
        self.generate.side_effect = concurrent_send
        await self.engine.round()
        row = self.store.page_records(self.client.uid, 'visits', ITEM.bvid, 1)['rows'][0]
        detail = json.loads(row['detail'])
        self.assertEqual(row['outcome'], 'deferred')
        self.assertEqual(detail['title'], ITEM.title)
        self.assertEqual(detail['comment'], json.loads(decision())['comment'])
        self.assertEqual(detail['decision_reason'], json.loads(decision())['reason'])
        self.assertIn('12:45:00', detail['reason'])
        self.client.publish.assert_not_awaited()
        self.assertFalse(self.engine.paused)
        self.assertFalse(self.store.seen(self.client.uid, ITEM.bvid, 'publish'))
        self.store.initialize()
        self.now += self.cfg.comment.min_interval_seconds
        self.client.recommended.return_value = []  # 推荐列表已变化，仍恢复保留的视频。
        self.generate.side_effect = None
        result = await self.engine.round()
        self.assertIn('已提交评论', result)
        self.assertEqual(self.generate.await_count, 2)  # 重新读取内容和判断，不能直接发送旧草稿。
        self.client.publish.assert_awaited_once()
        self.assertEqual(self.store.deferred_videos(self.client.uid), [])

    async def test_legacy_cooldown_record_is_reconsidered_when_recommended_again(self):
        await self.engine.start()
        self.store.visit(self.client.uid, ITEM.bvid, 'publish', self.now-7200, 'blocked', 'cooldown')
        self.assertIn('已提交评论', await self.engine.round())
        self.client.publish.assert_awaited_once()

    async def test_reducing_maximum_shortens_saved_wait_and_enforces_new_model_range(self):
        self.cfg.browse.autonomous_enabled = True
        self.cfg.browse.interval_seconds = 600
        self.cfg.browse.max_check_minutes = 30
        await self.engine.start()
        self.store.record_browse_decision(self.client.uid, self.now, False, "先休息", self.now + 7200)
        self.assertEqual(await self.engine.next_check_delay(), 1800)
        self.assertIn("10-19 12:30", await self.engine.status())
        self.now += 1801
        self.generate.return_value = json.dumps({"should_browse": False, "reason": "晚些再看", "next_check_minutes": 10})
        self.assertIn("暂时不刷", await self.engine.round())
        self.assertEqual(await self.engine.next_check_delay(), 600)
        self.assertIn("10～30", self.generate.await_args.args[0][0]["content"])
        self.client.recommended.assert_awaited_once()
        self.client.understand.assert_not_awaited()

    async def test_model_cannot_choose_more_than_configured_maximum(self):
        self.cfg.browse.autonomous_enabled = True
        self.cfg.browse.interval_seconds = 600
        self.cfg.browse.max_check_minutes = 30
        self.generate.return_value = json.dumps({"should_browse": True, "reason": "想看视频", "next_check_minutes": 120})
        await self.engine.start()
        self.assertIn("格式无效", await self.engine.round())
        self.assertEqual(await self.engine.next_check_delay(), 600)
        self.client.recommended.assert_awaited_once()
        self.client.understand.assert_not_awaited()

    async def test_new_recommendations_and_elapsed_rest_replace_stale_negative_context(self):
        self.cfg.browse.autonomous_enabled = True
        await self.engine.start()
        self.store.visit(self.client.uid, "BV0987654321", "preview", self.now - 3700,
                         "skipped", json.dumps({"reason": "之前的视频没兴趣"}))
        self.store.record_browse_decision(self.client.uid, self.now - 3600, False, "休息", self.now - 1)
        self.generate.side_effect = [json.dumps({"should_browse": True, "reason": "新的海洋科普有兴趣",
                                                 "next_check_minutes": 30}), decision()]
        self.assertIn("已提交评论", await self.engine.round())
        context = json.loads(self.generate.await_args_list[0].args[0][1]["content"])
        self.assertEqual(context["current_recommendations"][0]["title"], ITEM.title)
        self.assertEqual(context["minutes_since_last_video_analysis"], 61.7)
        self.assertEqual(context["recent_browsing"][0]["mode"], "preview")
        self.assertEqual(context["recent_browsing"][0]["minutes_ago"], 61.7)
        self.client.recommended.assert_awaited_once()
        self.client.understand.assert_awaited_once()

    async def test_pause_while_fetching_new_candidates_prevents_model_and_content_reads(self):
        self.cfg.browse.autonomous_enabled = True
        await self.engine.start()
        async def stop():
            await self.engine.pause("管理员手动暂停", kind="manual")
            return [ITEM]
        self.client.recommended.side_effect = stop
        self.assertIn("不浏览", await self.engine.round())
        self.generate.assert_not_awaited()
        self.client.understand.assert_not_awaited()
        self.client.publish.assert_not_awaited()

    async def test_active_exploration_and_lower_interest_gate_allow_a_concrete_comment(self):
        self.cfg.browse.autonomous_enabled = True
        self.cfg.browse.explore_more = True
        self.cfg.comment.min_interest = .55
        self.generate.side_effect = [json.dumps({"should_browse": True, "reason": "想试试新内容",
                                                 "next_check_minutes": 30}), decision(interest=.6)]
        await self.engine.start()
        self.assertIn("已提交评论", await self.engine.round())
        prompt = self.generate.await_args_list[0].args[0]
        self.assertIn("当前开启更积极探索", prompt[0]["content"])
        self.assertTrue(json.loads(prompt[1]["content"])["explore_more"])
        self.client.publish.assert_awaited_once()

    async def test_active_exploration_still_allows_model_to_rest_or_decline_comment(self):
        self.cfg.browse.autonomous_enabled = True
        self.cfg.browse.explore_more = True
        self.cfg.comment.min_interest = .55
        self.generate.return_value = json.dumps({"should_browse": False, "reason": "现在不想看",
                                                  "next_check_minutes": 30})
        await self.engine.start()
        self.assertIn("暂时不刷", await self.engine.round())
        self.client.understand.assert_not_awaited()
        self.now += 1801
        self.generate.side_effect = [json.dumps({"should_browse": True, "reason": "再看一条",
                                                 "next_check_minutes": 30}),
                                    decision(should_comment=False, interest=.8, comment="", evidence_quote="")]
        self.assertIn("看过不评论", await self.engine.round())
        self.client.publish.assert_not_awaited()


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = StateStore(Path(self.temp.name) / "state.sqlite3")
        self.store.initialize()
        self.store.set_pause(False, "")

    def reserve(self, uid="1", bvid=ITEM.bvid, text="深海生物的发光用途很有意思", now=NOW, limit=3):
        return self.store.reserve(uid, bvid, text, now, local_day_start(now, "Asia/Shanghai"), limit, 3600, .88)

    def test_existing_pause_reason_is_migrated_without_unpausing(self):
        for reason, expected in [("管理员手动暂停", "manual"), ("首次运行，请先登录并启动", "initial"),
                                 ("已在配置中禁用", "disabled"), ("运行异常：ValueError", "error")]:
            with self.subTest(reason=reason):
                self.store.set_pause(True, reason)
                with self.store.connect() as db:
                    db.execute("DELETE FROM meta WHERE key='pause_kind'")
                self.store.initialize()
                self.assertEqual(self.store.control()["pause_kind"], expected)
                self.assertEqual(self.store.control()["paused"], "1")

    def test_desire_history_and_recommendation_context_are_account_scoped(self):
        self.store.record_browse_decision("1", NOW, False, "休息一会儿", NOW + 3600)
        self.store.visit("1", ITEM.bvid, "publish", NOW, "skipped",
                         json.dumps({"title": ITEM.title, "reason": "话题不感兴趣", "comment": "PRIVATE_COMMENT"}))
        self.assertIsNone(self.store.latest_browse_decision("2"))
        self.assertEqual(self.store.browse_context("2"), [])
        self.assertNotIn("休息一会儿", self.store.history("2"))
        context = json.dumps(self.store.browse_context("1"), ensure_ascii=False)
        self.assertIn("话题不感兴趣", context)
        self.assertNotIn("PRIVATE_COMMENT", context)

    def test_pending_becomes_unknown_on_restart(self):
        self.assertEqual(self.reserve(), "reserved")
        self.store.initialize()
        self.assertEqual(self.store.stats("1", 0)["unresolved"], [ITEM.bvid])
        self.assertEqual(self.reserve(bvid="BV0987654321"), "unresolved")

    def test_deferred_video_with_unknown_attempt_is_never_retried(self):
        from dataclasses import asdict
        self.store.visit('1', ITEM.bvid, 'publish', NOW, 'deferred', json.dumps({'video': asdict(ITEM)}))
        self.assertEqual(self.store.deferred_videos('2'), [])
        self.assertEqual(self.store.deferred_videos('1'), [ITEM])
        self.assertEqual(self.reserve(), 'reserved')
        self.store.finish('1', ITEM.bvid, 'unknown')
        self.assertEqual(self.store.deferred_videos('1'), [])
        self.assertTrue(self.store.seen('1', ITEM.bvid, 'publish'))

    def test_structured_visit_details_remain_valid_beyond_1500_chars(self):
        detail = json.dumps({'title': ITEM.title, 'comment': '测试' * 900}, ensure_ascii=False)
        self.store.visit('1', ITEM.bvid, 'publish', NOW, 'deferred', detail)
        row = self.store.page_records('1', 'visits', ITEM.bvid, 1)['rows'][0]
        self.assertEqual(json.loads(row['detail'])['comment'], '测试' * 900)

    def test_atomic_reservation_prevents_concurrent_senders(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda b: self.reserve(bvid=b), [ITEM.bvid, "BV0987654321"]))
        self.assertEqual(results.count("reserved"), 1)
        self.assertEqual(self.store.stats("1", 0)["attempts"], 1)

    def test_limits_cooldown_and_account_isolation(self):
        self.assertEqual(self.reserve(limit=1), "reserved")
        self.store.finish("1", ITEM.bvid, "sent", "9")
        self.assertEqual(self.reserve(bvid="BV0987654321", text="小提琴的这段旋律令人难忘", limit=1), "daily_limit")
        self.assertEqual(self.reserve(bvid="BV0987654321", text="小提琴的这段旋律令人难忘"), "cooldown")
        self.assertEqual(self.reserve(uid="2"), "reserved")

    def test_text_dedup_survives_punctuation_and_restart(self):
        self.reserve()
        self.store.finish("1", ITEM.bvid, "sent", "9")
        self.store.initialize()
        result = self.reserve(bvid="BV0987654321", text="深海生物的发光用途，很有意思！", now=NOW+4000)
        self.assertEqual(result, "duplicate_text")

    def test_pause_is_rechecked_inside_transaction(self):
        self.store.set_pause(True, "stop")
        self.assertEqual(self.reserve(), "paused")


class ModelTests(unittest.TestCase):
    def test_daily_send_limits_accept_and_preserve_thirty_in_config_roundtrip(self):
        cfg = AutopilotConfig.model_validate({"comment": {"per_day": 30}, "replies": {"per_day": 30}})
        saved = AutopilotConfig.model_validate_json(cfg.model_dump_json())
        self.assertEqual(saved.comment.per_day, 30)
        self.assertEqual(saved.replies.per_day, 30)
        schema = AutopilotConfig.model_json_schema()
        for section in ("CommentSection", "ReplySection"):
            limits = schema["$defs"][section]["properties"]["per_day"]
            self.assertLessEqual(limits["minimum"], 30)
            self.assertGreaterEqual(limits["maximum"], 30)

    def test_minimum_and_maximum_interval_configuration_must_agree(self):
        with self.assertRaises(ValueError):
            AutopilotConfig.model_validate({"browse": {"interval_seconds": 1800, "max_check_minutes": 10}})
        cfg = AutopilotConfig()
        self.assertEqual(cfg.browse.interval_seconds, 600)
        self.assertEqual(cfg.browse.max_check_minutes, 20)

    def test_desire_validation_rejects_wrong_types_and_out_of_range_delays(self):
        base = {"should_browse": False, "reason": "晚些再看", "next_check_minutes": 30}
        self.assertFalse(parse_browse_decision(json.dumps(base), 30, 180).should_browse)
        for change in [{"should_browse": "true"}, {"next_check_minutes": True}, {"next_check_minutes": 5},
                       {"next_check_minutes": 181}, {"next_check_minutes": 30.5}, {"reason": ""},
                       {"reason": "test\nreason"}, {"extra": "bad"}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_browse_decision(json.dumps({**base, **change}), 30, 180)

    def test_rejects_ungrounded_quote_and_invalid_numeric_values(self):
        for change in [{"evidence_quote": "视频中不存在的细节"}, {"interest": float("nan")}, {"interest": True}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_decision(decision(**change), EVIDENCE, 160)

    def test_recommendations_ignore_ad_live_and_duplicates(self):
        base = {"bvid": ITEM.bvid, "id": 100, "goto": "av", "title": "视频", "duration": 60, "owner": {"mid": 2}}
        items = recommendations({"item": [base, base, {**base, "is_ad": True}, {**base, "goto": "live"}]})
        self.assertEqual(len(items), 1)

    def test_activity_supports_midnight_and_timezone(self):
        night = datetime(2026, 10, 19, 23, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
        self.assertTrue(active_time(night, "Asia/Shanghai", 22, 2))
        self.assertFalse(active_time(NOW, "Asia/Shanghai", 22, 2))
        self.assertFalse(active_time(night, "Asia/Shanghai", 9, 23))
        start = local_day_start(night, "Asia/Shanghai")
        self.assertEqual(datetime.fromtimestamp(start, ZoneInfo("Asia/Shanghai")).hour, 0)


if __name__ == "__main__":
    unittest.main()
