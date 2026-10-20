from dataclasses import replace
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import asyncio
import json
import unittest

from local_bilibili_autopilot.client import UnknownPublish
from local_bilibili_autopilot.config import AutopilotConfig
from local_bilibili_autopilot.engine import Autopilot
from local_bilibili_autopilot.models import IncomingReply, ModelTimeoutError, parse_reply_decision
from local_bilibili_autopilot.state import StateStore


NOW = datetime(2026,10,19,12,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
MESSAGE = IncomingReply('BV1234567890',100,500,900,500,'2','对方','你说的是哪段旋律？',
                        '这段旋律的处理很有意思','这段旋律的处理很有意思',NOW-100)
ANSWER = json.dumps({'should_reply':True,'reply':'就是主题旋律切换的那一段，听着挺有意思。','reason':'对方在认真讨论，可以自然接话'},ensure_ascii=False)


class ReplyEngineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.cfg=AutopilotConfig()
        self.cfg.replies.min_interval_seconds=600
        self.cfg.comment.publish_enabled=True
        self.cfg.browse.active_start_hour=self.cfg.browse.active_end_hour=0
        self.now=NOW
        self.client=AsyncMock()
        self.client.uid,self.client.name='1','模拟账号'
        self.client.check_account.return_value='1'
        self.client.incoming_replies.return_value=[MESSAGE]
        self.client.publish_reply.return_value='600'
        self.generate=AsyncMock(return_value=ANSWER)
        self.store=StateStore(Path(self.temp.name)/'state.sqlite3')
        self.engine=Autopilot(self.client,self.store,lambda:self.cfg,self.generate,AsyncMock(return_value='自然随和'),lambda:self.now)
        await self.engine.initialize()
        await self.engine.start()
        self.assertEqual(self.store.reserve('1',MESSAGE.bvid,MESSAGE.root_text,NOW-7200,0,3,3600,.88),'reserved')
        self.store.finish('1',MESSAGE.bvid,'sent','500')

    async def test_model_timeout_keeps_running_and_retries_at_next_poll(self):
        self.generate.side_effect = ModelTimeoutError("候选模型均超时")
        self.assertIn('超时', await self.engine.reply_round())
        self.assertFalse(self.engine.paused)
        self.assertEqual(self.store.control()['paused'], '0')
        self.client.publish_reply.assert_not_awaited()
        self.now += 601
        self.generate.side_effect = None
        self.assertIn('已提交跟进回复', await self.engine.reply_round())
        self.client.publish_reply.assert_awaited_once()
        self.now += 601
        await self.engine.reply_round()
        self.client.publish_reply.assert_awaited_once()

    async def test_legacy_unsent_timeout_is_retryable_after_reload(self):
        self.store.receive_replies('1', [MESSAGE])
        self.store.decide_reply('1', MESSAGE.source_rpid, 'publish', self.now,
                                'model_error', '', '模型调用失败：TimeoutError')
        self.store.initialize()
        self.assertIn('已提交跟进回复', await self.engine.reply_round())
        self.client.publish_reply.assert_awaited_once()

    async def test_live_reply_targets_incoming_message_and_is_not_repeated(self):
        self.assertIn('已提交跟进回复',await self.engine.reply_round())
        args=self.client.publish_reply.await_args.args
        self.assertEqual(args[0].source_rpid,900)
        self.assertEqual(args[0].root_rpid,500)
        self.assertTrue(args[2]())
        self.now+=601
        await self.engine.reply_round()
        self.client.publish_reply.assert_awaited_once()
        self.assertEqual(self.store.stats('1',0)['reply_attempts'],1)
        self.assertIn('600',self.store.own_comments('1'))
        self.client.publish.assert_not_awaited()

    async def test_preview_while_paused_never_sends_and_does_not_consume_live_opportunity(self):
        await self.engine.pause('管理员手动暂停',kind='manual')
        self.assertIn('预览跟进回复',await self.engine.reply_round(preview=True))
        self.client.publish_reply.assert_not_awaited()
        self.assertEqual(self.store.stats('1',0)['reply_attempts'],0)
        await self.engine.start()
        self.now+=601
        self.assertIn('已提交跟进回复',await self.engine.reply_round())
        self.client.publish_reply.assert_awaited_once()

    async def test_publish_switch_off_only_records_preview(self):
        self.cfg.comment.publish_enabled=False
        self.assertIn('预览跟进回复',await self.engine.reply_round())
        self.client.publish_reply.assert_not_awaited()

    async def test_model_can_decline_and_decline_survives_restart(self):
        self.generate.return_value=json.dumps({'should_reply':False,'reply':'','reason':'没有需要继续讨论的内容'})
        self.assertIn('不接话',await self.engine.reply_round())
        await self.engine.initialize();self.now+=601
        await self.engine.reply_round()
        self.generate.assert_awaited_once()
        self.client.publish_reply.assert_not_awaited()

    async def test_reply_to_own_followup_continues_same_thread(self):
        await self.engine.reply_round()
        self.now+=601
        message=replace(MESSAGE,source_rpid=901,target_rpid=600,text='原来如此，那段确实有变化',received_at=self.now)
        self.client.incoming_replies.return_value=[message]
        self.generate.return_value=json.dumps({'should_reply':True,'reply':'嗯嗯，这样前后听起来就能对上了。','reason':'自然接一句'})
        self.client.publish_reply.return_value='601'
        self.assertIn('已提交跟进回复',await self.engine.reply_round())
        known=self.client.incoming_replies.await_args.args[0]
        self.assertEqual(known['600']['root_rpid'],500)
        self.assertEqual(known['600']['root_text'],MESSAGE.root_text)
        self.assertEqual(self.client.publish_reply.await_args.args[0].source_rpid,901)

    async def test_per_thread_limit_prevents_endless_conversation(self):
        self.cfg.replies.per_thread=1
        await self.engine.reply_round();self.now+=601
        self.client.incoming_replies.return_value=[replace(MESSAGE,source_rpid=901,target_rpid=600,received_at=self.now)]
        await self.engine.reply_round()
        self.client.publish_reply.assert_awaited_once()
        self.generate.assert_awaited_once()

    async def test_cooldown_keeps_unprocessed_message_until_next_check(self):
        self.now=NOW-7100
        self.assertIn('待处理消息已保留',await self.engine.reply_round())
        self.generate.assert_not_awaited()
        self.assertEqual(len(self.store.pending_replies('1','publish',0)),1)
        self.now=NOW
        self.assertIn('已提交跟进回复',await self.engine.reply_round())

    async def test_final_cooldown_check_preserves_message_for_retry(self):
        original = self.store.reserve_reply
        self.store.reserve_reply = lambda *args: 'cooldown'
        await self.engine.reply_round()
        self.assertEqual(len(self.store.pending_replies('1', 'publish', 0)), 1)
        self.assertFalse(self.engine.paused)
        self.store.reserve_reply = original
        self.now += self.cfg.replies.interval_seconds
        self.assertIn('已提交跟进回复', await self.engine.reply_round())
        self.client.publish_reply.assert_awaited_once()

    async def test_pause_during_model_generation_prevents_send(self):
        async def paused(*args):
            await self.engine.pause('管理员手动暂停',kind='manual')
            return ANSWER
        self.generate.side_effect=paused
        await self.engine.reply_round()
        self.client.publish_reply.assert_not_awaited()
        self.assertEqual(self.store.stats('1',0)['reply_attempts'],0)

    async def test_configuration_change_during_model_generation_prevents_send(self):
        async def disabled(*args):
            self.cfg.replies.enabled=False
            return ANSWER
        self.generate.side_effect=disabled
        await self.engine.reply_round()
        self.client.publish_reply.assert_not_awaited()

    async def test_unknown_result_blocks_all_posts_and_ack_does_not_allow_resend(self):
        self.client.publish_reply.side_effect=UnknownPublish('timeout with secret')
        result=await self.engine.reply_round()
        self.assertIn('待核对',result);self.assertNotIn('secret',result)
        self.assertTrue(self.engine.paused)
        self.assertEqual(self.store.stats('1',0)['unresolved'],[MESSAGE.bvid])
        await self.engine.initialize()
        self.assertIn('待核对',await self.engine.start())
        self.assertTrue(self.store.acknowledge('1',MESSAGE.bvid))
        await self.engine.start();self.now+=7200
        await self.engine.reply_round()
        self.client.publish_reply.assert_awaited_once()

    async def test_reply_checks_are_independent_of_desire_and_video_quota(self):
        self.store.record_browse_decision('1',NOW,False,'暂时不刷',NOW+7200)
        self.cfg.browse.videos_per_day=1
        self.store.visit('1','BV0987654321','publish',NOW,'skipped','看过')
        self.assertIn('已提交跟进回复',await self.engine.reply_round())
        self.assertEqual(await self.engine.next_check_delay(),600)
        self.client.recommended.assert_not_awaited()

    async def test_reply_analysis_and_send_quotas_are_separate(self):
        self.cfg.replies.analyses_per_day=1
        await self.engine.reply_round(preview=True)
        self.now+=601
        self.assertIn('分析额度已满',await self.engine.reply_round())
        self.client.publish_reply.assert_not_awaited()

    async def test_disabled_paused_and_outside_active_time_do_not_fetch_feed(self):
        self.cfg.replies.enabled=False
        await self.engine.reply_round()
        self.cfg.replies.enabled=True
        await self.engine.pause('manual',kind='manual')
        await self.engine.reply_round()
        await self.engine.start()
        self.cfg.browse.active_start_hour,self.cfg.browse.active_end_hour=0,1
        await self.engine.reply_round()
        self.client.incoming_replies.assert_not_awaited()

    async def test_pending_attempt_is_unknown_after_restart_and_is_account_scoped(self):
        self.assertEqual(self.store.reserve_reply('1',MESSAGE,'一条合法且不重复的跟进内容',NOW,0,3,600,3,.88),'reserved')
        self.store.initialize()
        self.assertEqual(self.store.stats('1',0)['unresolved'],[MESSAGE.bvid])
        self.assertEqual(self.store.stats('2',0)['unresolved'],[])
        self.assertFalse(self.store.acknowledge('2',MESSAGE.bvid))

    async def test_reply_reservation_is_atomic_across_concurrent_senders(self):
        from concurrent.futures import ThreadPoolExecutor
        def reserve(source):
            message=replace(MESSAGE,source_rpid=source)
            return self.store.reserve_reply('1',message,'一条合法的跟进回复',NOW,0,3,600,3,.88)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(reserve,[900,901]))
        self.assertEqual(results.count('reserved'),1)
        self.assertEqual(self.store.stats('1',0)['reply_attempts'],1)

    async def test_unknown_followup_blocks_root_comment_reservation_too(self):
        self.assertEqual(self.store.reserve_reply('1',MESSAGE,'一条合法的跟进回复',NOW,0,3,600,3,.88),'reserved')
        self.store.finish_reply('1',900,'unknown')
        self.assertEqual(self.store.reserve('1','BV0987654321','新的独立视频评论',NOW+7200,0,3,600,.88),'unresolved')

    async def test_invalid_model_output_never_posts(self):
        self.generate.return_value='{"should_reply":true'
        self.assertIn('校验失败',await self.engine.reply_round())
        self.client.publish_reply.assert_not_awaited()


class ReplyModelTests(unittest.TestCase):
    def test_strict_schema_and_length(self):
        self.assertTrue(parse_reply_decision(ANSWER,160).should_reply)
        base={'should_reply':True,'reply':'谢谢回复','reason':'可以接话'}
        for change in [{'should_reply':'true'},{'reply':''},{'reply':'a'*161},{'reply':'test\ntext'},
                       {'reason':''},{'extra':'bad'}]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                parse_reply_decision(json.dumps({**base,**change}),160)


if __name__=='__main__':
    unittest.main()
