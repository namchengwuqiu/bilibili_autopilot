from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import asyncio
import json
import sqlite3
import unittest

from bilibili_api.exceptions import NetworkException, ResponseCodeException

import httpx

from local_bilibili_autopilot.client import BiliClient, PlatformError, VideoReadError
from local_bilibili_autopilot.config import AutopilotConfig
from local_bilibili_autopilot.engine import Autopilot
from local_bilibili_autopilot.models import DecisionValidationError, Evidence, ModelResponseError, VideoItem, parse_decision
from local_bilibili_autopilot.plugin import BilibiliAutopilotPlugin
from local_bilibili_autopilot.state import StateStore
from local_bilibili_autopilot.webui import local_time, record_row


try:
    from curl_cffi.requests.exceptions import ConnectionError as CurlConnectionError, Timeout as CurlTimeout
except ImportError:
    CURL_ERRORS = ()
else:
    CURL_ERRORS = (CurlConnectionError, CurlTimeout)


ITEM = VideoItem('BV1234567890',100,'深海发光','2','UP',180)
EVIDENCE = Evidence('字幕','深海生物利用发光器官吸引猎物，不同颜色与闪烁频率也能传递信号，这是海洋生态的一种适应方式。')
VALID = json.dumps({'should_comment':True,'interest':0.9,'comment':'原来深海的光也能传信，有意思。',
                    'reason':'对深海生态感兴趣','evidence_quote':'深海生物利用发光器官吸引猎物'},ensure_ascii=False)
INFO = {'aid':100,'bvid':ITEM.bvid,'title':ITEM.title,'duration':180,'owner':{'mid':2,'name':'UP'},'cid':10,'videos':1}


class ReadingRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.client = BiliClient(Path(self.temp.name)/'unused.json')
        self.addAsyncCleanup(self.client.close)
        self.resource = SimpleNamespace(get_info=AsyncMock(return_value=INFO),get_ai_conclusion=AsyncMock(),
            get_subtitle=AsyncMock(return_value={'subtitles':[{'lan':'zh-CN','subtitle_url':'https://aisubtitle.hdslb.com/test.json'}]}))
        self.sub = AsyncMock(return_value=EVIDENCE.text)
        self.patch = patch('local_bilibili_autopilot.client.video.Video',return_value=self.resource)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.client._subtitle_text = self.sub

    async def test_summary_unavailable_uses_subtitles_and_keeps_stage_warning(self):
        failures = [ResponseCodeException(-504,'PRIVATE'),asyncio.TimeoutError('PRIVATE'),httpx.ReadTimeout('PRIVATE')]
        failures += [NetworkException(code,'PRIVATE') for code in (408,500,502,503,504)]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                self.resource.get_ai_conclusion.side_effect=failure
                item,evidence=await self.client.understand(ITEM,12000)
                self.assertEqual(evidence.text,EVIDENCE.text)
                self.assertIn('字幕',evidence.source)
                self.assertIn('AI 摘要',evidence.read_warning)
                self.assertNotIn('PRIVATE',evidence.read_warning)
                self.assertEqual(item.bvid,ITEM.bvid)

    @unittest.skipUnless(CURL_ERRORS, 'curl_cffi 未安装')
    async def test_actual_curl_transport_errors_use_subtitles(self):
        for error in CURL_ERRORS:
            with self.subTest(error=error.__name__):
                self.resource.get_ai_conclusion.side_effect = error('PRIVATE')
                _, evidence = await self.client.understand(ITEM, 12000)
                self.assertEqual(evidence.text, EVIDENCE.text)
                self.assertIn('AI 摘要', evidence.read_warning)
                self.assertNotIn('PRIVATE', evidence.read_warning)

    async def test_bilibili_http_rejections_do_not_bypass_risk_controls(self):
        for code in (401,403,412,429):
            with self.subTest(code=code):
                self.resource.get_ai_conclusion.side_effect = NetworkException(code,'PRIVATE')
                with self.assertRaises(PlatformError) as raised:
                    await self.client.understand(ITEM,12000)
                self.assertEqual(raised.exception.code,code)
                self.assertIn('AI 摘要',raised.exception.operation)
                self.assertNotIn('PRIVATE',str(raised.exception))
                self.resource.get_subtitle.assert_not_awaited()

    async def test_permanent_http_read_error_is_not_retryable(self):
        self.resource.get_info.side_effect = NetworkException(404,'PRIVATE')
        with self.assertRaises(VideoReadError) as raised:
            await self.client.understand(ITEM,12000)
        self.assertFalse(raised.exception.retryable)
        self.assertIn('读取视频信息',str(raised.exception))
        self.assertNotIn('PRIVATE',str(raised.exception))

    async def test_cancellation_is_not_converted_to_a_network_error(self):
        self.resource.get_ai_conclusion.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.client.understand(ITEM,12000)
        self.resource.get_subtitle.assert_not_awaited()

    async def test_summary_risk_rejection_does_not_read_another_endpoint(self):
        self.resource.get_ai_conclusion.side_effect=ResponseCodeException(-412,'PRIVATE')
        with self.assertRaises(PlatformError) as raised:
            await self.client.understand(ITEM,12000)
        self.assertEqual(raised.exception.operation,'读取 AI 摘要')
        self.resource.get_subtitle.assert_not_awaited()
        self.sub.assert_not_awaited()

    async def test_missing_subtitles_after_summary_error_remains_retryable(self):
        self.resource.get_ai_conclusion.side_effect=ResponseCodeException(-504,'PRIVATE')
        self.resource.get_subtitle.return_value={'subtitles':[]}
        with self.assertRaises(VideoReadError) as raised:
            await self.client.understand(ITEM,12000)
        self.assertTrue(raised.exception.retryable)
        self.assertIn('没有可用字幕',str(raised.exception))

    async def test_short_subtitles_after_summary_error_remain_retryable(self):
        self.resource.get_ai_conclusion.side_effect=ResponseCodeException(-504,'PRIVATE')
        self.sub.return_value='太短'
        with self.assertRaises(VideoReadError) as raised:
            await self.client.understand(ITEM,12000)
        self.assertTrue(raised.exception.retryable)

    async def test_both_content_endpoints_failed_reports_both_stages(self):
        self.resource.get_ai_conclusion.side_effect=ResponseCodeException(-504,'PRIVATE')
        self.resource.get_subtitle.side_effect=asyncio.TimeoutError('PRIVATE')
        with self.assertRaises(VideoReadError) as raised:
            await self.client.understand(ITEM,12000)
        self.assertIn('AI 摘要',str(raised.exception))
        self.assertIn('字幕列表',str(raised.exception))
        self.assertNotIn('PRIVATE',str(raised.exception))

    async def test_info_failure_has_stage_and_never_fetches_summary(self):
        self.resource.get_info.side_effect=asyncio.TimeoutError('PRIVATE')
        with self.assertRaises(VideoReadError) as raised:
            await self.client.understand(ITEM,12000)
        self.assertIn('视频信息',str(raised.exception))
        self.resource.get_ai_conclusion.assert_not_awaited()

    async def test_good_summary_avoids_subtitle_request(self):
        self.resource.get_ai_conclusion.return_value={'model_result':{'summary':EVIDENCE.text}}
        _,evidence=await self.client.understand(ITEM,12000)
        self.assertEqual(evidence.read_warning,'')
        self.resource.get_subtitle.assert_not_awaited()

    async def test_http_risk_or_login_error_does_not_read_subtitles(self):
        for code in (401,403,412,429):
            with self.subTest(code=code):
                response=httpx.Response(code,request=httpx.Request('GET','https://example.invalid'))
                self.resource.get_ai_conclusion.side_effect=httpx.HTTPStatusError('PRIVATE',request=response.request,response=response)
                with self.assertRaises(PlatformError) as raised:
                    await self.client.understand(ITEM,12000)
                self.assertEqual(raised.exception.code,code)
                self.resource.get_subtitle.assert_not_awaited()


class EngineRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.now=100000
        self.cfg=AutopilotConfig()
        self.cfg.browse.autonomous_enabled=False
        self.cfg.browse.active_start_hour=self.cfg.browse.active_end_hour=0
        self.cfg.comment.publish_enabled=True
        self.client=AsyncMock()
        self.client.uid,self.client.name='1','模拟账号'
        self.client.check_account.return_value='1'
        self.client.recommended.return_value=[ITEM]
        self.client.understand.return_value=ITEM,EVIDENCE
        self.client.publish.return_value='1000'
        self.generate=AsyncMock(return_value=VALID)
        self.store=StateStore(Path(self.temp.name)/'state.sqlite3')
        self.engine=Autopilot(self.client,self.store,lambda:self.cfg,self.generate,
                             AsyncMock(return_value='自然随和'),lambda:self.now,review=AsyncMock(return_value=json.dumps({"supported":True,"kind":"factual","reason":"模拟复核通过"})))
        await self.engine.initialize()
        await self.engine.start()

    async def test_retry_survives_restart_and_recovers_video_no_longer_recommended(self):
        self.client.understand.side_effect=VideoReadError('下载字幕文件超时',True)
        await self.engine.round()
        self.now+=599
        self.assertEqual(self.store.retry_videos('1','publish',self.now),[])
        await self.engine.round()
        self.assertEqual(self.client.understand.await_count,1)
        self.now+=1
        await self.engine.initialize()
        self.client.recommended.return_value=[]
        self.client.understand.side_effect=None
        self.assertIn('已提交评论',await self.engine.round())
        self.client.publish.assert_awaited_once()
        self.assertEqual(self.store.stats('1',0)['views'],2)
        self.assertTrue(self.store.seen('1',ITEM.bvid,'publish',now=self.now+99999))

    async def test_repeated_failures_stop_after_two_rechecks(self):
        self.client.understand.side_effect=VideoReadError('字幕网络失败',True)
        await self.engine.round()
        self.now+=600
        await self.engine.round()
        self.now+=1199
        await self.engine.round()
        self.assertEqual(self.client.understand.await_count,2)
        self.now+=1
        await self.engine.round()
        await self.engine.initialize()
        self.now+=99999
        await self.engine.round()
        self.assertEqual(self.client.understand.await_count,3)
        row=self.store.page_records('1','visits','',1)['rows'][0]
        self.assertEqual(row['failures'],3)
        self.assertEqual(row['retry_at'],0)
        self.client.publish.assert_not_awaited()
        self.assertFalse(self.engine.paused)

    async def test_rechecks_consume_daily_analysis_quota(self):
        self.cfg.browse.videos_per_day=1
        self.client.understand.side_effect=VideoReadError('字幕超时',True)
        await self.engine.round()
        self.now+=600
        self.assertIn('额度已用完',await self.engine.round())
        self.assertEqual(self.client.understand.await_count,1)

    async def test_interrupted_recheck_resumes_after_restart_without_resetting_limits(self):
        self.client.understand.side_effect = VideoReadError('模拟临时读取失败',True)
        await self.engine.round()
        self.now += 600
        entered = asyncio.Event()

        async def interrupted_read(*args):
            entered.set()
            await asyncio.Event().wait()

        self.client.understand.side_effect = interrupted_read
        task = asyncio.create_task(self.engine.round())
        try:
            await asyncio.wait_for(entered.wait(),2)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        await self.engine.initialize()
        row = self.store.page_records('1','visits','',1)['rows'][0]
        self.assertEqual((row['failures'],row['retry_at']),(2,self.now+1200))
        self.assertEqual(self.store.stats('1',0)['views'],2)
        self.now += 100
        await self.engine.initialize()
        self.assertEqual(self.store.page_records('1','visits','',1)['rows'][0]['retry_at'],row['retry_at'])
        self.client.understand.side_effect = None
        self.client.recommended.return_value = []
        await self.engine.round()
        self.assertEqual(self.client.understand.await_count,2)
        self.now = row['retry_at']
        self.assertIn('已提交评论',await self.engine.round())
        self.assertEqual(self.store.stats('1',0)['views'],3)
        self.client.publish.assert_awaited_once()

    async def _check_filtered_queue(self, outcome):
        videos = [replace(ITEM,bvid=f'BV123456789{i}',up_uid='2' if i<3 else '3') for i in range(4)]
        for item in videos:
            await self.engine.record('1',item,'publish',outcome,'待后续处理')
            self.now += 1
        self.cfg.browse.blocked_up_uids = ['2']
        self.client.recommended.return_value = []
        self.client.understand.return_value = videos[-1],EVIDENCE
        self.now += 1000
        self.assertIn('已提交评论',await self.engine.round())
        self.client.understand.assert_awaited_once_with(videos[-1],self.cfg.model.content_max_chars)
        self.assertEqual(self.client.publish.await_args.args[0].bvid,videos[-1].bvid)

    async def test_filtered_retries_do_not_starve_later_due_videos(self):
        await self._check_filtered_queue('read_error')

    async def test_filtered_deferred_videos_do_not_starve_later_entries(self):
        await self._check_filtered_queue('deferred')

    async def test_format_error_is_regenerated_once_with_specific_reason(self):
        self.generate.side_effect=['invalid PRIVATE',VALID]
        self.assertIn('已提交评论',await self.engine.round())
        self.assertEqual(self.generate.await_count,2)
        correction=self.generate.await_args.args[0][-1]['content']
        self.assertIn('json_format',correction)
        self.assertNotIn('PRIVATE',correction)
        self.client.publish.assert_awaited_once()

    async def test_ungrounded_regeneration_never_posts_and_records_evidence_category(self):
        bad=json.loads(VALID);bad['evidence_quote']='未提供的视频内容'
        self.generate.return_value=json.dumps(bad)
        await self.engine.round()
        self.assertEqual(self.generate.await_count,2)
        self.client.publish.assert_not_awaited()
        row=self.store.page_records('1','visits','',1)['rows'][0]
        self.assertEqual(json.loads(row['detail'])['validation_error'],'evidence')
        self.assertEqual(row['failures'],1)
        self.assertEqual(row['retry_at'],self.now+600)

    async def test_pause_after_invalid_output_prevents_regeneration(self):
        async def pause(*args):
            await self.engine.pause('管理员手动暂停',kind='manual')
            return 'invalid'
        self.generate.side_effect=pause
        await self.engine.round()
        self.generate.assert_awaited_once()
        self.client.publish.assert_not_awaited()

    async def test_model_service_failure_keeps_processing_other_videos(self):
        second=replace(ITEM,bvid='BV0987654321')
        self.client.recommended.return_value=[ITEM,second]
        self.client.understand.side_effect=[(ITEM,EVIDENCE),(second,EVIDENCE)]
        self.generate.side_effect=[ModelResponseError('没有有效正文'),VALID]
        self.assertIn('已提交评论',await self.engine.round())
        self.assertFalse(self.engine.paused)
        self.assertEqual(self.client.publish.await_args.args[0].bvid,second.bvid)


class StoreRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'state.sqlite3'
        self.store=StateStore(self.path)
        self.store.initialize()
        self.store.set_pause(False,'')

    def test_interrupted_reads_count_once_and_stop_after_three_analyses(self):
        detail = json.dumps({'title':ITEM.title,'video':asdict(ITEM),'reason':'正在读取'},ensure_ascii=False)
        now = 100
        for failures in range(1,4):
            self.store.visit('1',ITEM.bvid,'publish',now,'reading',detail)
            self.store.initialize(now+10)
            row = self.store.page_records('1','visits','',1)['rows'][0]
            self.assertEqual(row['outcome'],'read_error')
            self.assertEqual(row['failures'],failures)
            self.assertEqual(row['ts'],100)
            self.assertEqual(json.loads(row['detail'])['video'],asdict(ITEM))
            self.assertEqual(self.store.stats('1',0)['views'],failures)
            self.store.initialize(now+20)
            self.assertEqual(self.store.page_records('1','visits','',1)['rows'][0],row)
            if failures < 3:
                self.assertEqual(row['retry_at'],now+10+600*failures)
                self.assertTrue(self.store.seen('1',ITEM.bvid,'publish',now=row['retry_at']-1))
                self.assertFalse(self.store.seen('1',ITEM.bvid,'publish',now=row['retry_at']))
                now = row['retry_at']
            else:
                self.assertEqual(row['retry_at'],0)
                self.assertTrue(self.store.seen('1',ITEM.bvid,'publish',now=999999))
                self.assertEqual(self.store.retry_videos('1','publish',999999),[])

    def test_restart_never_recovers_reading_with_any_send_attempt(self):
        for index,status in enumerate(('pending','sent','unknown','cancelled','rejected','acknowledged')):
            uid = str(index+10)
            self.store.visit(uid,ITEM.bvid,'publish',100,'reading',json.dumps({'video':asdict(ITEM)}))
            self.assertEqual(self.store.reserve(uid,ITEM.bvid,'模拟评论',100,0,12,0,.88),'reserved')
            if status == 'acknowledged':
                self.store.finish(uid,ITEM.bvid,'unknown')
                self.assertTrue(self.store.acknowledge(uid,ITEM.bvid))
            elif status != 'pending':
                self.store.finish(uid,ITEM.bvid,status,'123')
            self.store.initialize(200)
            row = self.store.page_records(uid,'visits','',1)['rows'][0]
            self.assertEqual((row['outcome'],row['failures'],row['retry_at']),('reading',0,0))
            self.assertEqual(self.store.retry_videos(uid,'publish',999999),[])
            self.assertTrue(self.store.seen(uid,ITEM.bvid,'publish',now=999999))

    def test_queue_filters_before_limit_and_preserves_excluded_records(self):
        items = [replace(ITEM,bvid=f'BV123456789{i}',duration=180 if i<3 else 60) for i in range(7)]
        for item in items:
            self.store.visit('1',item.bvid,'publish',100,'read_error',json.dumps({'video':asdict(item)}))
        self.assertEqual(self.store.retry_videos('1','publish',1000,lambda item:item.duration<100),items[3:6])
        self.assertEqual(self.store.retry_videos('1','publish',1000),items[:3])

    def test_old_records_without_metadata_do_not_hide_new_retry_queue(self):
        for index in range(4):
            self.store.visit('1',f'BV{index:010}','publish',100+index,'model_error','旧记录没有元数据')
        self.store.visit('1',ITEM.bvid,'publish',200,'read_error',json.dumps({'video':asdict(ITEM)}))
        self.assertEqual(self.store.retry_videos('1','publish',10000),[ITEM])

    def test_any_send_attempt_blocks_recheck_even_if_failure_record_is_retryable(self):
        self.store.visit('1',ITEM.bvid,'publish',100,'model_error',json.dumps({'video':asdict(ITEM)}))
        for index,status in enumerate(('sent','unknown','cancelled','rejected','acknowledged')):
            with self.subTest(status=status):
                uid=str(index+10)
                self.store.visit(uid,ITEM.bvid,'publish',100,'model_error',json.dumps({'video':asdict(ITEM)}))
                self.assertEqual(self.store.reserve(uid,ITEM.bvid,'唯一文字',200,0,12,0,.88),'reserved')
                if status == 'acknowledged':
                    self.store.finish(uid,ITEM.bvid,'unknown')
                    self.assertTrue(self.store.acknowledge(uid,ITEM.bvid))
                else:
                    self.store.finish(uid,ITEM.bvid,status,'123')
                self.assertTrue(self.store.seen(uid,ITEM.bvid,'publish',now=10000))
                self.assertEqual(self.store.retry_videos(uid,'publish',10000),[])
        self.assertFalse(self.store.seen('2',ITEM.bvid,'publish',now=10000))

    def test_first_time_update_and_send_time_are_separate_and_updates_sort_first(self):
        self.store.visit('1',ITEM.bvid,'publish',100,'reading','{}')
        self.store.visit('1','BV0987654321','publish',200,'skipped','{}')
        self.store.reserve('1',ITEM.bvid,'唯一评论',300,0,12,0,.88)
        self.store.finish('1',ITEM.bvid,'sent','123')
        self.store.visit('1',ITEM.bvid,'publish',310,'sent','{}')
        row=self.store.page_records('1','visits','',1)['rows'][0]
        self.assertEqual((row['ts'],row['updated_at'],row['send_ts']),(100,310,300))
        shown=record_row(row,'visits','Asia/Shanghai')
        self.assertEqual(shown['first_time'],local_time(100,'Asia/Shanghai'))
        self.assertEqual(shown['updated_time'],local_time(310,'Asia/Shanghai'))
        self.assertEqual(shown['send_time'],local_time(300,'Asia/Shanghai'))
        self.assertEqual(shown['time'],shown['updated_time'])
        self.assertEqual(self.store.stats('1',250)['views'],0)

    def test_new_day_reanalysis_counts_new_day_instead_of_first_seen_day(self):
        self.store.visit('1',ITEM.bvid,'preview',100,'reading','{}')
        self.store.visit('1',ITEM.bvid,'preview',101,'model_error','{}')
        self.store.visit('1',ITEM.bvid,'preview',90000,'reading','{}')
        self.store.visit('1',ITEM.bvid,'preview',90001,'skipped','{}')
        self.assertEqual(self.store.stats('1',86400)['views'],1)
        self.assertEqual(self.store.stats('1',0)['views'],2)

    def test_legacy_schema_upgrade_preserves_data_and_only_seeds_counts_once(self):
        self.path.unlink()
        with sqlite3.connect(self.path) as db:
            db.executescript('''CREATE TABLE visits(uid TEXT,bvid TEXT,mode TEXT,ts REAL,outcome TEXT,detail TEXT,PRIMARY KEY(uid,bvid,mode));
                CREATE TABLE attempts(uid TEXT,bvid TEXT,ts REAL,text TEXT,text_hash TEXT,status TEXT,rpid TEXT,PRIMARY KEY(uid,bvid));''')
            db.execute('INSERT INTO visits VALUES (?,?,?,?,?,?)',('1',ITEM.bvid,'publish',100,'sent','{}'))
            db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,?)',('1',ITEM.bvid,500,'评论','hash','sent','123'))
            db.execute('INSERT INTO visits VALUES (?,?,?,?,?,?)',('1','BV0987654321','preview',200,'model_error','旧记录'))
        self.store.initialize()
        self.store.initialize()
        row=self.store.page_records('1','visits',ITEM.bvid,1)['rows'][0]
        self.assertEqual((row['ts'],row['updated_at'],row['send_ts']),(100,500,500))
        self.assertEqual(self.store.stats('1',0)['views'],2)
        self.assertEqual(self.store.stats('1',0)['attempts'],1)
        self.assertTrue(self.store.seen('1','BV0987654321','preview',now=799))
        self.assertFalse(self.store.seen('1','BV0987654321','preview',now=800))


class ModelRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_and_failed_model_responses_try_next_candidate_once(self):
        for failure in [{'success':True,'response':'','reasoning':'PRIVATE'},
                        {'success':False,'error':'PRIVATE service error'}]:
            with self.subTest(failure=failure['success']):
                plugin=BilibiliAutopilotPlugin()
                generate=AsyncMock(side_effect=[failure,{'success':True,'response':VALID}])
                warnings=[]
                plugin._ctx=SimpleNamespace(llm=SimpleNamespace(generate=generate),
                    logger=SimpleNamespace(warning=lambda *args, warnings=warnings:warnings.append(args)))
                cfg=AutopilotConfig.model_validate({'model':{'candidates':['first','second']}})
                self.assertEqual(await plugin._generate('prompt',cfg),VALID)
                self.assertEqual([call.kwargs['model_name'] for call in generate.await_args_list],['first','second'])
                self.assertNotIn('PRIVATE',str(warnings))

    def test_json_and_evidence_failures_have_distinct_public_codes(self):
        for value,expected in [('PRIVATE','json_format'),(json.dumps({}),'fields')]:
            with self.subTest(expected=expected),self.assertRaises(DecisionValidationError) as raised:
                parse_decision(value,EVIDENCE,160)
            self.assertEqual(raised.exception.code,expected)
            self.assertNotIn('PRIVATE',str(raised.exception))
