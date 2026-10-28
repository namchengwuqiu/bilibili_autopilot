"""将插件的私有记录整理成管理页数据，不导出凭据或模型思考原文。"""

from datetime import datetime
from typing import Any, Dict
from zoneinfo import ZoneInfo

import json

from .sharing import SHARE_OUTCOMES
from .state import MAX_FAILURES, RETRY_OUTCOMES


OUTCOMES = {
    "reading": "正在读取", "read_error": "读取失败", "model_error": "模型调用失败", "model_timeout": "模型超时",
    "invalid_decision": "判断校验失败", "skipped": "看过不评论 / 跳过", "would_comment": "拟发表评论",
    "sent": "已提交评论", "rejected": "平台拒绝", "unknown": "发送结果待核对",
    "pending": "发送中", "cancelled": "已取消", "acknowledged": "已人工核对", "blocked": "发送被拦截",
    "deferred": "候选保留中（评论冷却）",
    "quota_wait": "候选保留中（今日额度已满）", "expired": "超过有效期",
    "evicted": "候选池已满，择优淘汰",
    "new": "等待判断", "would_reply": "拟跟进回复",
}
BLOCKS = {"daily_limit": "今日评论额度已满", "cooldown": "仍在评论间隔中", "duplicate_text": "评论完全重复",
          "similar_text": "评论过于相似", "duplicate_video": "该视频已尝试评论", "paused": "已暂停",
          "unresolved": "存在待核对的发送结果"}
BLOCKS.update(thread_limit="该讨论已达到跟进上限",duplicate_reply="该消息已处理",
              untracked_thread="根评论不在本地已发送记录中",untracked_target="被回复的评论不在本地记录中")


def local_time(timestamp: float, timezone: str) -> str:
    return datetime.fromtimestamp(timestamp, ZoneInfo(timezone)).strftime("%m-%d %H:%M:%S")


def record_row(row: Dict[str, Any], kind: str, timezone: str) -> Dict[str, Any]:
    result = {"time": local_time(row.get("ts",row.get("received_at")), timezone), "title": "", "bvid": "", "url": "",
              "outcome": "", "mode": "", "interest": "", "comment": "", "reason": "",
              "source": "", "rpid": "", "summary": "", "next_time": ""}
    result.update(sender='',incoming='',target='',source_rpid='',first_time='',updated_time='',send_time='',retry_status='')
    for field, timestamp in (('first_time',row.get('first_ts')),('updated_time',row.get('updated_at')),
                             ('send_time',row.get('send_ts'))):
        if timestamp is not None:
            result[field] = local_time(timestamp,timezone)
    if kind == 'visits' and row.get('updated_at') is not None:
        result['time'] = result['updated_time']
    result['retry_status'] = retry_status(row,timezone)
    if kind == "shares":
        result.update(title=row['title'],bvid=row['bvid'],url='https://www.bilibili.com/video/'+row['bvid'],
                      outcome=SHARE_OUTCOMES.get(row['status'],row['status']),mode='评论后群分享',
                      reason=f"QQ 群 {row['group_id']}；{row['detail']}",rpid=row['message_id'])
        result['summary'] = result['reason'][:100]
        return result
    if kind in {'replies','reply_attempts'}:
        message = json.loads(row['payload'])
        decision = row['decision']
        outcome = row.get('status') or decision.get('outcome','new')
        result.update(first_time=local_time(message.get('received_at',row.get('received_at',0)),timezone),
                      updated_time=local_time(decision['updated_at'],timezone) if decision.get('updated_at') is not None else '',
                      retry_status=retry_status(decision,timezone))
        result.update(time=local_time(row.get('ts',row.get('received_at')),timezone),
            title='跟进讨论',bvid=row['bvid'],url='https://www.bilibili.com/video/'+row['bvid'],
            sender=message.get('sender_name',''),incoming=message.get('text',''),target=message.get('target_text','')[:500],
            source_rpid=str(row['source_rpid']),outcome=OUTCOMES.get(outcome,outcome),
            mode='实际发送' if kind=='reply_attempts' else '预览' if decision.get('mode')=='preview' else '自动判断',
            comment=row['text'] if kind=='reply_attempts' else decision.get('text',''),
            reason=BLOCKS.get(decision.get('reason',''),decision.get('reason','')),rpid=row.get('rpid',''))
        if kind == 'replies' and result['updated_time']:
            result['time'] = result['updated_time']
        result['summary']=(result['comment'] or result['reason'])[:100]
        return result
    if kind == "decisions":
        result.update(title="自主浏览意愿", outcome="去浏览" if row["should_browse"] else "暂时不刷",
                      reason=row["reason"], summary=row["reason"],
                      next_time=local_time(row["next_check_at"], timezone))
        result['comment'] = '希望查看：'+', '.join(json.loads(row.get('selected_bvids','[]')))
        return result
    try:
        detail = json.loads(row["detail"])
    except (ValueError, TypeError):
        detail = {}
    if not isinstance(detail, dict):
        detail = {}
    outcome = row["status"] if kind == "attempts" else row["outcome"]
    bvid = row["bvid"]
    result.update(bvid=bvid, url="https://www.bilibili.com/video/" + bvid,
                  title=str(detail.get("title", bvid)), outcome=OUTCOMES.get(outcome, outcome),
                  mode="实际发送" if kind == "attempts" else ("手动/自动预览" if row["mode"] == "preview" else "自主浏览"),
                  interest=detail.get("interest", ""), source=str(detail.get("source", "")),
                  comment=row["text"] if kind == "attempts" else str(detail.get("comment", "")),
                  reason=BLOCKS.get(str(detail.get("reason", row["detail"])), str(detail.get("reason", row["detail"]))),
                  rpid=row["rpid"] if kind == "attempts" else "")
    result["summary"] = (result["comment"] or result["reason"])[:100]
    if outcome in {"deferred", "quota_wait"} and detail.get("selection_time") and not detail.get("selected"):
        result["outcome"] = "本次未选中，仍保留候选"
        result["reason"] += "；最近选择：" + str(detail.get("selection_reason", ""))
    if detail.get('read_warning'):
        result['reason'] += f"；读取提示：{detail['read_warning']}，已使用字幕"
    return result


def retry_status(row: Dict[str, Any], timezone: str) -> str:
    if row.get('outcome') in {'quota_wait','deferred'} and row.get('retry_at',0)>0:
        return f"{local_time(row['retry_at'],timezone)} 起可参与候选选择；选中后重新读取与判断，不直接发送旧草稿"
    if row.get('outcome') not in RETRY_OUTCOMES:
        return ''
    failures = row.get('failures',0)
    if failures >= MAX_FAILURES:
        return f"已失败 {failures} 次，达到重查上限"
    if row.get('retry_at',0) > 0:
        return f"已失败 {failures} 次，{local_time(row['retry_at'],timezone)} 起可重查；最多再查 {MAX_FAILURES-failures} 次"
    return '不自动重查，请检查错误原因'
