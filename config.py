"""通过 SDK 暴露 WebUI 配置；不在配置中存放账号密码或 Cookie。"""

from typing import List
from zoneinfo import ZoneInfo

from maibot_sdk import Field, PluginConfigBase
from pydantic import field_validator, model_validator


class PluginSection(PluginConfigBase):
    """设置插件开关，以及哪些账号可以使用登录、预览和启停命令。"""

    __ui_label__ = "基本设置"
    __ui_order__ = 0
    enabled: bool = Field(
        default=True,
        description="启用插件，登录后默认自动进入自主浏览；手动暂停仍需 start 恢复",
        json_schema_extra={
            "label": "启用插件",
            "hint": "关闭后停止后台浏览。默认登录后自动进入自主模式；手动 pause 或发生异常后，需要 start 恢复。",
        },
    )
    config_version: str = Field(
        default="0.2.0",
        description="插件自动维护的配置格式版本",
        json_schema_extra={
            "label": "配置格式版本（无需修改）",
            "hint": "由插件自动维护，用于识别配置格式，保持原值即可。",
            "disabled": True,
        },
    )
    admin_users: List[str] = Field(
        default_factory=list,
        description="可直接填写 QQ 号（自动补全 qq:），其他平台填写 平台:用户号；留空拒绝普通聊天命令",
        json_schema_extra={
            "label": "管理员账号",
            "hint": "QQ 管理员直接填 QQ 号，保存时自动补全 qq:；其他平台填写 平台:用户号。每项一个账号，留空时普通聊天中无法使用管理命令。",
            "placeholder": "例如：12345678 或 qq:12345678",
        },
    )

    @field_validator("admin_users")
    @classmethod
    def normalize_admins(cls, values: List[str]) -> List[str]:
        normalized: List[str] = []
        for value in values:
            value = value.strip()
            if value.isascii() and value.isdecimal():
                value = f"qq:{value}"
            if ":" not in value or any(not part.strip() for part in value.split(":", 1)):
                raise ValueError("管理员请填写 QQ 号或 平台:用户号，例如 123456 或 qq:123456")
            platform, user_id = value.split(":", 1)
            normalized.append(f"{platform.strip().lower()}:{user_id.strip()}")
        return list(dict.fromkeys(normalized))


class BrowseSection(PluginConfigBase):
    """决定多久浏览一次、每天看多少视频，以及感兴趣或需要跳过的内容。"""

    __ui_label__ = "浏览与兴趣"
    __ui_order__ = 1
    backlog_max_items: int = Field(default=24, ge=1, le=100, description="未发送等待视频队列上限，超限淘汰最旧项")
    backlog_max_age_hours: int = Field(default=48, ge=1, le=168, description="未发送等待视频有效期，小时")
    read_retry_base_seconds: int = Field(
        default=60, ge=10, le=600, description="只读网络失败退避起点，最多四次后每小时探测"
    )
    max_per_up: int = Field(default=2, ge=1, le=10, description="每轮同一 UP 主最多分析的视频数")
    max_per_topic: int = Field(
        default=2, ge=1, le=10, description="每轮同一已知分区题材最多分析的视频数，未知题材不强行分类"
    )
    auto_start: bool = Field(
        default=True,
        description="首次登录或重新启用插件后自动进入运行状态；不解除手动暂停或异常暂停",
        json_schema_extra={
            "label": "登录后自动开始",
            "hint": "默认开启，扫码成功后无需发送 start。运行状态在重启后保留；手动 pause 或异常暂停不会被自动解除。关闭后首次使用需发送 start。",
        },
    )
    autonomous_enabled: bool = Field(
        default=True,
        description="先由模型决定这次是否浏览，以及下次何时再考虑",
        json_schema_extra={
            "label": "自主决定刷不刷 B 站",
            "hint": "开启后，麦麦结合人设、兴趣、时间和最近浏览反馈决定刷不刷，也可以选择晚些再看。关闭后按浏览间隔定时刷。评论仍单独判断。",
        },
    )
    explore_more: bool = Field(
        default=False,
        description="自主判断时更愿意尝试新的感兴趣视频，仍可选择休息",
        json_schema_extra={
            "label": "更积极地探索新视频",
            "hint": "开启后，有感兴趣的新推荐时更愿意先看一两条，不因最近没有想发的评论就一直休息。仍由麦麦自主决定，不保证浏览或评论比例。关闭时维持随性节奏。",
        },
    )
    interval_seconds: int = Field(
        default=600,
        ge=300,
        description="最短浏览意愿检查间隔，秒",
        json_schema_extra={
            "label": "最短浏览意愿检查间隔（秒）",
            "hint": "自主模式下至少间隔多久再考虑刷 B 站，默认 600 秒＝10 分钟，最少 300 秒＝5 分钟。麦麦可在最短和最长间隔之间决定等待多久；关闭自主判断后按此固定间隔运行。",
        },
    )
    max_check_minutes: int = Field(
        default=20,
        ge=5,
        le=360,
        description="自主模式下两次浏览意愿检查最多相隔多少分钟",
        json_schema_extra={
            "label": "最长浏览意愿检查间隔（分钟）",
            "hint": "默认 20 分钟，可填 5～360，不能小于最短检查间隔。模型不能自行休息超过这个上限；调小后，当前尚未结束的等待也会缩短。非活动时段和每日视频额度仍然生效。",
        },
    )
    videos_per_round: int = Field(
        default=3,
        ge=1,
        le=10,
        description="每轮最多分析的视频数",
        json_schema_extra={
            "label": "每轮最多分析视频数",
            "hint": "一轮最多读取并分析几个视频，默认 3 个，可填 1～10。每轮最多发表 1 条评论。",
        },
    )
    videos_per_day: int = Field(
        default=180,
        ge=1,
        le=200,
        description="每天分析视频的上限，预览也计入",
        json_schema_extra={
            "label": "每日视频分析上限",
            "hint": "达到上限后当天停止分析，手动预览也计入。默认每天 180 个，按所选时区的零点重新统计。",
        },
    )
    max_duration_seconds: int = Field(
        default=1200,
        ge=30,
        le=7200,
        description="跳过超过此时长的视频",
        json_schema_extra={
            "label": "视频最长时长（秒）",
            "hint": "超过此时长的视频直接跳过。默认 1200 秒＝20 分钟，可填 30～7200 秒。",
        },
    )
    interests: str = Field(
        default="",
        max_length=2000,
        description="兴趣偏好，留空时由麦麦人设和账号推荐决定",
        json_schema_extra={
            "label": "感兴趣的话题",
            "hint": "告诉麦麦你希望它关注哪些内容，例如科普、游戏或音乐。留空时结合麦麦人设和账号推荐自行判断。",
            "placeholder": "例如：喜欢科普、独立游戏和音乐制作",
        },
    )
    blocked_keywords: List[str] = Field(
        default_factory=list,
        description="标题、简介或视频内容命中后跳过",
        json_schema_extra={
            "label": "屏蔽关键词",
            "hint": "标题、简介或读取到的视频内容包含任意一个关键词，就跳过该视频。每项填一个词，留空不限制。",
            "placeholder": "例如：剧透",
        },
    )
    blocked_up_uids: List[str] = Field(
        default_factory=list,
        description="跳过这些 UP 主的 UID",
        json_schema_extra={
            "label": "屏蔽的 UP 主 UID",
            "hint": "不浏览这些 UP 主的视频。填写 B 站个人空间中的数字 UID，不是昵称；每项一个，留空不屏蔽。",
            "placeholder": "填写 UP 主的数字 UID",
        },
    )
    active_start_hour: int = Field(
        default=9,
        ge=0,
        le=23,
        description="开始活动的小时",
        json_schema_extra={
            "label": "开始浏览时间（小时）",
            "hint": "使用 24 小时制，填 0～23。例如填 9 表示从每天 09:00 开始允许自动浏览。",
        },
    )
    active_end_hour: int = Field(
        default=23,
        ge=0,
        le=23,
        description="结束活动的小时；与开始相同表示全天，支持跨午夜",
        json_schema_extra={
            "label": "结束浏览时间（小时）",
            "hint": "例如填 23 表示从 23:00 起停止。结束时间早于开始时间表示跨午夜；两者相同表示全天运行。",
        },
    )
    timezone: str = Field(
        default="Asia/Shanghai",
        description="活动时间与每日限额使用的时区",
        json_schema_extra={
            "label": "活动与统计时区",
            "hint": "活动时段和每日限额都按这个时区计算。默认 Asia/Shanghai，表示北京时间，通常保持默认即可。",
            "placeholder": "Asia/Shanghai",
        },
    )

    @field_validator("timezone")
    @classmethod
    def check_timezone(cls, value: str) -> str:
        ZoneInfo(value)
        return value

    @model_validator(mode="after")
    def validate_check_interval(self) -> "BrowseSection":
        if self.max_check_minutes * 60 < self.interval_seconds:
            raise ValueError("最长浏览意愿检查间隔不能小于最短检查间隔，请同时调整这两项")
        return self


class CommentSection(PluginConfigBase):
    """决定是否实际发表评论，以及发送频率、评论长度和重复检查的规则。"""

    __ui_label__ = "评论"
    __ui_order__ = 2
    publish_enabled: bool = Field(
        default=False,
        description="开启后 start 会自动发布；关闭时只预览并记录判断",
        json_schema_extra={
            "label": "自动发表评论",
            "hint": "关闭时只生成预览记录；开启后，运行中的插件可把评论发到 B 站。使用 /bili_auto start 启动；/bili_auto preview 始终只预览。",
        },
    )
    per_day: int = Field(
        default=12,
        ge=1,
        le=200,
        description="每天最多发送尝试数，失败或结果未知也计入",
        json_schema_extra={
            "label": "每日评论发送上限",
            "hint": "默认每天最多 12 次发送尝试，可填 1～200。失败或结果未知也计入；达到上限后停止发送一级评论，仍可浏览视频。与跟进回复的每日额度分开统计，按所选时区零点重新统计。",
        },
    )
    min_interval_seconds: int = Field(
        default=2700,
        ge=300,
        description="两次发送尝试至少间隔多少秒",
        json_schema_extra={
            "label": "评论最短间隔（秒）",
            "hint": "发送视频下一级评论前，距离最近一次评论或跟进回复至少等待多久。默认 2700 秒＝45 分钟；跟进别人回复使用下面独立的最短间隔。",
        },
    )
    max_chars: int = Field(
        default=160,
        ge=20,
        le=500,
        description="评论最大字数",
        json_schema_extra={
            "label": "单条评论最大字数",
            "hint": "默认最多 160 字，可填 20～500。模型生成的评论超过上限时会跳过，不会直接截断后发送。",
        },
    )
    min_interest: float = Field(
        default=0.75,
        ge=0,
        le=1,
        description="模型给出的兴趣分低于该值时不评论；不是校准概率",
        json_schema_extra={
            "label": "发表评论的最低兴趣分",
            "hint": "可填 0～1，默认 0.75。越高越挑剔，低于该分数就不评论；这是模型的兴趣评分，不是发表评论的概率。",
            "step": 0.01,
        },
    )
    similarity_threshold: float = Field(
        default=0.88,
        ge=0.5,
        le=1,
        description="与近期评论过于相似时跳过",
        json_schema_extra={
            "label": "重复评论相似度阈值",
            "hint": "与近期评论的相似度达到此值就跳过。默认 0.88，可填 0.5～1；越低查重越严格。完全重复的文字始终拦截。",
            "step": 0.01,
        },
    )


class ModelSection(PluginConfigBase):
    """选择麦麦已有的模型任务，设置评论语气、等待时间和读取内容的长度。"""

    __ui_label__ = "模型"
    __ui_order__ = 3
    decision_budget_seconds: int = Field(
        default=240, ge=20, le=900, description="每条评论或回复生成、复核及一次重生成的总时限"
    )
    review_timeout_seconds: int = Field(default=45, ge=1, le=120, description="单次正文依据复核时限，失败不发送")
    evidence_retention_days: int = Field(default=7, ge=1, le=30, description="本地引用片段与时间位置保留天数")
    task: str = Field(
        default="replyer",
        description="Host 中用于判断兴趣和生成评论的模型任务",
        json_schema_extra={
            "label": "评论使用的模型任务",
            "hint": "填写麦麦模型配置中已有的任务名。默认 replyer，使用麦麦的回复模型；这里不是填写模型厂商或具体型号。",
            "placeholder": "replyer",
        },
    )
    candidates: List[str] = Field(
        default_factory=list,
        description="按顺序尝试的具体模型名；留空时由 Host 任务选择模型",
        json_schema_extra={
            "label": "候选模型（按顺序尝试）",
            "hint": "填写麦麦已配置的具体模型名，每项一个。单个模型超时后换下一个，每轮每个模型最多尝试一次；留空时由模型任务选择。",
        },
    )

    @field_validator("candidates")
    @classmethod
    def normalize_candidates(cls, values: List[str]) -> List[str]:
        if any(not value.strip() for value in values):
            raise ValueError("候选模型名不能为空")
        return list(dict.fromkeys(value.strip() for value in values))

    timeout_seconds: int = Field(
        default=90,
        ge=10,
        le=180,
        description="每个候选模型的一次调用超时",
        json_schema_extra={
            "label": "模型回答等待时间（秒）",
            "hint": "每个候选模型最多等待多久，默认 90 秒，可填 10～180 秒。超时换下一个；全部超时跳过本次，自动运行保持开启。",
        },
    )
    extra_style: str = Field(
        default="",
        max_length=2000,
        description="在麦麦人设之外补充的 B 站评论风格",
        json_schema_extra={
            "label": "额外评论风格",
            "hint": "在麦麦原有人设基础上补充评论语气，例如简短、自然、少用表情。留空时沿用麦麦人设。",
            "placeholder": "例如：评论简短自然，有具体感受再发言",
        },
    )
    content_max_chars: int = Field(
        default=12000,
        ge=500,
        le=24000,
        description="提供给模型的字幕或摘要字符上限",
        json_schema_extra={
            "label": "最多读取的字幕或摘要字数",
            "hint": "默认最多向模型提供12000字。长字幕兼顾相关片段、开头、中段和结尾，保留时间位置；不读取画面或音频。",
        },
    )


class ReplySection(PluginConfigBase):
    """检查别人对麦麦评论的回复，并决定是否继续对话。"""

    __ui_label__ = "跟进别人回复"
    __ui_order__ = 4
    enabled: bool = Field(
        default=True,
        description="检查并跟进别人对本插件已发送评论的回复",
        json_schema_extra={
            "label": "启用跟进回复",
            "hint": "默认开启。只处理别人直接回复麦麦已发送评论的消息，不处理私信、单独 @ 或无关楼层。实际发送仍受“自动发表评论”总开关控制。",
        },
    )
    interval_seconds: int = Field(
        default=600,
        ge=300,
        le=3600,
        description="收到回复的检查间隔，秒",
        json_schema_extra={
            "label": "收到回复检查间隔（秒）",
            "hint": "默认 600 秒＝10 分钟，不受麦麦是否想刷视频影响。暂停、退出账号或非活动时段不自动检查。",
        },
    )
    per_day: int = Field(
        default=8,
        ge=1,
        le=200,
        description="每天最多尝试发送的跟进回复数",
        json_schema_extra={
            "label": "每日跟进回复发送上限",
            "hint": "默认每天最多 8 次发送尝试，可填 1～200。失败或结果未知也计入；与视频下一级评论的每日额度分开统计。达到上限后保留待处理回复，在有效期内下次再考虑；按所选时区零点重新统计。实际发送还受每日分析额度、发送间隔和单条讨论次数限制。",
        },
    )
    analyses_per_day: int = Field(
        default=20,
        ge=1,
        le=100,
        description="每天最多分析多少条收到的回复，预览也计入",
        json_schema_extra={
            "label": "每日收到回复分析上限",
            "hint": "默认每天最多分析 20 条，手动预览也计入；与视频分析额度分开统计，防止收到大量回复时持续调用模型。",
        },
    )
    per_thread: int = Field(
        default=3,
        ge=1,
        le=10,
        description="同一条根评论讨论中最多跟进的次数",
        json_schema_extra={
            "label": "同一条讨论最多跟进次数",
            "hint": "默认在一条根评论下最多跟进 3 次，按账号长期累计，失败也计入，防止与别人反复对话。",
        },
    )
    min_interval_seconds: int = Field(
        default=900,
        ge=300,
        le=86400,
        description="发送跟进回复前与最近一次评论或回复至少间隔多少秒",
        json_schema_extra={
            "label": "跟进回复最短发送间隔（秒）",
            "hint": "默认 900 秒＝15 分钟，从最近一次评论或跟进回复发送尝试算起；等待期间保留收到的消息，下次检查时再考虑。",
        },
    )
    max_age_hours: int = Field(
        default=72,
        ge=1,
        le=168,
        description="仅处理最近多少小时内收到的回复",
        json_schema_extra={
            "label": "只跟进最近多少小时的回复",
            "hint": "默认 72 小时＝3 天，可填 1～168，避免上线后突然回复很久以前的讨论。每次最多读取三页通知。",
        },
    )


class ShareSection(PluginConfigBase):
    """评论成功后，将感兴趣的视频分享至已有 QQ 群聊。"""

    __ui_label__ = "评论后分享视频"
    __ui_order__ = 5
    polish_enabled: bool = Field(default=True, description="每条分享最多调用一次模型润色，所有群复用文案")
    polish_model: str = Field(default="", max_length=200, description="润色使用 Host 已配置的模型名，留空沿用评论任务")
    polish_timeout_seconds: int = Field(default=30, ge=1, le=120, description="分享润色独立时限，包含正文依据复核")
    total_timeout_seconds: int = Field(default=120, ge=10, le=360, description="分享草稿、润色及复核的总时限")
    qq_group_ids: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="评论成功后分享视频的 QQ 群号；默认留空，不分享",
        json_schema_extra={
            "label": "分享视频的 QQ 群号",
            "hint": "默认留空，不分享。每项填写一个麦麦已加入的 QQ 群号，最多 10 个。仅实际提交成功的视频一级评论触发，预览、跳过和跟进回复不触发。结合人设和视频内容生成一句随手分享的感想，再附上链接；同一视频对同一群只尝试分享一次，失败或结果未知不会自动重发。",
            "placeholder": "例如：123456789",
        },
    )

    @field_validator("qq_group_ids")
    @classmethod
    def validate_groups(cls, groups: List[str]) -> List[str]:
        result = []
        for raw in groups:
            group = raw.strip()
            if not (group.isascii() and group.isdecimal() and 5 <= len(group) <= 12 and not group.startswith("0")):
                raise ValueError("QQ 群号请填写 5～12 位数字，不含 qq: 前缀，例如 123456789；不分享时删除该项留空")
            if group not in result:
                result.append(group)
        return result


class AutopilotConfig(PluginConfigBase):
    plugin: PluginSection = Field(default_factory=PluginSection)
    browse: BrowseSection = Field(default_factory=BrowseSection)
    comment: CommentSection = Field(default_factory=CommentSection)
    model: ModelSection = Field(default_factory=ModelSection)
    replies: ReplySection = Field(default_factory=ReplySection)
    share: ShareSection = Field(default_factory=ShareSection)
