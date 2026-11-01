import asyncio
import json
import random
import re
import time
import uuid
from pathlib import Path

import aiohttp

from astrbot.api.all import (
    AstrBotConfig,
    AstrMessageEvent,
    Context,
    MessageChain,
    Plain,
    Record,
    Star,
    logger,
)
from astrbot.api.event import filter
from astrbot.api.provider import LLMResponse
from astrbot.api.star import StarTools
from astrbot.core.star.filter.command import GreedyStr

# 音色 JSON 里的字段
_VOICE_FIELDS = ("id", "name", "category", "vip_use", "platform")


def _atomic_write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class KKttsClient:
    """kktts API 客户端：音色列表 + 语音合成"""

    def __init__(self, api_base: str, timeout: int, cache_dir: Path):
        self.api_base = api_base.rstrip("?&")
        self.timeout = timeout
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._headers = {"User-Agent": "Mozilla/5.0 astrbot-plugin-kktts/1.0"}

    async def fetch_voices(self) -> list[dict]:
        """拉取全部音色，返回 [{id,name,category,vip_use,platform}]，失败抛异常"""
        params = {"action": "list", "detail": 1}
        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.get(
                self.api_base, params=params, timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
        if data.get("meta", {}).get("status") != 200:
            raise RuntimeError(f"获取音色列表失败: {data.get('meta')}")
        seen: dict[str, dict] = {}
        for cat in data.get("data", []):
            cat_name = cat.get("name", "")
            for v in cat.get("list", []):
                vid = v.get("id")
                if not vid or vid in seen:
                    continue
                seen[vid] = {
                    "id": vid,
                    "name": v.get("name", ""),
                    "category": cat_name,
                    "vip_use": v.get("vip_use", ""),
                    "platform": v.get("platform", ""),
                }
        return list(seen.values())

    async def synthesize(self, text: str, voice_id: str) -> tuple[Path | None, str | None]:
        """合成语音并下载到本地。返回 (本地文件路径, None) 或 (None, 错误信息)"""
        params = {"action": "voice", "content": text, "voice_id": voice_id}
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        try:
            async with aiohttp.ClientSession(headers=self._headers) as session:
                async with session.get(self.api_base, params=params, timeout=timeout) as resp:
                    if resp.status != 200:
                        return None, f"HTTP {resp.status}"
                    try:
                        data = await resp.json(content_type=None)
                    except Exception:
                        return None, "接口返回非 JSON"
                code = data.get("code")
                if code != 0:
                    return None, f"code={code} {data.get('msg') or ''}".strip()
                audio_url = (data.get("data") or {}).get("url", "")
                if not audio_url:
                    return None, "接口未返回音频地址"
                async with session.get(audio_url, timeout=timeout) as aresp:
                    if aresp.status != 200:
                        return None, f"下载音频失败 HTTP {aresp.status}"
                    content = await aresp.read()
        except asyncio.TimeoutError:
            return None, "请求超时"
        except Exception as e:
            return None, str(e)

        if len(content) < 1024:
            return None, f"音频内容异常（{len(content)} 字节）"
        out = self.cache_dir / f"{uuid.uuid4().hex}.mp3"
        out.write_bytes(content)
        return out, None

    def cleanup_cache(self, keep_seconds: int = 3600) -> None:
        """清理过期的音频缓存文件"""
        now = time.time()
        try:
            for f in self.cache_dir.glob("*.mp3"):
                try:
                    if now - f.stat().st_mtime > keep_seconds:
                        f.unlink(missing_ok=True)
                except OSError:
                    pass
        except OSError:
            pass


class PluginState:
    """会话级设置持久化：音色、自动语音概率、开关"""

    def __init__(self, path: Path):
        self.path = path
        self._data: dict = {}
        try:
            if path.exists():
                self._data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"[kktts] 读取会话配置失败: {e}")
            self._data = {}

    def get(self, umo: str) -> dict:
        return self._data.get(umo) or {}

    def set(self, umo: str, **kwargs) -> dict:
        cfg = self._data.setdefault(umo, {})
        cfg.update(kwargs)
        try:
            _atomic_write_json(self.path, self._data)
        except Exception as e:
            logger.error(f"[kktts] 保存会话配置失败: {e}")
        return cfg

    def pop(self, umo: str, key: str):
        cfg = self._data.get(umo)
        if cfg and key in cfg:
            cfg.pop(key, None)
            try:
                _atomic_write_json(self.path, self._data)
            except Exception as e:
                logger.error(f"[kktts] 保存会话配置失败: {e}")


class KKttsPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.data_dir = StarTools.get_data_dir("ldmpl_kktts")
        self.client = KKttsClient(
            api_base=str(config.get("api_base", "https://api-v2.yuafeng.cn/API/kktts.php")),
            timeout=int(config.get("tts_timeout", 60)),
            cache_dir=self.data_dir / "audio_cache",
        )
        self.state = PluginState(self.data_dir / "sessions.json")
        self.voices_file = self.data_dir / "voices.json"
        self._voices: list[dict] = []
        self._voices_by_id: dict[str, dict] = {}
        self._voices_fetched_at: float = 0.0
        # 会话 -> 上次自动转语音时间戳（仅内存，重启清零无妨）
        self._last_auto_ts: dict[str, float] = {}
        # 后台预加载音色列表，不阻塞插件加载
        self._load_task = asyncio.create_task(self._ensure_voices(force=False))

    # ---------------- 音色管理 ----------------

    async def _ensure_voices(self, force: bool = False) -> bool:
        """确保音色列表可用。返回是否成功"""
        cache_hours = int(self.config.get("cache_hours", 24))
        if not force and self._voices and time.time() - self._voices_fetched_at < cache_hours * 3600:
            return True
        # 内存没有时先读文件缓存
        if not self._voices and self.voices_file.exists():
            try:
                cached = json.loads(self.voices_file.read_text(encoding="utf-8"))
                self._voices = cached.get("voices", [])
                self._voices_fetched_at = cached.get("fetched_at", 0.0)
                self._reindex()
            except Exception:
                pass
        if not force and self._voices and time.time() - self._voices_fetched_at < cache_hours * 3600:
            return True
        try:
            voices = await self.client.fetch_voices()
            self._voices = voices
            self._voices_fetched_at = time.time()
            self._reindex()
            _atomic_write_json(
                self.voices_file, {"fetched_at": self._voices_fetched_at, "voices": voices}
            )
            logger.info(f"[kktts] 音色列表已更新，共 {len(voices)} 个音色")
            return True
        except Exception as e:
            logger.warning(f"[kktts] 拉取音色列表失败: {e}")
            return bool(self._voices)

    def _reindex(self) -> None:
        self._voices_by_id = {v["id"]: v for v in self._voices}

    def _find_voice(self, keyword: str) -> dict | None:
        """按 ID 精确 / 名称精确 / 名称模糊 查找音色"""
        kw = keyword.strip()
        if not kw:
            return None
        if kw in self._voices_by_id:
            return self._voices_by_id[kw]
        low = kw.lower()
        for v in self._voices:
            if v.get("name", "").lower() == low:
                return v
        for v in self._voices:
            if low in v.get("name", "").lower() or low in v.get("id", "").lower():
                return v
        return None

    def _default_voice(self) -> dict:
        vid = str(self.config.get("default_voice_id", "") or "")
        v = self._voices_by_id.get(vid)
        if v:
            return v
        return {
            "id": vid or "cc073894e597a60a8a784ef4b4e9b473",
            "name": str(self.config.get("default_voice_name", "") or "默认音色"),
            "category": "",
            "vip_use": "",
            "platform": "",
        }

    def _session_voice(self, event: AstrMessageEvent) -> dict:
        s = self.state.get(event.unified_msg_origin)
        vid = s.get("voice_id")
        if vid and vid in self._voices_by_id:
            return self._voices_by_id[vid]
        if vid:  # 会话设置的音色不在列表里，也允许直接用
            return {"id": vid, "name": s.get("voice_name", vid), "category": "", "vip_use": "", "platform": ""}
        return self._default_voice()

    # ---------------- 内部工具 ----------------

    @staticmethod
    def _clean_for_tts(text: str) -> str:
        """把 LLM 输出清理成适合朗读的文本"""
        t = re.sub(r"```.*?```", "，此处省略一段代码，", text, flags=re.DOTALL)
        t = re.sub(r"https?://\S+", "，链接已省略，", t)
        t = re.sub(r"`([^`]*)`", r"\1", t)
        t = re.sub(r"[*_#>~|\[\]()（）]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t or text.strip()

    def _session_probability(self, event: AstrMessageEvent) -> int:
        s = self.state.get(event.unified_msg_origin)
        prob = s.get("probability")
        if prob is None:
            prob = int(self.config.get("auto_probability", 10))
        return int(prob)

    def _auto_enabled(self, event: AstrMessageEvent) -> bool:
        if not bool(self.config.get("auto_tts", True)):
            return False
        s = self.state.get(event.unified_msg_origin)
        if s.get("enabled") is False:
            return False
        return True

    async def _synthesize_or_none(self, event: AstrMessageEvent, text: str):
        """用会话音色合成，返回 (path, err)"""
        voice = self._session_voice(event)
        path, err = await self.client.synthesize(text, voice["id"])
        if path:
            self.client.cleanup_cache()
        return path, err, voice

    @staticmethod
    def _voice_desc(v: dict) -> str:
        name = v.get("name") or v.get("id", "?")
        extras = []
        if v.get("category"):
            extras.append(f"分类:{v['category']}")
        if v.get("vip_use") not in ("", None):
            extras.append(f"vip_use={v['vip_use']}")
        return f"{name}（{'，'.join(extras)}）" if extras else name

    # ---------------- 指令组：tts ----------------

    @filter.command_group("tts", alias={"语音合成", "说"})
    def tts_group(self) -> None:
        """KK语音合成"""

    def _help_text(self) -> str:
        return (
            "📖 KK语音合成 指令帮助\n"
            "· tts <文本>：把文本转成语音发送\n"
            "· tts 生成 <提示词>：让 LLM 写文案再转语音\n"
            "· tts 角色 [名称/ID/随机/重置]：查看或切换本会话音色\n"
            "· tts 列表 [关键词] [页码]：搜索音色列表\n"
            "· tts 自动 [0-100/开/关/重置]：LLM 回复自动转语音设置\n"
            "· tts 状态：查看当前插件状态\n"
            "· tts 刷新：强制刷新音色列表缓存\n"
            "· tts help：显示本帮助\n"
            "提示：子指令也可连写，如 tts生成 / tts角色 / tts帮助，效果相同"
        )

    @tts_group.command("help", alias={"帮助", "menu"})
    async def tts_group_help(self, event: AstrMessageEvent):
        """显示全部指令帮助"""
        yield event.plain_result(self._help_text())

    @tts_group.command("say", alias={"读"})
    async def tts_group_say(self, event: AstrMessageEvent, text: GreedyStr = ""):
        """把文本转成语音发送：tts say <文本>"""
        async for r in self.tts_say(event, text):
            yield r

    @tts_group.command("*")
    async def tts_group_fallback(self, event: AstrMessageEvent, text: GreedyStr = ""):
        """把文本转成语音发送：tts <文本>"""
        async for r in self.tts_say(event, text):
            yield r

    @tts_group.command("角色", alias={"voice", "音色"})
    async def tts_group_voice(self, event: AstrMessageEvent, keyword: GreedyStr = ""):
        """查看或设置本会话音色：tts 角色 [名称/ID/随机/重置]"""
        async for r in self.tts_voice(event, keyword):
            yield r

    @tts_group.command("列表", alias={"list", "音色列表"})
    async def tts_group_list(self, event: AstrMessageEvent, args: GreedyStr = ""):
        """搜索音色列表：tts 列表 [关键词] [页码]"""
        async for r in self.tts_list(event, args):
            yield r

    @tts_group.command("生成", alias={"gen"})
    async def tts_group_llm(self, event: AstrMessageEvent, prompt: GreedyStr = ""):
        """LLM 生成文案后合成语音：tts 生成 <提示词>"""
        async for r in self.tts_llm(event, prompt):
            yield r

    @tts_group.command("自动", alias={"auto"})
    async def tts_group_auto(self, event: AstrMessageEvent, args: GreedyStr = ""):
        """设置 LLM 回复自动转语音：tts 自动 [0-100/开/关/重置]"""
        async for r in self.tts_auto(event, args):
            yield r

    @tts_group.command("状态", alias={"status"})
    async def tts_group_status(self, event: AstrMessageEvent):
        """查看插件当前状态"""
        async for r in self.tts_status(event):
            yield r

    @tts_group.command("刷新", alias={"refresh"})
    async def tts_group_refresh(self, event: AstrMessageEvent):
        """强制刷新音色列表缓存"""
        async for r in self.tts_refresh(event):
            yield r

    @filter.command("tts帮助", alias={"语音帮助", "tts指令"})
    async def tts_help(self, event: AstrMessageEvent):
        """显示全部指令的用法帮助"""
        yield event.plain_result(self._help_text())

    # ---------------- 指令：语音合成 ----------------

    async def tts_say(self, event: AstrMessageEvent, text: GreedyStr = ""):
        """用当前会话音色把文本合成语音发送，失败回退文字（指令组 tts 的实现体）"""
        content = str(text).strip()
        if not content:
            yield event.plain_result(
                "用法：tts <要合成的文本>\n查看全部指令：tts help"
            )
            return
        await self._ensure_voices()
        path, err, voice = await self._synthesize_or_none(event, content)
        if path:
            record = Record.fromFileSystem(str(path))
            record.text = content
            yield event.chain_result([record])
        else:
            # 合成失败，回退文字
            yield event.plain_result(f"[kktts] 语音合成失败：{err}\n文字回退：\n{content}")

    # ---------------- 指令：指定音色角色 ----------------

    @filter.command("tts角色", alias={"语音角色", "tts音色"})
    async def tts_voice(self, event: AstrMessageEvent, keyword: GreedyStr = ""):
        """查看或设置当前会话的音色角色（名称或 ID），「随机」随机选择，「重置」恢复默认"""
        kw = str(keyword).strip()
        umo = event.unified_msg_origin
        if not kw:
            cur = self._session_voice(event)
            yield event.plain_result(
                f"当前会话音色：{self._voice_desc(cur)}\n"
                f"音色ID：{cur['id']}\n"
                "设置音色：tts角色 <音色名称或ID>\n"
                "随机换一个：tts角色 随机\n"
                "恢复默认：tts角色 重置\n"
                "查找音色：tts列表 <关键词>"
            )
            return
        await self._ensure_voices()
        if kw in ("重置", "默认", "reset"):
            self.state.pop(umo, "voice_id")
            self.state.pop(umo, "voice_name")
            d = self._default_voice()
            yield event.plain_result(f"已恢复默认音色：{self._voice_desc(d)}")
            return
        if kw in ("随机", "random"):
            if not self._voices:
                yield event.plain_result("音色列表为空，稍后再试或使用 tts刷新")
                return
            v = random.choice(self._voices)
            self.state.set(umo, voice_id=v["id"], voice_name=v.get("name", ""))
            yield event.plain_result(f"已随机切换音色：{self._voice_desc(v)}\n音色ID：{v['id']}")
            return
        v = self._find_voice(kw)
        if not v:
            yield event.plain_result(
                f"未找到音色「{kw}」，用 tts列表 <关键词> 搜索，或 tts角色 随机 抽一个"
            )
            return
        self.state.set(umo, voice_id=v["id"], voice_name=v.get("name", ""))
        yield event.plain_result(
            f"当前会话音色已切换：{self._voice_desc(v)}\n音色ID：{v['id']}\n试一试：tts 你好呀"
        )

    # ---------------- 指令：音色列表 ----------------

    @filter.command("tts列表", alias={"音色列表", "tts音色列表"})
    async def tts_list(self, event: AstrMessageEvent, args: GreedyStr = ""):
        """搜索音色列表：tts列表 [关键词] [页码]"""
        parts = str(args).split()
        keyword = parts[0] if parts and not parts[0].isdigit() else ""
        page = 1
        for p in parts:
            if p.isdigit():
                page = max(1, int(p))
                break
        ok = await self._ensure_voices()
        if not ok or not self._voices:
            yield event.plain_result("音色列表获取失败，请稍后用 tts刷新 重试")
            return
        pool = self._voices
        if keyword:
            low = keyword.lower()
            pool = [
                v
                for v in self._voices
                if low in v.get("name", "").lower()
                or low in v.get("id", "").lower()
                or low in v.get("category", "").lower()
            ]
        total = len(pool)
        per = 10
        pages = max(1, (total + per - 1) // per)
        page = min(page, pages)
        chunk = pool[(page - 1) * per : page * per]
        lines = [f"音色共 {total} 个，第 {page}/{pages} 页："]
        for i, v in enumerate(chunk, start=(page - 1) * per + 1):
            if total == 1:
                # 唯一命中：给出完整 ID，方便精确设置或分享
                lines.append(f"{i}. {self._voice_desc(v)}")
                lines.append(f"   完整ID：{v['id']}")
            else:
                lines.append(f"{i}. {self._voice_desc(v)}  id={v['id'][:8]}…")
        lines.append(
            "设置：tts角色 <名称或ID>（一般直接用名称即可；仅搜到 1 个时会显示完整 ID）"
        )
        yield event.plain_result("\n".join(lines))

    # ---------------- 指令：LLM 生成后合成 ----------------

    @filter.command("tts生成", alias={"tts说", "语音生成"})
    async def tts_llm(self, event: AstrMessageEvent, prompt: GreedyStr = ""):
        """调用 LLM 根据提示词生成文案，再合成语音发送，失败回退文字"""
        p = str(prompt).strip()
        if not p:
            yield event.plain_result("用法：tts生成 <想让 LLM 写什么，如：用傲娇的语气说晚安>")
            return
        provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        if provider is None:
            yield event.plain_result("未配置可用的 LLM 服务商，无法使用 tts生成")
            return
        await self._ensure_voices()
        yield event.plain_result("正在生成文案并合成语音…")
        try:
            resp = await provider.text_chat(
                prompt=p,
                system_prompt=str(self.config.get("llm_system_prompt", "") or "") or None,
            )
        except Exception as e:
            yield event.plain_result(f"LLM 调用失败：{e}")
            return
        if resp.role == "err" or not (resp.completion_text or "").strip():
            yield event.plain_result("LLM 没有返回有效内容")
            return
        text = resp.completion_text.strip()
        path, err, voice = await self._synthesize_or_none(event, self._clean_for_tts(text))
        if path:
            record = Record.fromFileSystem(str(path))
            record.text = text
            yield event.chain_result([record])
        else:
            # 合成失败，回退文字（附上 LLM 生成的文案）
            yield event.plain_result(
                f"[kktts] 语音合成失败：{err}（音色：{voice.get('name', '?')}）\n"
                f"文字回退：\n{text}"
            )

    # ---------------- 指令：自动转语音设置 ----------------

    @filter.command("tts自动", alias={"自动语音", "tts自动语音"})
    async def tts_auto(self, event: AstrMessageEvent, args: GreedyStr = ""):
        """查看或设置当前会话的 LLM 回复自动转语音概率：tts自动 [0-100|开|关]"""
        umo = event.unified_msg_origin
        arg = str(args).strip()
        if not arg:
            s = self.state.get(umo)
            prob = self._session_probability(event)
            src = "会话" if s.get("probability") is not None else "全局"
            enabled = self._auto_enabled(event)
            yield event.plain_result(
                f"本会话自动转语音：{'✅ 开启' if enabled else '❌ 关闭'}，概率 {prob}%（{src}）\n"
                "设置概率：tts自动 <0-100>\n"
                "本会话总开关：tts自动 开 / 关\n"
                "跟随全局：tts自动 重置"
            )
            return
        if arg in ("开", "开启", "on"):
            self.state.pop(umo, "enabled")
            prob = self._session_probability(event)
            yield event.plain_result(f"本会话自动转语音已开启，概率 {prob}%")
            return
        if arg in ("关", "关闭", "off"):
            self.state.set(umo, enabled=False)
            yield event.plain_result("本会话自动转语音已关闭，语音合成指令不受影响")
            return
        if arg in ("重置", "reset"):
            self.state.pop(umo, "enabled")
            self.state.pop(umo, "probability")
            prob = int(self.config.get("auto_probability", 10))
            yield event.plain_result(f"已重置为跟随全局：{'开启' if prob > 0 else '关闭'}，概率 {prob}%")
            return
        if arg.isdigit():
            prob = max(0, min(100, int(arg)))
            self.state.set(umo, probability=prob, enabled=True)
            yield event.plain_result(
                f"本会话自动转语音概率已设为 {prob}%{'（已关闭）' if prob == 0 else ''}"
            )
            return
        yield event.plain_result("用法：tts自动 <0-100> / 开 / 关 / 重置")

    # ---------------- 指令：状态与刷新 ----------------

    @filter.command("tts状态")
    async def tts_status(self, event: AstrMessageEvent):
        """查看插件当前状态"""
        voice = self._session_voice(event)
        mode = str(self.config.get("auto_mode", "替换文本"))
        yield event.plain_result(
            "KK语音合成状态：\n"
            f"· 音色列表：{len(self._voices)} 个（{'已缓存' if self._voices else '未加载'}）\n"
            f"· 当前会话音色：{self._voice_desc(voice)}\n"
            f"· 自动转语音：{'开' if self._auto_enabled(event) else '关'}，"
            f"概率 {self._session_probability(event)}%，模式 {mode}\n"
            f"· 文本长度限制：{self.config.get('min_text_length', 2)}~"
            f"{self.config.get('max_text_length', 300)} 字，冷却 {self.config.get('cooldown', 30)} 秒"
        )

    @filter.command("tts刷新")
    async def tts_refresh(self, event: AstrMessageEvent):
        """强制刷新音色列表缓存"""
        ok = await self._ensure_voices(force=True)
        if ok:
            yield event.plain_result(f"音色列表已刷新，共 {len(self._voices)} 个音色")
        else:
            yield event.plain_result("刷新失败，接口暂不可用（将继续使用本地缓存）")

    # ---------------- LLM 工具（Function-calling） ----------------

    async def _speak_send(self, event: AstrMessageEvent, text: str, voice: dict):
        """合成并直接发送语音。返回 (path, err)"""
        path, err = await self.client.synthesize(text, voice["id"])
        if not path:
            return None, err
        self.client.cleanup_cache()
        record = Record.fromFileSystem(str(path))
        record.text = text
        await event.send(MessageChain(chain=[record]))
        return path, None

    @filter.llm_tool(name="kktts_speak")
    async def tool_speak(self, event: AstrMessageEvent, text: str, voice: str = ""):
        """让机器人立刻把一段文本用语音说出来，直接发到当前聊天。适用于用户要求「说/读/用语音讲」某段话、或你想主动开口说话时。

        调用成功后语音已发出，不要再把同样的文本用纯文字重复发一遍。

        Args:
            text(string): 要朗读的文本，适合口语，控制在 200 字以内
            voice(string): 可选。音色名称或音色 ID，留空则用当前会话已设置的音色。不确定名称时可先用 kktts_search_voice 查询
        """
        try:
            content = str(text).strip()
            if not content:
                return "text 为空，未合成。"
            await self._ensure_voices()
            kw = str(voice or "").strip()
            if kw:
                v = self._find_voice(kw)
                if not v:
                    return f"未找到音色「{kw}」，可用 kktts_search_voice 查询后重试。"
            else:
                v = self._session_voice(event)
            path, err = await self._speak_send(event, self._clean_for_tts(content), v)
            if not path:
                return f"语音合成失败：{err}。请改用纯文本回复。"
            # 工具已发语音，借用冷却窗口抑制紧随其后的概率自动转语音，避免连发两条
            self._last_auto_ts[event.unified_msg_origin] = time.time()
            logger.info(f"[kktts] LLM 工具已发送语音：{v.get('name', '?')} 长度={len(content)}")
            return f"已用音色「{v.get('name', '?')}」发送语音，长度 {len(content)} 字。"
        except Exception as e:
            logger.error(f"[kktts] kktts_speak 工具异常：{e}", exc_info=True)
            return f"语音发送异常：{e}。请改用纯文本回复。"

    @filter.llm_tool(name="kktts_search_voice")
    async def tool_search_voice(self, event: AstrMessageEvent, keyword: str):
        """按关键词搜索可用的音色列表，返回名称、分类和音色 ID。当用户想换音色但没给出确切名称、或你要为 kktts_speak 找音色 ID 时调用。

        Args:
            keyword(string): 关键词，如「甜妹」「萝莉」「御姐」，支持名称、分类、ID 的模糊匹配
        """
        try:
            kw = str(keyword or "").strip()
            if not kw:
                return "关键词为空。"
            ok = await self._ensure_voices()
            if not ok or not self._voices:
                return "音色列表暂不可用，请直接使用当前音色。"
            low = kw.lower()
            hits = [
                v
                for v in self._voices
                if low in v.get("name", "").lower()
                or low in v.get("id", "").lower()
                or low in v.get("category", "").lower()
            ]
            if not hits:
                return f"没有找到与「{kw}」相关的音色。"
            shown = hits[:10]
            lines = [f"共 {len(hits)} 个匹配（最多显示 10 个）："]
            lines += [f"- {self._voice_desc(v)}  id={v['id']}" for v in shown]
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[kktts] kktts_search_voice 工具异常：{e}", exc_info=True)
            return f"查询异常：{e}"

    # ---------------- LLM 回复概率自动转语音 ----------------

    def _auto_mode(self) -> str:
        return str(self.config.get("auto_mode", "替换文本"))

    def _should_auto_speak(self, event: AstrMessageEvent, text: str) -> bool:
        """自动转语音门槛：开关 / 概率 / 长度 / 冷却。任一不过返回 False。

        通过时会就地写下冷却时间戳，保证一次回复只转一条语音。
        """
        if not self._auto_enabled(event):
            return False
        prob = self._session_probability(event)
        if prob <= 0:
            return False
        min_len = int(self.config.get("min_text_length", 2))
        max_len = int(self.config.get("max_text_length", 300))
        if len(text) < min_len or len(text) > max_len:
            return False
        if random.uniform(0, 100) >= prob:
            return False
        umo = event.unified_msg_origin
        now = time.time()
        cooldown = int(self.config.get("cooldown", 30))
        if now - self._last_auto_ts.get(umo, 0.0) < cooldown:
            return False
        self._last_auto_ts[umo] = now
        return True

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """附加语音模式：文字照发，语音额外追发一条。

        替换文本模式刻意不在这一步做：本钩子之后还有 persist_llm_response（落群聊
        上下文）、trace、历史落库等消费者要读同一个 resp。在这里把 result_chain
        换成纯 Record，会让它们看到「正文已被删空的回复」——completion_text 是
        result_chain 的派生属性，链里没有 Plain 就读成空串，历史会整轮丢弃。
        替换动作交给发送前钩子 on_decorating_result。
        """
        try:
            if getattr(resp, "role", "") == "err":
                return
            if self._auto_mode() != "附加语音":
                return
            text = (resp.completion_text or "").strip()
            if not text or not self._should_auto_speak(event, text):
                return

            await self._ensure_voices()
            path, err, voice = await self._synthesize_or_none(event, self._clean_for_tts(text))
            if not path:
                # 合成失败：什么都不改，LLM 文字回复照常发送（失败回退文字）
                logger.warning(f"[kktts] 自动语音合成失败，回退文字：{err}")
                return

            record = Record.fromFileSystem(str(path))
            record.text = text
            try:
                await event.send(MessageChain(chain=[record]))
            except Exception as e:
                logger.error(f"[kktts] 追加语音发送失败：{e}")
                return
            logger.info(
                f"[kktts] 已自动转语音（附加语音）音色={voice.get('name', '?')} 长度={len(text)}"
            )
        except Exception as e:
            # 任何异常都不能影响正常的文字回复
            logger.error(f"[kktts] on_llm_response 处理异常：{e}", exc_info=True)

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        """替换文本模式：发送前把出站消息链换成语音。

        放在这里的原因：此阶段已无其他消费者要读 LLMResponse，替换出站链不会
        动 resp.result_chain，completion_text 保持完整，历史与群聊上下文都能拿到
        正文，Record 组件也不会进上下文。
        只处理非流式 LLM 结果：流式文字已先行流出、无法撤回，替换模式会变成
        「文字已发、语音再发一条」；指令/错误等非 LLM 结果一律跳过。
        """
        try:
            if self._auto_mode() != "替换文本":
                return
            result = event.get_result()
            if result is None or not result.chain:
                return
            if not result.is_llm_result():
                return  # 流式结果 / 指令结果 / 错误结果：不转语音
            text = " ".join(
                comp.text for comp in result.chain if isinstance(comp, Plain)
            ).strip()
            if not text or not self._should_auto_speak(event, text):
                return

            await self._ensure_voices()
            path, err, voice = await self._synthesize_or_none(event, self._clean_for_tts(text))
            if not path:
                logger.warning(f"[kktts] 自动语音合成失败，回退文字：{err}")
                return

            record = Record.fromFileSystem(str(path))
            record.text = text
            result.chain = [record]
            logger.info(
                f"[kktts] 已自动转语音（替换文本）音色={voice.get('name', '?')} 长度={len(text)}"
            )
        except Exception as e:
            # 出任何问题都保持原文字，不影响发送
            logger.error(f"[kktts] on_decorating_result 处理异常：{e}", exc_info=True)

    # ---------------- 生命周期 ----------------

    async def terminate(self):
        if self._load_task and not self._load_task.done():
            self._load_task.cancel()
        self.client.cleanup_cache(keep_seconds=0)
