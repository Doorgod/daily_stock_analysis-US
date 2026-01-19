# -*- coding: utf-8 -*-
"""
===================================
大盘复盘分析模块
===================================

职责：
1. 获取大盘指数数据（CN：上证/深证/创业板；US：S&P500/Nasdaq/Dow/Russell）
2. 搜索市场新闻形成复盘情报
3. 使用大模型生成每日大盘复盘报告

说明：
- 通过环境变量 MARKET_REGION 切换市场：CN / US（默认 CN）
- US 模式下：
  - 指数与行业强弱通过 yfinance 获取
  - "全市场上涨/下跌家数"等统计需要全市场股票列表源，默认不启用（保留字段兼容输出）
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

import akshare as ak
import pandas as pd
import yfinance as yf

from config import get_config
from search_service import SearchService

logger = logging.getLogger(__name__)


@dataclass
class MarketIndex:
    """大盘指数数据"""
    code: str
    name: str
    current: float = 0.0
    change: float = 0.0
    change_pct: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    prev_close: float = 0.0
    volume: float = 0.0
    amount: float = 0.0
    amplitude: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'current': self.current,
            'change': self.change,
            'change_pct': self.change_pct,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'amount': self.amount,
            'amplitude': self.amplitude,
        }


@dataclass
class MarketOverview:
    """市场概览数据"""
    date: str
    indices: List[MarketIndex] = field(default_factory=list)

    # CN 可用统计；US 模式默认不统计（保留字段兼容输出）
    up_count: int = 0
    down_count: int = 0
    flat_count: int = 0
    limit_up_count: int = 0
    limit_down_count: int = 0

    total_amount: float = 0.0  # 亿元（CN）
    north_flow: float = 0.0    # 亿元（CN）

    top_sectors: List[Dict] = field(default_factory=list)
    bottom_sectors: List[Dict] = field(default_factory=list)


class MarketAnalyzer:
    """大盘复盘分析器"""

    CN_MAIN_INDICES = {
        'sh000001': '上证指数',
        'sz399001': '深证成指',
        'sz399006': '创业板指',
        'sh000688': '科创50',
        'sh000016': '上证50',
        'sh000300': '沪深300',
    }

    US_MAIN_INDICES = {
        '^GSPC': 'S&P 500',
        '^IXIC': 'NASDAQ',
        '^DJI': 'Dow Jones',
        '^RUT': 'Russell 2000',
    }

    # 用行业 ETF 近似美股"板块"强弱（1日涨跌幅）
    US_SECTOR_ETFS = {
        'Technology': 'XLK',
        'Financials': 'XLF',
        'Health Care': 'XLV',
        'Energy': 'XLE',
        'Industrials': 'XLI',
        'Consumer Discretionary': 'XLY',
        'Consumer Staples': 'XLP',
        'Utilities': 'XLU',
        'Materials': 'XLB',
        'Real Estate': 'XLRE',
        'Comm Services': 'XLC',
    }

    def __init__(self, search_service: Optional[SearchService] = None, analyzer=None):
        self.config = get_config()
        self.search_service = search_service
        self.analyzer = analyzer
        self.market_region = (getattr(self.config, 'market_region', 'CN') or 'CN').upper()

    def get_market_overview(self) -> MarketOverview:
        today = datetime.now().strftime('%Y-%m-%d')
        overview = MarketOverview(date=today)

        overview.indices = self._get_main_indices()
        self._get_market_statistics(overview)
        self._get_sector_rankings(overview)

        return overview

    # -----------------------------
    # AkShare (CN)
    # -----------------------------
    def _call_akshare_with_retry(self, fn, name: str, attempts: int = 2):
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as e:
                last_error = e
                logger.warning(f"[大盘/CN] {name} 获取失败 (attempt {attempt}/{attempts}): {e}")
                if attempt < attempts:
                    time.sleep(min(2 ** attempt, 5))
        logger.error(f"[大盘/CN] {name} 最终失败: {last_error}")
        return None

    def _get_cn_main_indices_ak(self) -> List[MarketIndex]:
        indices: List[MarketIndex] = []
        try:
            logger.info('[大盘/CN] 获取主要指数实时行情...')
            df = self._call_akshare_with_retry(ak.stock_zh_index_spot_sina, '指数行情', attempts=2)
            if df is None or df.empty:
                return indices

            for code, name in self.CN_MAIN_INDICES.items():
                row = df[df['代码'] == code]
                if row.empty:
                    row = df[df['代码'].astype(str).str.contains(code, na=False)]
                if row.empty:
                    continue

                r = row.iloc[0]
                idx = MarketIndex(
                    code=code,
                    name=name,
                    current=float(r.get('最新价', 0) or 0),
                    change=float(r.get('涨跌额', 0) or 0),
                    change_pct=float(r.get('涨跌幅', 0) or 0),
                    open=float(r.get('今开', 0) or 0),
                    high=float(r.get('最高', 0) or 0),
                    low=float(r.get('最低', 0) or 0),
                    prev_close=float(r.get('昨收', 0) or 0),
                    volume=float(r.get('成交量', 0) or 0),
                    amount=float(r.get('成交额', 0) or 0),
                )
                if idx.prev_close > 0:
                    idx.amplitude = (idx.high - idx.low) / idx.prev_close * 100
                indices.append(idx)

            logger.info(f"[大盘/CN] 获取到 {len(indices)} 个指数行情")
        except Exception as e:
            logger.error(f"[大盘/CN] 获取指数行情失败: {e}")
        return indices

    def _get_cn_market_statistics_ak(self, overview: MarketOverview):
        try:
            logger.info('[大盘/CN] 获取市场涨跌统计...')
            df = self._call_akshare_with_retry(ak.stock_zh_a_spot_em, 'A股实时行情', attempts=2)
            if df is None or df.empty:
                return

            change_col = '涨跌幅'
            if change_col in df.columns:
                df[change_col] = pd.to_numeric(df[change_col], errors='coerce')
                overview.up_count = int((df[change_col] > 0).sum())
                overview.down_count = int((df[change_col] < 0).sum())
                overview.flat_count = int((df[change_col] == 0).sum())
                overview.limit_up_count = int((df[change_col] >= 9.9).sum())
                overview.limit_down_count = int((df[change_col] <= -9.9).sum())

            amount_col = '成交额'
            if amount_col in df.columns:
                df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
                overview.total_amount = float(df[amount_col].sum() / 1e8)  # 亿元

            logger.info(
                f"[大盘/CN] 涨:{overview.up_count} 跌:{overview.down_count} 平:{overview.flat_count} "
                f"涨停:{overview.limit_up_count} 跌停:{overview.limit_down_count} "
                f"成交额:{overview.total_amount:.0f}亿"
            )
        except Exception as e:
            logger.error(f"[大盘/CN] 获取涨跌统计失败: {e}")

    def _get_cn_sector_rankings_ak(self, overview: MarketOverview):
        try:
            logger.info('[大盘/CN] 获取板块涨跌榜...')
            df = self._call_akshare_with_retry(ak.stock_board_industry_name_em, '行业板块行情', attempts=2)
            if df is None or df.empty:
                return

            change_col = '涨跌幅'
            if change_col not in df.columns:
                return

            df[change_col] = pd.to_numeric(df[change_col], errors='coerce')
            df = df.dropna(subset=[change_col])

            top = df.nlargest(5, change_col)
            overview.top_sectors = [
                {'name': row['板块名称'], 'change_pct': float(row[change_col])}
                for _, row in top.iterrows()
            ]

            bottom = df.nsmallest(5, change_col)
            overview.bottom_sectors = [
                {'name': row['板块名称'], 'change_pct': float(row[change_col])}
                for _, row in bottom.iterrows()
            ]

            logger.info(f"[大盘/CN] 领涨板块: {[s['name'] for s in overview.top_sectors]}")
            logger.info(f"[大盘/CN] 领跌板块: {[s['name'] for s in overview.bottom_sectors]}")
        except Exception as e:
            logger.error(f"[大盘/CN] 获取板块涨跌榜失败: {e}")

    # -----------------------------
    # yfinance (US)
    # -----------------------------
    def _download_yf_daily(self, tickers: List[str], period: str = '10d') -> Optional[pd.DataFrame]:
        try:
            df = yf.download(
                tickers=tickers,
                period=period,
                interval='1d',
                group_by='ticker',
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            if df is None or df.empty:
                return None
            return df
        except Exception as e:
            logger.error(f"[大盘/US] yfinance download 失败: {e}")
            return None

    def _get_us_main_indices_yf(self) -> List[MarketIndex]:
        indices: List[MarketIndex] = []
        logger.info('[大盘/US] 获取主要指数行情 (yfinance)...')

        tickers = list(self.US_MAIN_INDICES.keys())
        df = self._download_yf_daily(tickers, period='10d')
        if df is None:
            return indices

        is_multi = isinstance(df.columns, pd.MultiIndex)

        for code, name in self.US_MAIN_INDICES.items():
            try:
                if is_multi:
                    if code not in df.columns.get_level_values(0):
                        continue
                    sub = df[code].dropna()
                else:
                    # 单 ticker 情况
                    sub = df.dropna()

                if sub.empty:
                    continue

                last = sub.iloc[-1]
                prev = sub.iloc[-2] if len(sub) >= 2 else last

                current = float(last.get('Close', 0) or 0)
                prev_close = float(prev.get('Close', 0) or 0)
                change = current - prev_close
                change_pct = (change / prev_close * 100) if prev_close else 0.0

                idx = MarketIndex(
                    code=code,
                    name=name,
                    current=current,
                    change=change,
                    change_pct=change_pct,
                    open=float(last.get('Open', 0) or 0),
                    high=float(last.get('High', 0) or 0),
                    low=float(last.get('Low', 0) or 0),
                    prev_close=prev_close,
                    volume=float(last.get('Volume', 0) or 0),
                    amount=0.0,
                )
                if idx.prev_close > 0:
                    idx.amplitude = (idx.high - idx.low) / idx.prev_close * 100
                indices.append(idx)
            except Exception as e:
                logger.warning(f"[大盘/US] {code} 获取失败: {e}")

        logger.info(f"[大盘/US] 获取到 {len(indices)} 个指数行情")
        return indices

    def _get_us_sector_rankings_yf(self, overview: MarketOverview):
        logger.info('[大盘/US] 获取行业强弱 (sector ETFs, yfinance)...')

        mapping = self.US_SECTOR_ETFS
        tickers = list(mapping.values())
        df = self._download_yf_daily(tickers, period='10d')
        if df is None:
            return

        is_multi = isinstance(df.columns, pd.MultiIndex)
        rows: List[Dict[str, Any]] = []

        for sector, ticker in mapping.items():
            try:
                if is_multi:
                    if ticker not in df.columns.get_level_values(0):
                        continue
                    sub = df[ticker].dropna()
                else:
                    sub = df.dropna()

                if sub.empty:
                    continue

                last = sub.iloc[-1]
                prev = sub.iloc[-2] if len(sub) >= 2 else last
                c = float(last.get('Close', 0) or 0)
                p = float(prev.get('Close', 0) or 0)
                chg = ((c - p) / p * 100) if p else 0.0
                rows.append({'name': sector, 'ticker': ticker, 'change_pct': chg})
            except Exception as e:
                logger.warning(f"[大盘/US] 行业ETF {ticker} 失败: {e}")

        if not rows:
            return

        rows = sorted(rows, key=lambda x: x['change_pct'], reverse=True)
        overview.top_sectors = [{'name': r['name'], 'change_pct': float(r['change_pct'])} for r in rows[:5]]
        overview.bottom_sectors = [{'name': r['name'], 'change_pct': float(r['change_pct'])} for r in rows[-5:]][::-1]

        logger.info(f"[大盘/US] 领涨行业: {[s['name'] for s in overview.top_sectors]}")
        logger.info(f"[大盘/US] 领跌行业: {[s['name'] for s in overview.bottom_sectors]}")

    # -----------------------------
    # Routing
    # -----------------------------
    def _get_main_indices(self) -> List[MarketIndex]:
        if self.market_region == 'US':
            return self._get_us_main_indices_yf()
        return self._get_cn_main_indices_ak()

    def _get_market_statistics(self, overview: MarketOverview):
        if self.market_region == 'US':
            logger.info('[大盘/US] 市场涨跌统计未启用（需要全市场股票列表源）')
            return
        return self._get_cn_market_statistics_ak(overview)

    def _get_sector_rankings(self, overview: MarketOverview):
        if self.market_region == 'US':
            return self._get_us_sector_rankings_yf(overview)
        return self._get_cn_sector_rankings_ak(overview)

    # -----------------------------
    # News + LLM
    # -----------------------------
    def search_market_news(self) -> List[Dict]:
        if not self.search_service:
            logger.warning('[大盘] 搜索服务未配置，跳过新闻搜索')
            return []

        all_news: List[Dict] = []
        today = datetime.now()

        if self.market_region == 'US':
            month_str = today.strftime('%B %Y')
            search_queries = [
                f"US stock market recap {month_str}",
                f"S&P 500 market wrap {month_str}",
                f"NASDAQ market wrap {month_str}",
                f"Fed rate outlook stock market {month_str}",
            ]
            stock_name = 'US Market'
        else:
            month_str = f"{today.year}年{today.month}月"
            search_queries = [
                f"A股 大盘 复盘 {month_str}",
                f"股市 行情 分析 今日 {month_str}",
                f"A股 市场 热点 板块 {month_str}",
            ]
            stock_name = '大盘'

        try:
            logger.info('[大盘] 开始搜索市场新闻...')
            for query in search_queries:
                response = self.search_service.search_stock_news(
                    stock_code='market',
                    stock_name=stock_name,
                    max_results=3,
                    focus_keywords=query.split(),
                )
                if response and getattr(response, 'results', None):
                    all_news.extend(response.results)
                    logger.info(f"[大盘] 搜索 '{query}' 获取 {len(response.results)} 条结果")
            logger.info(f"[大盘] 共获取 {len(all_news)} 条市场新闻")
        except Exception as e:
            logger.error(f"[大盘] 搜索市场新闻失败: {e}")

        return all_news

    def generate_market_review(self, overview: MarketOverview, news: List) -> str:
        if not self.analyzer or not self.analyzer.is_available():
            logger.warning('[大盘] AI分析器未配置或不可用，使用模板生成报告')
            return self._generate_template_review(overview, news)

        prompt = self._build_review_prompt(overview, news)

        try:
            logger.info('[大盘] 调用大模型生成复盘报告...')
            generation_config = {'temperature': 0.7, 'max_output_tokens': 2048}

            if getattr(self.analyzer, '_use_openai', False):
                review = self.analyzer._call_openai_api(prompt, generation_config)
            else:
                response = self.analyzer._model.generate_content(prompt, generation_config=generation_config)
                review = response.text.strip() if response and getattr(response, 'text', None) else None

            if review:
                logger.info(f"[大盘] 复盘报告生成成功，长度: {len(review)} 字符")
                return review

            logger.warning('[大盘] 大模型返回为空')
            return self._generate_template_review(overview, news)
        except Exception as e:
            logger.error(f"[大盘] 大模型生成复盘报告失败: {e}")
            return self._generate_template_review(overview, news)

    def _build_review_prompt(self, overview: MarketOverview, news: List) -> str:
        # 指数行情信息
        indices_lines: List[str] = []
        for idx in overview.indices:
            direction = '↑' if idx.change_pct > 0 else '↓' if idx.change_pct < 0 else '-'
            indices_lines.append(f"- {idx.name}: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)")
        indices_text = "\n".join(indices_lines) if indices_lines else "(暂无指数数据)"

        # 板块/行业信息
        top_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.top_sectors[:3]])
        bottom_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.bottom_sectors[:3]])

        # 新闻信息
        news_lines: List[str] = []
        for i, n in enumerate(news[:6], 1):
            if hasattr(n, 'title'):
                title = (n.title or '')[:60]
                snippet = (n.snippet or '')[:140]
            else:
                title = (n.get('title', '') or '')[:60]
                snippet = (n.get('snippet', '') or '')[:140]
            if title or snippet:
                news_lines.append(f"{i}. {title}\n   {snippet}")
        news_text = "\n".join(news_lines) if news_lines else "暂无相关新闻"

        if self.market_region == 'US':
            prompt = f"""你是一位专业的美股市场分析师，请根据以下数据生成一份简洁的【美股大盘复盘 / market wrap】报告。

【重要】输出要求：
- 必须输出纯 Markdown 文本格式
- 禁止输出 JSON 格式
- 禁止输出代码块
- emoji 仅在标题处少量使用（每个标题最多1个）

---

# 今日市场数据

## 日期
{overview.date}

## 主要指数
{indices_text}

## 行业表现（用 Sector ETFs 近一日涨跌幅近似）
强势: {top_sectors_text if top_sectors_text else '暂无'}
弱势: {bottom_sectors_text if bottom_sectors_text else '暂无'}

## 市场新闻
{news_text}

---

# 输出格式模板（请严格按此格式输出）

## 📊 {overview.date} 美股大盘复盘

### 一、市场总结
（2-3句话概括今日市场整体表现，包括主要指数涨跌、风险偏好变化）

### 二、指数点评
（分别点评 S&P 500、NASDAQ、Dow 等走势特点）

### 三、行业/主题强弱
（分析强势与弱势行业可能的驱动因素，如利率、油价、财报、政策预期等）

### 四、关键新闻与事件
（结合新闻列出 3-5 个关键驱动因素，并解释对市场的影响链条）

### 五、后市展望
（给出短期关注点：重要数据/财报/美联储表态、关键技术位等）

### 六、风险提示
（需要关注的风险点）

---

请直接输出复盘报告内容，不要输出其他说明文字。
"""
            return prompt

        # CN 模式
        prompt = f"""你是一位专业的A股市场分析师，请根据以下数据生成一份简洁的大盘复盘报告。

【重要】输出要求：
- 必须输出纯 Markdown 文本格式
- 禁止输出 JSON 格式
- 禁止输出代码块
- emoji 仅在标题处少量使用（每个标题最多1个）

---

# 今日市场数据

## 日期
{overview.date}

## 主要指数
{indices_text}

## 市场概况
- 上涨: {overview.up_count} 家 | 下跌: {overview.down_count} 家 | 平盘: {overview.flat_count} 家
- 涨停: {overview.limit_up_count} 家 | 跌停: {overview.limit_down_count} 家
- 两市成交额: {overview.total_amount:.0f} 亿元
- 北向资金: {overview.north_flow:+.2f} 亿元

## 板块表现
领涨: {top_sectors_text if top_sectors_text else '暂无'}
领跌: {bottom_sectors_text if bottom_sectors_text else '暂无'}

## 市场新闻
{news_text}

---

# 输出格式模板（请严格按此格式输出）

## 📊 {overview.date} 大盘复盘

### 一、市场总结
（2-3句话概括今日市场整体表现，包括指数涨跌、成交量变化）

### 二、指数点评
（分析上证、深证、创业板等各指数走势特点）

### 三、资金动向
（解读成交额和北向资金流向的含义）

### 四、热点解读
（分析领涨领跌板块背后的逻辑和驱动因素）

### 五、后市展望
（结合当前走势和新闻，给出明日市场预判）

### 六、风险提示
（需要关注的风险点）

---

请直接输出复盘报告内容，不要输出其他说明文字。
"""
        return prompt

    def _generate_template_review(self, overview: MarketOverview, news: List) -> str:
        # 选一个代表指数判断市场情绪
        if self.market_region == 'US':
            key = '^GSPC'
        else:
            key = 'sh000001'

        rep = next((idx for idx in overview.indices if idx.code == key), None)
        if rep:
            if rep.change_pct > 1:
                market_mood = '强势上涨'
            elif rep.change_pct > 0:
                market_mood = '小幅上涨'
            elif rep.change_pct > -1:
                market_mood = '小幅下跌'
            else:
                market_mood = '明显下跌'
        else:
            market_mood = '震荡整理'

        indices_text = ''
        for idx in overview.indices[:4]:
            direction = '↑' if idx.change_pct > 0 else '↓' if idx.change_pct < 0 else '-'
            indices_text += f"- **{idx.name}**: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"

        top_text = '、'.join([s['name'] for s in overview.top_sectors[:3]]) if overview.top_sectors else '暂无'
        bottom_text = '、'.join([s['name'] for s in overview.bottom_sectors[:3]]) if overview.bottom_sectors else '暂无'

        if self.market_region == 'US':
            report = f"""## 📊 {overview.date} 美股大盘复盘

### 一、市场总结
今日美股市场整体呈现**{market_mood}**态势。

### 二、主要指数
{indices_text if indices_text else '(暂无指数数据)'}

### 三、行业强弱（Sector ETFs 近一日涨跌幅近似）
- **强势**: {top_text}
- **弱势**: {bottom_text}

### 四、风险提示
市场有风险，投资需谨慎。以上内容仅供参考，不构成投资建议。

---
*复盘时间: {datetime.now().strftime('%H:%M')}*
"""
            return report

        report = f"""## 📊 {overview.date} 大盘复盘

### 一、市场总结
今日A股市场整体呈现**{market_mood}**态势。

### 二、主要指数
{indices_text if indices_text else '(暂无指数数据)'}

### 三、涨跌统计
| 指标 | 数值 |
|------|------|
| 上涨家数 | {overview.up_count} |
| 下跌家数 | {overview.down_count} |
| 涨停 | {overview.limit_up_count} |
| 跌停 | {overview.limit_down_count} |
| 两市成交额 | {overview.total_amount:.0f}亿 |
| 北向资金 | {overview.north_flow:+.2f}亿 |

### 四、板块表现
- **领涨**: {top_text}
- **领跌**: {bottom_text}

### 五、风险提示
市场有风险，投资需谨慎。以上数据仅供参考，不构成投资建议。

---
*复盘时间: {datetime.now().strftime('%H:%M')}*
"""
        return report

    def run_daily_review(self) -> str:
        logger.info('========== 开始大盘复盘分析 =========')

        overview = self.get_market_overview()
        news = self.search_market_news()
        report = self.generate_market_review(overview, news)

        logger.info('========== 大盘复盘分析完成 =========')
        return report


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    )

    analyzer = MarketAnalyzer()
    overview = analyzer.get_market_overview()

    print('\n=== 市场概览 ===')
    print(f'市场区域: {analyzer.market_region}')
    print(f'日期: {overview.date}')
    print(f'指数数量: {len(overview.indices)}')
    for idx in overview.indices:
        print(f'  {idx.name}: {idx.current:.2f} ({idx.change_pct:+.2f}%)')
