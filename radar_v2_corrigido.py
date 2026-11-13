"""Radar contextual APITube v2 — somente informativo, sem decisões de trading."""
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

STATE = Path('radar_noticias_v12.json')  # mantém histórico existente
URL = 'https://api.apitube.io/v1/news/everything'
MAX_REQUESTS_DAY = 4
TOPICS = [('bitcoin', 'BTC / cripto'), ('brazil', 'Brasil / macro')]
MAX_ALERTS = 3

# Regras transparentes e conservadoras: relevância textual, NÃO previsão de preço.
ASSETS = {
    'BTC': ('bitcoin', 'btc', 'bitcoin etf', 'etf de bitcoin'),
    'ETH': ('ethereum', 'ether', 'eth', 'ethereum etf'),
    'PETR4': ('petrobras', 'petr4'),
    'VALE3': ('vale sa', 'vale s.a.', 'vale3', 'mineradora vale'),
    'ITUB4': ('itau', 'itaú', 'itub4'),
    'BBAS3': ('banco do brasil', 'bbas3'),
    'WEGE3': ('weg sa', 'weg s.a.', 'wege3'),
    'ITSA4': ('itausa', 'itaúsa', 'itsa4'),
    'BBDC4': ('bradesco', 'bbdc4'),
    'B3SA3': ('b3sa3', 'bolsa brasileira', 'bolsa de valores brasileira'),
    'PRIO3': ('prio3', 'petro rio', 'prIO petróleo'),
    'ABEV3': ('ambev', 'abev3'),
}
EVENTS = (
    'juros', 'interest rate', 'selic', 'fed ', 'federal reserve', 'copom',
    'inflacao', 'inflação', 'inflation', 'ipca', 'payroll', 'emprego',
    'balanco', 'balanço', 'earnings', 'lucro', 'prejuizo', 'prejuízo',
    'dividend', 'regulacao', 'regulação', 'regulation', 'sec ', 'cvm',
    'etf', 'hack', 'fraude', 'sanction', 'sanção', 'tarifa', 'tariff',
    'guerra', 'war ', 'oil', 'petroleo', 'petróleo', 'opep', 'opec',
    'dolar', 'dólar', 'cambio', 'câmbio', 'pib', 'gdp', 'banco central',
    'central bank', 'bitcoin', 'ethereum', 'criptomoeda', 'crypto',
)
MACRO = ('brazil', 'brasil', 'brazilian', 'brasileir', 'ibovespa', 'b3 ',
         'selic', 'copom', 'ipca', 'banco central', 'federal reserve',
         'fed ', 'juros', 'inflation', 'inflação', 'dolar', 'dólar',
         'pib', 'gdp', 'tarifa', 'tariff', 'oil', 'petroleo', 'petróleo')


def _now():
    return datetime.now(timezone.utc)


def _load():
    try:
        value = json.loads(STATE.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(value):
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(STATE)


def _articles(payload):
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ('results', 'articles', 'news', 'data'):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            result = _articles(value)
            if result:
                return result
    return []


def _plain(value):
    if not isinstance(value, (str, int, float)):
        return ''
    return re.sub(r'\s+', ' ', str(value)).strip()


def _url(value):
    value = _plain(value)
    try:
        parsed = urlparse(value)
        if parsed.scheme in ('http', 'https') and parsed.netloc and not parsed.username:
            return value[:1000]
    except ValueError:
        pass
    return ''


def _source(article):
    source = article.get('source')
    if isinstance(source, dict):
        name = _plain(source.get('name') or source.get('title') or source.get('domain'))
        link = _url(source.get('url') or source.get('link'))
    else:
        name = _plain(source)
        link = ''
    name = name or _plain(article.get('source_name') or article.get('sourceName') or article.get('domain'))
    return name[:100] or 'não identificada', link


def _date(article):
    value = article.get('published_at') or article.get('publishedAt') or article.get('date') or article.get('published')
    if isinstance(value, dict):
        value = value.get('date') or value.get('value')
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value / (1000 if value > 1e11 else 1), tz=timezone.utc).strftime('%d/%m/%Y %H:%M UTC')
        except (ValueError, OverflowError, OSError):
            return 'não informado'
    value = _plain(value)
    if not value:
        return 'não informado'
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            return value[:35] + ' (fuso não informado)'
        return dt.astimezone(timezone.utc).strftime('%d/%m/%Y %H:%M UTC')
    except ValueError:
        return value[:45]


def _score(article, category):
    title = _plain(article.get('title')).lower()
    description = _plain(article.get('description') or article.get('summary')).lower()
    text = title + ' ' + description
    assets = [ticker for ticker, terms in ASSETS.items() if any(term.lower() in text for term in terms)]
    event_hits = sum(1 for term in EVENTS if term in text)
    macro_hits = sum(1 for term in MACRO if term in text)
    if not assets and not macro_hits:
        return 0, []
    # Notícias de BTC precisam falar de BTC/ETH; macro Brasil precisa de contexto econômico.
    if category == 'BTC / cripto' and not any(x in assets for x in ('BTC', 'ETH')):
        return 0, []
    if category == 'Brasil / macro' and not macro_hits and not assets:
        return 0, []
    points = min(5, 3 * len(assets)) + min(3, event_hits) + min(2, macro_hits)
    return points, assets


def executar_radar(enviar=None):
    """Até 2 janelas UTC/dia, 2 consultas/janela; preserva estado v12."""
    key = os.getenv('APITUBE_KEY', '').strip()
    if not key:
        print('📰 Radar: APITUBE_KEY ausente; ignorando.')
        return
    now = _now()
    day = now.date().isoformat()
    state = _load()
    if state.get('day') != day:
        state['day'], state['requests'], state['runs'] = day, 0, []
    runs = state.get('runs') if isinstance(state.get('runs'), list) else []
    try:
        requests_today = max(0, int(state.get('requests', 0)))
    except (ValueError, TypeError):
        requests_today = MAX_REQUESTS_DAY  # estado inválido: falha segura
    slot = 'am' if now.hour < 12 else 'pm'
    if slot in runs or requests_today >= MAX_REQUESTS_DAY:
        print('📰 Radar: janela já executada ou limite interno atingido.')
        return
    seen = state.get('seen') if isinstance(state.get('seen'), dict) else {}
    candidates = []
    for topic, category in TOPICS:
        if requests_today >= MAX_REQUESTS_DAY:
            break
        requests_today += 1
        state['requests'] = requests_today
        _save(state)  # reserva a consulta antes de chamar a API
        try:
            response = requests.get(
                URL, params={'title': topic, 'per_page': 10},
                headers={'X-API-Key': key, 'Accept': 'application/json'}, timeout=15,
            )
            if response.status_code != 200:
                print(f'📰 APITube HTTP {response.status_code} ({category}); sem retry.')
                continue
            articles = _articles(response.json())
            print(f'📰 APITube {category}: {len(articles)} artigos recebidos.')
            for article in articles:
                if not isinstance(article, dict):
                    continue
                title = _plain(article.get('title'))[:180]
                if not title:
                    continue
                link = _url(article.get('url') or article.get('link') or article.get('article_url'))
                identifier = hashlib.sha256((link or title.lower()).encode('utf-8')).hexdigest()[:20]
                if identifier in seen:
                    continue
                # Preserva comportamento original: marca todos como vistos, mesmo os filtrados.
                seen[identifier] = day
                points, assets = _score(article, category)
                if points < 3:
                    continue
                source, source_link = _source(article)
                candidates.append({
                    'score': points, 'category': category, 'title': title,
                    'link': link or source_link, 'source': source,
                    'published': _date(article), 'assets': assets,
                })
        except (requests.RequestException, ValueError) as exc:
            print(f'📰 Falha APITube ({category}): {type(exc).__name__}')
    runs.append(slot)
    state['runs'] = runs
    state['seen'] = dict(list(seen.items())[-600:])
    _save(state)
    candidates.sort(key=lambda item: item['score'], reverse=True)
    print(f'📰 Radar v2: {len(candidates)} notícias relevantes inéditas; '
          f'{requests_today}/{MAX_REQUESTS_DAY} consultas reservadas hoje.')
    if not callable(enviar):
        return
    for item in candidates[:MAX_ALERTS]:
        msg = (
            '📰 RADAR DE NOTÍCIAS — CONTEXTO (v2)\n'
            '⚠️ APITube Free: acesso com atraso de 12h; não é alerta em tempo real.\n'
            f"Tema: {item['category']}\n"
            f"Ativos citados: {', '.join(item['assets']) if item['assets'] else 'contexto macro'}\n"
            f"Título: {item['title']}\n"
            f"Publicado: {item['published']}\n"
            f"Consultado: {now.strftime('%d/%m/%Y %H:%M UTC')}\n"
            f"Veículo: {item['source']}\n"
            f"Link: {item['link'] or 'não disponível'}\n"
            'Relevância: filtro textual (não mede impacto ou confiabilidade).\n'
            'Não é recomendação de compra ou venda.'
        )
        try:
            enviar(msg, markdown=False)
        except Exception as exc:
            print(f'📰 Falha ao enviar alerta: {type(exc).__name__}')
