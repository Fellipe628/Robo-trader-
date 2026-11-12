"""Radar APITube v12: contextual, delayed, read-only; never generates trades."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
import requests

STATE = Path('radar_noticias_v12.json')
URL = 'https://api.apitube.io/v1/news/everything'
MAX_REQUESTS_DAY = 4  # deliberately conservative; Free plan permits up to 100/day
TOPICS = [('bitcoin', 'BTC / cripto'), ('brazil', 'Brasil / macro')]

def _now():
    return datetime.now(timezone.utc)

def _load():
    try:
        data = json.loads(STATE.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}

def _save(data):
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(STATE)

def _articles(payload):
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ('results', 'articles', 'news'):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    data = payload.get('data')
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return _articles(data)
    return []

def _plain(value):
    return str(value or '').replace('\n', ' ').replace('\r', ' ').strip()

def executar_radar(enviar=None):
    """Runs at most twice per UTC day; persists budget and dedup state in repository."""
    key = os.getenv('APITUBE_KEY', '').strip()
    if not key:
        print('📰 Radar APITube: chave ausente; ignorando.')
        return
    now = _now()
    day = now.date().isoformat()
    state = _load()
    if state.get('day') != day:
        state['day'], state['requests'], state['runs'] = day, 0, []
    runs = state.get('runs', [])
    if not isinstance(runs, list):
        runs = []
    # Two windows daily, to avoid a request on every 15-minute workflow run.
    slot = 'am' if now.hour < 12 else 'pm'
    if slot in runs or int(state.get('requests', 0)) >= MAX_REQUESTS_DAY:
        print('📰 Radar APITube: janela já consultada ou orçamento esgotado.')
        return
    seen = state.get('seen', {})
    if not isinstance(seen, dict):
        seen = {}
    new_items = []
    for topic, category in TOPICS:
        if int(state.get('requests', 0)) >= MAX_REQUESTS_DAY:
            break
        # Charge budget BEFORE network call: failures count conservatively.
        state['requests'] = int(state.get('requests', 0)) + 1
        _save(state)
        try:
            response = requests.get(URL, params={'title': topic, 'per_page': 10},
                                    headers={'X-API-Key': key, 'Accept': 'application/json'}, timeout=15)
            if response.status_code != 200:
                print(f'📰 APITube HTTP {response.status_code} ({category}); sem repetir nesta janela.')
                continue
            articles = _articles(response.json())
            print(f'📰 APITube {category}: {len(articles)} artigos retornados.')
            for article in articles:
                if not isinstance(article, dict):
                    continue
                title = _plain(article.get('title'))[:180]
                link = _plain(article.get('url') or article.get('link'))
                if not title:
                    continue
                identifier = hashlib.sha256((link or title.lower()).encode('utf-8')).hexdigest()[:20]
                if identifier in seen:
                    continue
                seen[identifier] = day
                published = _plain(article.get('published_at') or article.get('publishedAt') or article.get('date'))
                new_items.append((category, title, link, published))
        except (requests.RequestException, ValueError) as exc:
            print(f'📰 APITube falhou ({category}): {type(exc).__name__}')
    runs.append(slot)
    state['runs'] = runs
    # Keep only recently seen items, limiting repository growth.
    state['seen'] = dict(list(seen.items())[-600:])
    _save(state)
    print(f'📰 Radar: {len(new_items)} artigos inéditos; {state["requests"]}/{MAX_REQUESTS_DAY} consultas reservadas hoje.')
    # Explicitly contextual, never labeled real-time; do not infer market direction.
    if new_items and callable(enviar):
        for category, title, link, published in new_items[:3]:
            if link and urlparse(link).scheme not in ('https', 'http'):
                link = ''
            msg = ('📰 RADAR DE NOTÍCIAS — CONTEXTO\n'
                   '⚠️ APITube Free: notícias com atraso de até 12h ou mais.\n'
                   f'Tema: {category}\nTítulo: {title}\n'
                   f'Publicado: {published or "não informado"}\n'
                   f'Fonte: {link or "não informada"}\n'
                   'Não é recomendação de compra ou venda.')
            # Use plain text to avoid Markdown parse errors from article titles.
            try:
                enviar(msg, markdown=False)
            except Exception as exc:
                print(f'📰 Não foi possível enviar alerta: {type(exc).__name__}')
