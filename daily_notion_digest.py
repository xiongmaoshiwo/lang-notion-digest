import os, re, random, feedparser, requests, jieba
from dateutil import tz
from datetime import datetime
from langdetect import detect
from trafilatura import fetch_url, extract
import yaml
import nltk
from nltk import pos_tag, word_tokenize
from notion_helper import create_page, text_block, heading, table_block, list_block

JST = tz.gettz("Asia/Tokyo")

with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

ITEMS_PER_LANG = CFG.get("items_per_lang", 1)
MIN_CHARS = CFG.get("min_chars", 800)
MAX_CHARS = CFG.get("max_chars", 8000)
USE_OPENAI = bool(CFG.get("use_openai", True))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if USE_OPENAI and not OPENAI_API_KEY:
    USE_OPENAI = False

EN_STOP = set("a an the of to in on for with and or as at by from is are was were be been being it its that this those these which who whom whose will would can could should may might must do does did have has had not no yes than then over under after before into out up down off about across among between within without per among through during despite because although unless including upon around each other more most such many much one two three new same own also just even still very really".split())
ZH_STOP = set(list("的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老儿尔两"))

def normalize_spaces(s:str)->str:
    return re.sub(r"\s+", " ", s or "").strip()

def fetch_article(url):
    html = fetch_url(url)
    if not html: return None
    text = extract(html, favor_recall=True, include_comments=False) or ""
    return normalize_spaces(text)

def pick_from_rss(feeds, want_lang):
    random.shuffle(feeds)
    for feed in feeds:
        fp = feedparser.parse(feed)
        for e in fp.entries[:10]:
            url = e.get("link")
            title = normalize_spaces(e.get("title",""))
            if not url: continue
            body = fetch_article(url)
            if not body: continue
            if len(body) < MIN_CHARS or len(body) > MAX_CHARS:
                continue
            try:
                lang = detect(body[:1000])
            except Exception:
                continue
            if (want_lang == "EN" and lang.startswith("en")) or (want_lang == "ZH" and lang.startswith("zh")):
                source = feed.split('/')[2].replace('www.','')
                return {"title":title, "url":url, "body":body, "lang":want_lang, "source":source}
    return None

def english_vocab(text, topn=12):
    words = [w for w in word_tokenize(text) if re.match(r"[A-Za-z][A-Za-z\-']*", w)]
    words = [w for w in words if w.lower() not in EN_STOP and len(w) > 2]
    tags = pos_tag(words)
    cand = [w for w,t in tags if t.startswith("NN") or t.startswith("VB") or t.startswith("JJ") or t=="FW"]
    freq = {}
    for w in cand:
        key = w.lower()
        freq[key] = freq.get(key,0)+1
    sorted_words = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for w,_ in sorted_words[:topn]]

def chinese_vocab(text, topn=12):
    import re as _re
    tokens = [t for t in jieba.cut(text) if len(t)>=2 and t not in ZH_STOP and not _re.match(r"^[0-9A-Za-z]+$", t)]
    freq = {}
    for w in tokens:
        freq[w] = freq.get(w,0)+1
    sorted_words = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for w,_ in sorted_words[:topn]]

def english_grammar_points(text):
    import re as _re
    pts = []
    if _re.search(r"\b(be|am|is|are|was|were|been|being)\b\s+\b[A-Za-z]+ed\b", text, _re.I):
        pts.append("受動態: be + 過去分詞（~される）")
    if _re.search(r"\bif\b.*\b(would|could|might|should)\b", text, _re.I):
        pts.append("仮定法: if + ... would/could ...")
    if _re.search(r"\b(has|have|had)\b\s+\b[A-Za-z]+ed\b", text, _re.I):
        pts.append("完了形: have/has/had + 過去分詞")
    if _re.search(r"\b(which|that|who|whom|whose)\b\s+\b\w+\b", text, _re.I):
        pts.append("関係節: 関係代名詞")
    if _re.search(r"\b(can|could|may|might|must|shall|should|will|would)\b\s+\w+", text, _re.I):
        pts.append("助動詞: 可能・推量・義務など")
    return list(dict.fromkeys(pts))

def chinese_grammar_points(text):
    pts = []
    if "被" in text:
        pts.append("被字句: 受け身（〜に〜される）")
    if "把" in text:
        pts.append("把字句: 目的語前置（〜を〜する）")
    if "了" in text:
        pts.append("アスペクト了: 完了・変化の了")
    if "过" in text:
        pts.append("過去経験の过: 〜したことがある")
    if "着" in text:
        pts.append("持続の着: 〜している(状態)")
    if "比" in text:
        pts.append("比較構文: A 比 B + 形容詞")
    return list(dict.fromkeys(pts))

def llm_explain(lang, title, body, vocab, grammar):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if lang == "EN":
        prompt = f"以下は英語記事です。日本語で要約（3-4文）し、語彙と文法を解説してください。\n候補語彙: {', '.join(vocab)}\n検出文法: {', '.join(grammar)}\nタイトル: {title}\n本文: {body[:2000]}"
    else:
        prompt = f"以下は中国語記事です。日本語で要約（3-4文）し、語彙と文法を解説してください。\n候補語彙: {', '.join(vocab)}\n検出文法: {', '.join(grammar)}\nタイトル: {title}\n本文: {body[:2000]}"
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.2)
    return resp.choices[0].message.content

def rule_based_explain(lang, title, body, vocab, grammar):
    summary = (body[:300] + "...") if len(body)>300 else body
    out = ["## 要約", f"- {summary}", "", "## 語彙（簡易）"]
    for w in vocab:
        out.append(f"- {w} : 重要語（頻出度ベース）")
    out += ["", "## 文法ポイント"] + [f"- {g}" for g in grammar]
    return "\n".join(out)

def main():
    en_item = pick_from_rss(CFG.get("english_rss", []), "EN")
    zh_item = pick_from_rss(CFG.get("chinese_rss", []), "ZH")
    picks = [x for x in [en_item, zh_item] if x][:ITEMS_PER_LANG*2]
    if not picks:
        print("No items picked.")
        return

    date_str = datetime.now(JST).strftime("%Y-%m-%d")
    notion_token = os.environ["NOTION_TOKEN"]
    notion_db = os.environ["NOTION_DB_ID"]

    for item in picks:
        title, url, body, lang, source = item["title"], item["url"], item["body"], item["lang"], item["source"]
        if lang == "EN":
            vocab = english_vocab(body)
            grammar = english_grammar_points(body)
        else:
            vocab = chinese_vocab(body)
            grammar = chinese_grammar_points(body)

        explanation = llm_explain(lang, title, body, vocab, grammar) if USE_OPENAI else rule_based_explain(lang, title, body, vocab, grammar)

        blocks = []
        blocks += [heading(f"{title}", 2), text_block(f"Source: {source}"), text_block(url)]
        blocks += [heading("解説", 2), ]
        for line in explanation.splitlines():
            if line.strip().startswith("## "):
                blocks.append(heading(line.strip().replace("## ", ""), 3))
            elif line.strip().startswith("- "):
                blocks.append({"object":"block","type":"bulleted_list_item","bulleted_list_item":{"rich_text":[{"type":"text","text":{"content":line[2:]}}]}})
            elif line.strip()=="":
                continue
            else:
                blocks.append(text_block(line.strip()))
        if not any((b.get("type") or "").startswith("heading_") and ("語彙" in str(b) or "文法" in str(b)) for b in blocks):
            blocks += table_block([(w, w, "") for w in vocab])
            if grammar:
                blocks += list_block("文法ポイント（検出）", grammar)

        create_page(notion_token, notion_db, title, date_str, lang, source, url, blocks)

if __name__ == "__main__":
    main()
