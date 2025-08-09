from notion_client import Client

def create_page(notion_token, database_id, title, date_str, language, source, url, blocks):
    notion = Client(auth=notion_token)
    props = {
        "Title": {"title": [{"text": {"content": title}}]},
        "Date": {"date": {"start": date_str}},
        "Language": {"multi_select": [{"name": language}]},
        "Source": {"rich_text": [{"text": {"content": source}}]},
        "URL": {"url": url},
    }
    return notion.pages.create(parent={"database_id": database_id}, properties=props, children=blocks)

def text_block(text):
    return {"object": "block","type": "paragraph","paragraph": {"rich_text": [{"type": "text","text": {"content": text}}]}}

def heading(text, level=2):
    key = f"heading_{level}"
    return {"object":"block","type":key, key: {"rich_text":[{"type":"text","text":{"content":text}}]}}

def bulleted_item(text):
    return {"object":"block","type":"bulleted_list_item",
            "bulleted_list_item":{"rich_text":[{"type":"text","text":{"content":text}}]}}

def table_block(vocab_rows):
    blocks = [heading("語彙（Vocab）", 3)]
    for jp, en_zh, hint in vocab_rows:
        line = f"{en_zh}  —  {jp}  ({hint})" if hint else f"{en_zh}  —  {jp}"
        blocks.append(bulleted_item(line))
    return blocks

def list_block(title, items):
    blocks = [heading(title, 3)]
    for it in items:
        blocks.append(bulleted_item(it))
    return blocks
