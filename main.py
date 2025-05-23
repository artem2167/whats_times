# main.py

import os
import json
from datetime import datetime, timezone
import openai
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# Загрузим переменные из .env
load_dotenv()

# 1) Настройка OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# 2) Описываем схемы состояний для LangGraph
class InputState(TypedDict):
    message: str

class OutputState(TypedDict):
    response: str

class OverallState(InputState, OutputState):
    """Объединённый тип: содержит поля message и response"""
    ...

# 3) Инструменты и логика узлов
def get_current_time() -> dict:
    """Возвращает текущее время UTC в ISO‑8601 с суффиксом Z"""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    return {"utc": now.isoformat().replace("+00:00", "Z")}


def is_time_request(msg: str) -> bool:
    """True, если сообщение явно спрашивает текущее время"""
    system = {
        "role": "system",
        "content": (
            "Ты классификатор: отвечай только TRUE или FALSE.\n"
            "TRUE — если спрашивают текущее время; FALSE — если нет."
        )
    }
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system, {"role": "user", "content": msg}],
        temperature=0.0,
        max_tokens=5
    )
    return resp.choices[0].message.content.strip().upper().startswith("T")


def chat_node(state: InputState) -> OutputState:
    """Узел графа: принимает {'message': str}, возвращает {'response': str}. Если вопрос о времени — возвращает время, иначе проксирует в ChatGPT."""
    text = state["message"].strip()
    low = text.lower()
    # if any(k in low for k in ("сколько", "время", "time")):
    #     return {"response": f"Текущее время UTC: {get_current_time()['utc']}"}
    if is_time_request(text):
        return {"response": f"Текущее время UTC: {get_current_time()['utc']}"}
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": text}],
        temperature=0.7
    )
    return {"response": resp.choices[0].message.content.strip()}

# 4) Сборка графа
builder = StateGraph(
    OverallState,
    input=InputState,
    output=OutputState
)
builder.add_node("chat_node", chat_node)
builder.add_edge(START, "chat_node")
builder.add_edge("chat_node", END)

# 5) Экспортируем скомпилированный граф как ASGI-приложение
app = builder.compile()

# 6) Генерация конфигурации для langgraph-cli
def main():
    # Создаём конфигурацию CLI: graphs и зависимости
    graph_name = "chat_graph"
    config = {
        "graphs": {
            graph_name: f"{os.path.basename(__file__)[:-3]}:app"
        },
        # указываем зависимости (имена графов)
        "dependencies": [graph_name]
    }
    with open("langgraph.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print("langgraph.json создан — запустите 'langgraph dev' или используйте Docker")

if __name__ == "__main__":
    main()
