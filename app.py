import json
import csv
import io
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import streamlit as st
from snowflake.snowpark.context import get_active_session

DEFAULT_SYSTEM_PROMPT = "Tu es un assistant utile."
HISTORY_TABLE = "CHAT_MESSAGES"
CROSS_REGION_SQL = "ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';"

FALLBACK_MODELS: List[str] = [
    "claude-4-sonnet",
    "claude-3-7-sonnet",
    "claude-3-5-sonnet",
    "openai-gpt-4.1",
    "openai-o4-mini",
    "llama3.1-70b",
    "mistral-large2",
    "snowflake-arctic",
]


def get_models(session) -> Tuple[List[str], str]:
    """Try to list Cortex-supported models from account metadata."""
    try:
        rows = session.sql("SHOW MODELS IN SCHEMA SNOWFLAKE.MODELS").collect()
        names: List[str] = []
        for row in rows:
            data = row.as_dict()
            name = data.get("name") or data.get("NAME")
            if name:
                names.append(str(name))

        if names:
            return sorted(set(names), key=str.lower), ""
        return FALLBACK_MODELS, ""
    except Exception as exc:  # noqa: BLE001
        _ = exc
        return FALLBACK_MODELS, ""


def ensure_history_table(session) -> str:
    """Create persistence table if needed. Returns empty string on success."""
    try:
        session.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {HISTORY_TABLE} (
                conversation_id STRING,
                user_name STRING,
                \"timestamp\" TIMESTAMP_NTZ,
                role STRING,
                content STRING
            )
            """
        ).collect()
        session.sql(f"ALTER TABLE {HISTORY_TABLE} ADD COLUMN IF NOT EXISTS user_name STRING").collect()
        return ""
    except Exception as exc:  # noqa: BLE001
        return f"Table de persistance non disponible ({exc})."


def get_current_user(session) -> str:
    try:
        rows = session.sql("SELECT CURRENT_USER() AS user_name").collect()
        if not rows:
            return "UNKNOWN_USER"
        row = rows[0].as_dict()
        value = row.get("USER_NAME") or row.get("user_name")
        return str(value) if value else "UNKNOWN_USER"
    except Exception:
        return "UNKNOWN_USER"


def insert_message(session, conversation_id: str, user_name: str, role: str, content: str) -> None:
    if not content:
        return
    session.sql(
        f"""
        INSERT INTO {HISTORY_TABLE} (conversation_id, user_name, \"timestamp\", role, content)
        VALUES (?, ?, CURRENT_TIMESTAMP(), ?, ?)
        """,
        params=[conversation_id, user_name, role, content],
    ).collect()


def list_user_conversations(session, user_name: str, limit: int = 50) -> List[Dict[str, str]]:
    rows = session.sql(
        f"""
        WITH user_rows AS (
            SELECT conversation_id, \"timestamp\", role, content
            FROM {HISTORY_TABLE}
            WHERE user_name = ?
        ),
        stats AS (
            SELECT conversation_id, MAX(\"timestamp\") AS last_timestamp, COUNT(*) AS message_count
            FROM user_rows
            GROUP BY conversation_id
        ),
        first_user AS (
            SELECT conversation_id, content AS first_user_message
            FROM (
                SELECT
                    conversation_id,
                    content,
                    ROW_NUMBER() OVER (PARTITION BY conversation_id ORDER BY \"timestamp\" ASC) AS rn
                FROM user_rows
                WHERE role = 'user'
            )
            WHERE rn = 1
        )
        SELECT
            s.conversation_id,
            s.last_timestamp,
            s.message_count,
            COALESCE(f.first_user_message, 'Nouvelle conversation') AS conversation_title
        FROM stats s
        LEFT JOIN first_user f ON s.conversation_id = f.conversation_id
        ORDER BY s.last_timestamp DESC
        LIMIT {int(limit)}
        """,
        params=[user_name],
    ).collect()

    conversations: List[Dict[str, str]] = []
    for row in rows:
        data = row.as_dict()
        conversation_id = data.get("CONVERSATION_ID") or data.get("conversation_id")
        last_timestamp = data.get("LAST_TIMESTAMP") or data.get("last_timestamp")
        message_count = data.get("MESSAGE_COUNT") or data.get("message_count") or 0
        conversation_title = data.get("CONVERSATION_TITLE") or data.get("conversation_title") or ""
        if conversation_id:
            conversations.append(
                {
                    "conversation_id": str(conversation_id),
                    "last_timestamp": str(last_timestamp),
                    "message_count": str(message_count),
                    "conversation_title": str(conversation_title),
                }
            )
    return conversations


def load_conversation(session, conversation_id: str, user_name: str) -> List[Dict[str, str]]:
    rows = session.sql(
        f"""
        SELECT role, content
        FROM {HISTORY_TABLE}
        WHERE conversation_id = ?
          AND user_name = ?
        ORDER BY \"timestamp\" ASC
        """,
        params=[conversation_id, user_name],
    ).collect()

    messages: List[Dict[str, str]] = []
    for row in rows:
        role = str(row["ROLE"]).lower()
        content = str(row["CONTENT"])
        if role in {"system", "user", "assistant"}:
            messages.append({"role": role, "content": content})

    if messages and messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

    return messages


def build_full_history(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    system_messages = [msg for msg in messages if msg["role"] == "system"]
    dialogue = [msg for msg in messages if msg["role"] in {"user", "assistant"}]
    if system_messages:
        return [system_messages[0], *dialogue]
    return dialogue


def conversation_export_payload(messages: List[Dict[str, str]], conversation_id: str, user_name: str) -> str:
    payload = {
        "conversation_id": conversation_id,
        "user_name": user_name,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "messages": messages,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def conversation_to_csv(messages: List[Dict[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["index", "role", "content"])
    for idx, message in enumerate(messages, start=1):
        writer.writerow([idx, message.get("role", ""), message.get("content", "")])
    return buffer.getvalue()


def parse_cortex_response(raw_response: Any) -> str:
    if raw_response is None:
        return "Aucune reponse retournee par Cortex."

    payload: Any = raw_response
    if isinstance(raw_response, str):
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError:
            return raw_response

    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                text = first.get("messages") or first.get("text")
                if isinstance(text, str):
                    return text.strip()

                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content.strip()
                    if isinstance(content, list):
                        parts = [part.get("text", "") for part in content if isinstance(part, dict)]
                        joined = "\n".join(part for part in parts if part).strip()
                        if joined:
                            return joined

        for key in ("response", "output_text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return json.dumps(payload, ensure_ascii=False, indent=2)

    return str(raw_response)


def format_cortex_exception(exc: Exception, model: str) -> str:
    message = str(exc)
    lowered = message.lower()

    if "unavailable in your region" in lowered or "cross region inference" in lowered:
        return (
            f"Le modele `{model}` est indisponible dans la region actuelle.\n\n"
            "Demandez a un ACCOUNTADMIN d'activer le cross-region inference:\n"
            f"`{CROSS_REGION_SQL}`\n\n"
            "Puis relancez le chat, ou choisissez un autre modele."
        )

    return f"Erreur lors de l'appel Cortex: {message}"


def call_cortex(session, model: str, history: List[Dict[str, str]], temperature: float) -> str:
    query = """
    SELECT SNOWFLAKE.CORTEX.TRY_COMPLETE(
        ?,
        PARSE_JSON(?),
        PARSE_JSON(?)
    ) AS RESPONSE
    """
    effective_temperature = min(max(temperature, 0.0), 1.0)
    options = {"temperature": effective_temperature}

    rows = session.sql(
        query,
        params=[
            model,
            json.dumps(history, ensure_ascii=False),
            json.dumps(options),
        ],
    ).collect()

    raw_response = rows[0]["RESPONSE"] if rows else None
    if raw_response is None:
        return (
            "Cortex n'a pas pu traiter cette requete (TRY_COMPLETE a retourne NULL).\n\n"
            "Cause probable: modele indisponible dans votre region ou acces modele non autorise.\n"
            "Demandez a un ACCOUNTADMIN d'executer:\n"
            f"`{CROSS_REGION_SQL}`\n\n"
            "Ensuite, reessayez ou changez de modele."
        )

    return parse_cortex_response(raw_response)


def init_state() -> None:
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid4())

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": st.session_state.system_prompt}]

    if "models" not in st.session_state:
        st.session_state.models = FALLBACK_MODELS

    if "model_warning" not in st.session_state:
        st.session_state.model_warning = ""

    if "table_warning" not in st.session_state:
        st.session_state.table_warning = ""

    if "system_saved" not in st.session_state:
        st.session_state.system_saved = False

    if "current_user" not in st.session_state:
        st.session_state.current_user = "UNKNOWN_USER"

    if "user_conversations" not in st.session_state:
        st.session_state.user_conversations = []

    if "selected_conversation_id" not in st.session_state:
        st.session_state.selected_conversation_id = ""


def start_new_chat(session, system_prompt: str, user_name: str) -> None:
    st.session_state.conversation_id = str(uuid4())
    st.session_state.selected_conversation_id = st.session_state.conversation_id
    st.session_state.messages = [{"role": "system", "content": system_prompt}]
    st.session_state.system_saved = False

    if not st.session_state.table_warning:
        try:
            insert_message(session, st.session_state.conversation_id, user_name, "system", system_prompt)
            st.session_state.system_saved = True
        except Exception as exc:  # noqa: BLE001
            st.session_state.table_warning = f"Insertion en base impossible ({exc})."


def refresh_user_conversations(session) -> None:
    if st.session_state.table_warning:
        return
    try:
        st.session_state.user_conversations = list_user_conversations(session, st.session_state.current_user)
    except Exception:
        st.session_state.user_conversations = []


st.set_page_config(page_title="Chatbot Cortex - Snowflake", layout="wide")
st.title("Chatbot Cortex dans Snowflake")

session = get_active_session()
init_state()
st.session_state.current_user = get_current_user(session)
st.session_state.model_warning = ""

if st.session_state.models == FALLBACK_MODELS:
    models, _ = get_models(session)
    st.session_state.models = models

if not st.session_state.table_warning:
    st.session_state.table_warning = ensure_history_table(session)

if not st.session_state.table_warning and not st.session_state.system_saved:
    try:
        system_message = st.session_state.messages[0]["content"]
        insert_message(session, st.session_state.conversation_id, st.session_state.current_user, "system", system_message)
        st.session_state.system_saved = True
    except Exception as exc:  # noqa: BLE001
        st.session_state.table_warning = f"Insertion systeme impossible ({exc})."

refresh_user_conversations(session)

with st.sidebar:
    st.subheader("Parametres")

    if st.button("Rafraichir modeles"):
        models, _ = get_models(session)
        st.session_state.models = models

    model_list = st.session_state.models
    default_model = "claude-3-5-sonnet"
    default_index = model_list.index(default_model) if default_model in model_list else 0

    selected_model = st.selectbox("Modele Cortex", model_list, index=default_index)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.2, step=0.1)
    if temperature > 1.0:
        st.caption("Snowflake Cortex limite `temperature` a 1.0. La valeur envoyee sera 1.0.")

    prompt_input = st.text_area(
        "Instruction systeme",
        value=st.session_state.system_prompt,
        help="Ce message est stocke avec le role 'system' mais n'est pas affiche dans le chat.",
    )

    if st.button("Nouveau Chat"):
        cleaned = prompt_input.strip() or DEFAULT_SYSTEM_PROMPT
        st.session_state.system_prompt = cleaned
        start_new_chat(session, cleaned, st.session_state.current_user)
        refresh_user_conversations(session)
        st.rerun()

    st.markdown("### Mes conversations")
    if st.button("Rafraichir mes conversations"):
        st.session_state.user_conversations = list_user_conversations(session, st.session_state.current_user)

    conversation_options = st.session_state.user_conversations
    if conversation_options:
        conv_by_id: Dict[str, Dict[str, str]] = {
            conv["conversation_id"]: conv for conv in conversation_options
        }
        option_ids = list(conv_by_id.keys())

        if st.session_state.selected_conversation_id not in option_ids:
            if st.session_state.conversation_id in option_ids:
                st.session_state.selected_conversation_id = st.session_state.conversation_id
            else:
                st.session_state.selected_conversation_id = option_ids[0]

        def format_conversation_option(conv_id: str) -> str:
            conv = conv_by_id[conv_id]
            title = " ".join(conv["conversation_title"].split())
            if len(title) > 70:
                title = f"{title[:67]}..."
            return f"{title} ({conv['message_count']} msg)"

        selected_id = st.selectbox(
            "Conversations de l'utilisateur courant",
            option_ids,
            format_func=format_conversation_option,
            key="selected_conversation_id",
        )

        if selected_id != st.session_state.conversation_id:
            try:
                loaded = load_conversation(session, selected_id, st.session_state.current_user)
                if loaded:
                    st.session_state.conversation_id = selected_id
                    st.session_state.messages = loaded
                    st.session_state.system_prompt = loaded[0]["content"] if loaded[0]["role"] == "system" else DEFAULT_SYSTEM_PROMPT
                    st.session_state.system_saved = True
                    refresh_user_conversations(session)
                    st.rerun()
                else:
                    st.warning("Aucune conversation trouvee pour cet utilisateur.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Erreur de chargement: {exc}")
    else:
        st.caption("Aucune conversation enregistree pour cet utilisateur.")

    st.markdown("### Export conversation")
    export_json = conversation_export_payload(
        st.session_state.messages, st.session_state.conversation_id, st.session_state.current_user
    )
    export_csv = conversation_to_csv(st.session_state.messages)
    file_prefix = f"conversation_{st.session_state.conversation_id}"
    st.download_button(
        "Exporter JSON",
        data=export_json,
        file_name=f"{file_prefix}.json",
        mime="application/json",
    )
    st.download_button(
        "Exporter CSV",
        data=export_csv,
        file_name=f"{file_prefix}.csv",
        mime="text/csv",
    )

cleaned_prompt = prompt_input.strip() or DEFAULT_SYSTEM_PROMPT
if cleaned_prompt != st.session_state.system_prompt:
    st.session_state.system_prompt = cleaned_prompt
    if st.session_state.messages and st.session_state.messages[0]["role"] == "system":
        st.session_state.messages[0]["content"] = cleaned_prompt

st.caption(
    f"Utilisateur: `{st.session_state.current_user}` | Conversation ID: `{st.session_state.conversation_id}`"
)

if st.session_state.table_warning:
    st.warning(st.session_state.table_warning)

if len(st.session_state.messages) == 1:
    st.info("Posez votre premiere question dans la zone de saisie en bas.")

for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input("Ecrivez votre message...")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    if not st.session_state.table_warning:
        try:
            insert_message(
                session, st.session_state.conversation_id, st.session_state.current_user, "user", user_prompt
            )
            refresh_user_conversations(session)
        except Exception as exc:  # noqa: BLE001
            st.session_state.table_warning = f"Insertion utilisateur impossible ({exc})."

    with st.chat_message("assistant"):
        payload = build_full_history(st.session_state.messages)
        with st.spinner("Generation en cours..."):
            try:
                displayed_answer = call_cortex(session, selected_model, payload, temperature)
            except Exception as exc:  # noqa: BLE001
                displayed_answer = format_cortex_exception(exc, selected_model)

        st.markdown(displayed_answer)

    st.session_state.messages.append({"role": "assistant", "content": displayed_answer})

    if not st.session_state.table_warning:
        try:
            insert_message(
                session, st.session_state.conversation_id, st.session_state.current_user, "assistant", displayed_answer
            )
            refresh_user_conversations(session)
        except Exception as exc:  # noqa: BLE001
            st.session_state.table_warning = f"Insertion assistant impossible ({exc})."
