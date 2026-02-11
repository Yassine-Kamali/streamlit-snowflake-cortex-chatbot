import json
import csv
import io
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import streamlit as st
from snowflake.snowpark.context import get_active_session

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

DEFAULT_SYSTEM_PROMPT = "Tu es un assistant utile."
HISTORY_TABLE = "CHAT_MESSAGES"
RAG_DOCS_TABLE = "RAG_DOCUMENTS"
RAG_CHUNKS_TABLE = "RAG_CHUNKS"
CROSS_REGION_SQL = "ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';"
RAG_EMBED_MODEL = "multilingual-e5-large"
RAG_EMBED_DIMS = 1024

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


def ensure_rag_tables(session) -> str:
    try:
        session.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {RAG_DOCS_TABLE} (
                doc_id STRING,
                user_name STRING,
                doc_name STRING,
                source_type STRING,
                created_at TIMESTAMP_NTZ
            )
            """
        ).collect()
        session.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {RAG_CHUNKS_TABLE} (
                chunk_id STRING,
                doc_id STRING,
                user_name STRING,
                chunk_index NUMBER,
                chunk_text STRING,
                embedding VECTOR(FLOAT, {RAG_EMBED_DIMS}),
                created_at TIMESTAMP_NTZ
            )
            """
        ).collect()
        return ""
    except Exception as exc:  # noqa: BLE001
        return f"Tables RAG indisponibles ({exc})."


def split_text_into_chunks(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    clean = " ".join(text.split())
    if not clean:
        return []

    chunks: List[str] = []
    start = 0
    text_len = len(clean)
    step = max(chunk_size - overlap, 1)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start += step
    return chunks


def extract_document_title_and_content(uploaded_file) -> Tuple[str, str, str]:
    file_name = uploaded_file.name if uploaded_file is not None else "document"
    title = Path(file_name).stem.replace("_", " ").strip() or "document"
    raw = uploaded_file.getvalue() if uploaded_file is not None else b""
    suffix = Path(file_name).suffix.lower()
    warning = ""

    text = ""
    if suffix in {".txt", ".md", ".csv", ".json"}:
        text = raw.decode("utf-8", errors="ignore")
    elif suffix == ".pdf":
        if PdfReader is None:
            warning = "Lecture PDF limitee dans cet environnement. Preferer .txt/.md pour de meilleurs resultats."
            text = raw.decode("utf-8", errors="ignore")
        else:
            try:
                reader = PdfReader(io.BytesIO(raw))
                text = "\n".join([(page.extract_text() or "") for page in reader.pages])
            except Exception:
                warning = "Extraction PDF partielle ou impossible. Preferer .txt/.md."
                text = raw.decode("utf-8", errors="ignore")
    else:
        warning = "Format non optimal pour extraction texte. Preferer .txt/.md/.pdf."
        text = raw.decode("utf-8", errors="ignore")

    return title, text.strip(), warning


def auto_index_uploaded_rag_file(session, user_name: str, uploaded_file) -> Tuple[bool, str]:
    if uploaded_file is None:
        return False, "Aucun fichier fourni."

    raw = uploaded_file.getvalue()
    file_sig = hashlib.sha256(raw).hexdigest()
    if st.session_state.get("last_indexed_rag_file_sig") == file_sig:
        return True, "Document deja indexe."

    title, content, warning = extract_document_title_and_content(uploaded_file)
    if not content:
        return False, "Impossible d'extraire le contenu du document."

    ok, msg = index_rag_document(
        session=session,
        user_name=user_name,
        doc_name=title,
        content=content,
        source_type="file",
    )
    if ok:
        st.session_state.last_indexed_rag_file_sig = file_sig
        if warning:
            msg = f"{msg} {warning}"
    return ok, msg


def index_rag_document(
    session,
    user_name: str,
    doc_name: str,
    content: str,
    source_type: str = "manual",
    chunk_size: int = 900,
    overlap: int = 150,
) -> Tuple[bool, str]:
    chunks = split_text_into_chunks(content, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return False, "Le document est vide apres nettoyage."

    doc_id = str(uuid4())
    try:
        session.sql(
            f"""
            INSERT INTO {RAG_DOCS_TABLE} (doc_id, user_name, doc_name, source_type, created_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP())
            """,
            params=[doc_id, user_name, doc_name, source_type],
        ).collect()

        for idx, chunk in enumerate(chunks, start=1):
            session.sql(
                f"""
                INSERT INTO {RAG_CHUNKS_TABLE}
                    (chunk_id, doc_id, user_name, chunk_index, chunk_text, embedding, created_at)
                SELECT
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    SNOWFLAKE.CORTEX.EMBED_TEXT_1024(?, ?),
                    CURRENT_TIMESTAMP()
                """,
                params=[str(uuid4()), doc_id, user_name, idx, chunk, RAG_EMBED_MODEL, chunk],
            ).collect()

        return True, f"Document indexe ({len(chunks)} chunks)."
    except Exception as exc:  # noqa: BLE001
        return False, f"Indexation RAG impossible ({exc})."


def retrieve_rag_chunks(
    session, user_name: str, query: str, top_k: int = 3
) -> Tuple[List[Dict[str, str]], str]:
    if not query.strip():
        return [], ""

    try:
        rows = session.sql(
            f"""
            WITH q AS (
                SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_1024(?, ?) AS qvec
            )
            SELECT
                c.chunk_text,
                d.doc_name,
                VECTOR_COSINE_SIMILARITY(c.embedding, q.qvec) AS score
            FROM {RAG_CHUNKS_TABLE} c
            JOIN {RAG_DOCS_TABLE} d ON c.doc_id = d.doc_id
            CROSS JOIN q
            WHERE c.user_name = ?
            ORDER BY score DESC
            LIMIT {int(top_k)}
            """,
            params=[RAG_EMBED_MODEL, query, user_name],
        ).collect()

        results: List[Dict[str, str]] = []
        for row in rows:
            row_data = row.as_dict()
            doc_name = row_data.get("DOC_NAME") or row_data.get("doc_name")
            chunk_text = row_data.get("CHUNK_TEXT") or row_data.get("chunk_text")
            score = row_data.get("SCORE") or row_data.get("score")
            results.append(
                {
                    "doc_name": str(doc_name or ""),
                    "chunk_text": str(chunk_text or ""),
                    "score": f"{float(score):.4f}" if score is not None else "0.0000",
                }
            )

        return results, ""
    except Exception as exc:  # noqa: BLE001
        return [], f"Recherche RAG impossible ({exc})."


def inject_rag_context(history: List[Dict[str, str]], rag_chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not rag_chunks:
        return history

    context_lines = []
    for idx, chunk in enumerate(rag_chunks, start=1):
        context_lines.append(
            f"[Source {idx}] {chunk['doc_name']} (score={chunk['score']}): {chunk['chunk_text']}"
        )

    rag_instruction = (
        "Contexte documentaire RAG:\n"
        + "\n\n".join(context_lines)
        + "\n\nUtilise ce contexte en priorite. Si l'information manque, dis-le clairement."
    )

    if history and history[0]["role"] == "system":
        merged_system = f"{history[0]['content']}\n\n{rag_instruction}"
        return [{"role": "system", "content": merged_system}, *history[1:]]

    return [{"role": "system", "content": rag_instruction}, *history]


def is_empty_model_answer(answer: str) -> bool:
    normalized = " ".join(answer.strip().split())
    if not normalized:
        return True
    if normalized in {"{}", "[]"}:
        return True
    if '"choices": [ {} ]' in normalized:
        return True
    return False


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

    if "rag_warning" not in st.session_state:
        st.session_state.rag_warning = ""

    if "last_rag_sources" not in st.session_state:
        st.session_state.last_rag_sources = []

    if "last_indexed_rag_file_sig" not in st.session_state:
        st.session_state.last_indexed_rag_file_sig = ""


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
if not st.session_state.rag_warning:
    st.session_state.rag_warning = ensure_rag_tables(session)

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

    st.markdown("### Mini-RAG")
    enable_rag = st.checkbox("Activer mini-RAG", value=False)
    rag_top_k = st.slider("Top K RAG", min_value=1, max_value=8, value=3, step=1)
    st.caption("Ajout de document: utilisez l'icone `+` pres de la zone de message.")

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
if st.session_state.rag_warning:
    st.warning(st.session_state.rag_warning)

if len(st.session_state.messages) == 1:
    st.info("Posez votre premiere question dans la zone de saisie en bas.")

for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

input_cols = st.columns([1, 12], vertical_alignment="bottom")
with input_cols[0]:
    with st.popover("➕"):
        rag_inline_file = st.file_uploader(
            "Ajouter un document (txt, md, pdf)",
            type=["txt", "md", "pdf"],
            key="rag_inline_file",
        )
        if rag_inline_file is not None:
            if st.session_state.rag_warning:
                st.error(st.session_state.rag_warning)
            else:
                ok, msg = auto_index_uploaded_rag_file(
                    session=session,
                    user_name=st.session_state.current_user,
                    uploaded_file=rag_inline_file,
                )
                if ok:
                    refresh_user_conversations(session)
                    if msg != "Document deja indexe.":
                        st.success(msg)
                else:
                    st.error(msg)

with input_cols[1]:
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
        payload_without_rag = build_full_history(st.session_state.messages)
        payload = payload_without_rag
        rag_chunks: List[Dict[str, str]] = []
        rag_runtime_warning = ""
        if enable_rag and not st.session_state.rag_warning:
            rag_chunks, rag_runtime_warning = retrieve_rag_chunks(
                session=session,
                user_name=st.session_state.current_user,
                query=user_prompt,
                top_k=rag_top_k,
            )
            if rag_chunks:
                payload = inject_rag_context(payload, rag_chunks)

        with st.spinner("Generation en cours..."):
            try:
                displayed_answer = call_cortex(session, selected_model, payload, temperature)
            except Exception as exc:  # noqa: BLE001
                displayed_answer = format_cortex_exception(exc, selected_model)

        if enable_rag and rag_chunks and is_empty_model_answer(displayed_answer):
            with st.spinner("Repli sans RAG (reponse vide)..."):
                try:
                    displayed_answer = call_cortex(session, selected_model, payload_without_rag, temperature)
                    st.caption("Reponse vide avec RAG: repli automatique sans contexte RAG.")
                except Exception as exc:  # noqa: BLE001
                    displayed_answer = format_cortex_exception(exc, selected_model)

        st.markdown(displayed_answer)
        if rag_runtime_warning:
            st.caption(rag_runtime_warning)
        if enable_rag and rag_chunks:
            with st.expander("Sources RAG utilisees"):
                for idx, chunk in enumerate(rag_chunks, start=1):
                    st.markdown(
                        f"**{idx}. {chunk['doc_name']}** (score={chunk['score']})\n\n{chunk['chunk_text']}"
                    )

        st.session_state.last_rag_sources = rag_chunks

    st.session_state.messages.append({"role": "assistant", "content": displayed_answer})

    if not st.session_state.table_warning:
        try:
            insert_message(
                session, st.session_state.conversation_id, st.session_state.current_user, "assistant", displayed_answer
            )
            refresh_user_conversations(session)
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.session_state.table_warning = f"Insertion assistant impossible ({exc})."
