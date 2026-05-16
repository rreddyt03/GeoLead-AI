import json
import logging
import os
import platform
import re
import smtplib
import sqlite3
import subprocess
import uuid
from pathlib import Path
from email.message import EmailMessage

from flask import Flask, jsonify, render_template, request, session
from automation_engine import decide_action
from chat_reputation import mark_completed, should_block_repeat
from email_orchestrator import should_send_email
from enrichment_service import enrich_ip
from geo_intelligence import enrich_request
from lead_scoring import classify_lead, score_lead
from openai_client import analyze_chat_message, generate_chat_reply
from salesforce import sync_lead_to_salesforce
from safe_logger import log_event
from segmentation import segment_lead


DB_PATH = os.environ.get("DB_PATH", "leads.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.row_factory = sqlite3.Row


def ensure_table_schema():
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            role TEXT,
            message TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            project_intent TEXT,
            lead_data TEXT,
            score INTEGER,
            status TEXT DEFAULT 'new',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    existing_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(leads)").fetchall()
    }

    required_columns = {
        "customer_name": "TEXT",
        "customer_email": "TEXT",
        "customer_phone": "TEXT",
        "conversation_summary": "TEXT",
        "email_sent": "INTEGER DEFAULT 0",
        "salesforce_lead_id": "TEXT",
        "salesforce_sync_status": "TEXT DEFAULT 'pending'",
    }

    for column_name, column_type in required_columns.items():
        if column_name not in existing_columns:
            conn.execute(f"ALTER TABLE leads ADD COLUMN {column_name} {column_type}")

    conn.commit()


ensure_table_schema()


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev")

logging.basicConfig(level=os.environ.get("APP_LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

STYLE_GUIDANCE = {
    "sliding": "Sliding windows are a great fit when you want clean horizontal lines, easy operation, and a more contemporary look.",
    "casement": "Casement windows work well when airflow matters and you want a more tailored, architectural feel.",
    "picture": "Picture windows are ideal when the goal is light, views, and a dramatic focal point.",
    "double-hung": "Double-hung windows are versatile, classic, and easy to blend into traditional homes.",
}

MODEL_GUIDANCE = {
    "100 Series": "Budget-friendly and practical, with a cleaner entry-point price for value-focused projects.",
    "400 Series": "A balanced premium option with strong everyday performance and broad homeowner appeal.",
    "E-Series": "Best for a higher-end look, aluminum-clad durability, and more design-forward projects.",
}

FIELD_PROMPT_VARIATIONS = {
    "project_intent": [
        "Are you replacing existing windows or planning a brand-new installation?",
        "To tailor this right, is this a replacement project or a new installation?",
    ],
    "count": [
        "Nice, how many windows are we working with?",
        "Great start. Roughly how many windows should we plan for?",
    ],
    "size": [
        "Great. What size are the windows? Something like 5x4 or 36 by 48 works perfectly.",
        "Helpful next detail: what window size are you planning? A format like 5x4 works well.",
    ],
    "style": [
        "What style are you leaning toward right now: sliding, casement, picture, or double-hung?",
        "Do you already prefer a style such as sliding, casement, picture, or double-hung?",
    ],
    "material": [
        "Do you have a material preference yet: vinyl, wood, or aluminum?",
        "Which material direction feels best right now: vinyl, wood, or aluminum?",
    ],
    "timeline": [
        "When would you like to move on this project: ASAP, within a few weeks, or within a month?",
        "What timeline are you targeting: ASAP, a few weeks, or around a month?",
    ],
    "budget": [
        "Last planning question: are you aiming for a budget-friendly, balanced, or premium option?",
        "To keep recommendations realistic, should we target budget-friendly, balanced, or premium?",
    ],
}

OBJECTION_PATTERNS = re.compile(
    r"\b(expensive|too much|costly|not now|later|busy|just browsing|don't call|stop|annoying|frustrated|angry)\b"
)

BUSINESS_FROM_EMAIL = os.environ.get("BUSINESS_FROM_EMAIL", "rreddyt08@gmail.com")
PRODUCT_KNOWLEDGE_PATH = Path(__file__).with_name("product_knowledge.json")
PRODUCT_DATA_PATH = Path(__file__).with_name("product_data.json")


def load_json_file(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


PRODUCT_KNOWLEDGE = load_json_file(PRODUCT_KNOWLEDGE_PATH)
PRODUCT_DATA = load_json_file(PRODUCT_DATA_PATH)


def ensure_user_id():
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    return session["user_id"]


def default_project_data():
    return {
        "project_intent": None,
        "count": None,
        "size": None,
        "style": None,
        "material": None,
        "room": None,
        "timeline": None,
        "budget": None,
        "recommended_model": None,
        "lead_captured": False,
        "lead_id": None,
        "customer_name": None,
        "customer_email": None,
        "customer_phone": None,
        "email_sent": False,
    }


def default_context():
    return {
        "intent": None,
        "stage": "idle",
        "data": default_project_data(),
        "last_intent": None,
        "active_flow": None,
        "contact_offer_made": False,
        "awaiting_contact_field": None,
        "asked_fields": {},
        "fallback_count": 0,
        "customer_goal": None,
        "next_best_question": None,
        "geo_data": {},
        "conversation": [],
    }


def merge_missing_defaults(existing, defaults):
    for key, value in defaults.items():
        if key not in existing:
            existing[key] = value
            continue
        if isinstance(value, dict) and isinstance(existing.get(key), dict):
            merge_missing_defaults(existing[key], value)
    return existing


def get_context():
    context = session.get("context")
    if not context:
        context = default_context()
    else:
        context = merge_missing_defaults(context, default_context())
    session["context"] = context
    return context


def add_context_message(context, role, message):
    if not message:
        return
    conversation = context.setdefault("conversation", [])
    conversation.append({"role": role, "message": message})
    if len(conversation) > 20:
        del conversation[:-20]


def build_context_payload(context):
    data = context.get("data", {})
    return {
        "intent": context.get("intent"),
        "window_count": data.get("count"),
        "size": data.get("size"),
        "style": data.get("style"),
        "material": data.get("material"),
        "room": data.get("room"),
        "timeline": data.get("timeline"),
        "customer_name": data.get("customer_name"),
        "customer_email": data.get("customer_email"),
        "customer_phone": data.get("customer_phone"),
        "conversation": context.get("conversation", []),
    }


def build_lead_payload(context):
    data = context.get("data", {})
    summary = build_conversation_summary(data) if data.get("recommended_model") else None
    score = calculate_lead_score(data, context)
    status = lead_status_from_score(score)
    return {
        "name": data.get("customer_name"),
        "email": data.get("customer_email"),
        "phone": data.get("customer_phone"),
        "summary": summary,
        "score": score,
        "status": status,
        "salesforce_sync_status": data.get("salesforce_sync_status"),
        "salesforce_lead_id": data.get("salesforce_lead_id"),
    }


def lead_ready_for_sync(data):
    return bool(
        data.get("recommended_model")
        and data.get("customer_name")
        and data.get("customer_email")
        and data.get("customer_phone")
    )


def ensure_recommended_model(data):
    if data.get("recommended_model"):
        return data["recommended_model"]

    required_project_fields = ["project_intent", "count", "size", "style", "material", "timeline", "budget"]
    if all(data.get(field) for field in required_project_fields):
        data["recommended_model"] = recommend_model(data)
    return data.get("recommended_model")


def get_request_ip(req):
    forwarded_for = req.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return (req.remote_addr or "").strip()


def augment_lead_with_qualification(data, context=None):
    geo_data = (context or {}).get("geo_data") or {}
    city = geo_data.get("city") or data.get("city")
    region = geo_data.get("region") or data.get("region")
    zip_code = geo_data.get("zip") or data.get("zip")
    timeline_value = (data.get("timeline") or "").lower()
    scoring_timeline = "immediate" if timeline_value == "asap" else timeline_value
    analysis = (context or {}).get("message_analysis") or {}

    lead_score = score_lead(
        {
            "city": city,
            "window_count": data.get("count") or 0,
            "timeline": scoring_timeline,
            "email": data.get("customer_email"),
        }
    )
    lead_type = classify_lead(lead_score)
    segment = segment_lead(lead_score, city)
    action = decide_action(lead_type)

    data["city"] = city
    data["region"] = region
    data["zip"] = zip_code
    data["lead_score_v2"] = lead_score
    data["lead_type"] = lead_type
    data["lead_segment"] = segment
    data["automation_action"] = action
    data["recommended_product"] = data.get("recommended_product") or data.get("recommended_model")
    data["confidence_score"] = data.get("confidence_score") or analysis.get("confidence")

    log_event(
        "Lead Qualification",
        {
            "score": lead_score,
            "type": lead_type,
            "segment": segment,
            "action": action,
            "city": city,
            "region": region,
            "zip": zip_code,
        },
    )


def save_message(user_id, role, message):
    conn.execute(
        "INSERT INTO chats (user_id, role, message) VALUES (?, ?, ?)",
        (user_id, role, message),
    )
    conn.commit()


def get_recent_messages(user_id, limit=8):
    rows = conn.execute(
        "SELECT role, message FROM chats WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return [(row["role"], row["message"]) for row in reversed(rows)]


def detect_intent(message, context):
    if context.get("awaiting_contact_field") in {"customer_name", "customer_email", "customer_phone"}:
        return "contact_capture"

    if context.get("contact_offer_made") and re.search(
        r"\b(yes|quote|consult|consultation|call me|contact me|reach me)\b",
        message,
    ):
        return "contact_capture"

    missing_field = next_missing_field(context.get("data", {})) if context.get("data") else None

    if re.search(r"\b(continue|go on|resume)\b", message):
        return "continue"
    if re.search(r"\b(replace|replacement|install|installation|new windows?|window project)\b", message):
        return "project"
    if any(
        phrase in message
        for phrase in [
            "get a quote",
            "need a quote",
            "want a quote",
            "send quote",
            "schedule consultation",
            "book consultation",
            "call me",
            "contact me",
        ]
    ):
        return "contact_capture"
    if re.search(r"\b(what kind|what types|which types|available windows|window options|products do you have|what do you offer)\b", message):
        return "style_query"
    if any(word in message for word in ["style", "design", "look"]) or message.startswith("show styles"):
        return "style_query"
    if any(word in message for word in ["model", "series", "e-series", "100 series", "400 series"]):
        return "model_query"
    if any(word in message for word in ["price", "pricing", "cost"]):
        return "pricing"
    if any(word in message for word in ["email", "mail", "send this", "send summary", "send quote", "consultation", "consult", "call me", "contact me"]):
        return "contact_capture"
    if OBJECTION_PATTERNS.search(message):
        return "objection"
    if "help" in message:
        return "help"

    if context["active_flow"] == "project":
        if any(
            [
                extract_count(
                    message,
                    allow_loose=missing_field == "count" or is_correction_message(message),
                ),
                extract_size(message),
                extract_style(message),
                extract_material(message),
                extract_timeline(message),
                extract_budget(message),
                is_style_recommendation_question(message, missing_field),
            ]
        ):
            return "project"

    return "general"


def normalize_ai_analysis(analysis):
    if not isinstance(analysis, dict):
        return {}

    normalized = dict(analysis)
    for key in [
        "intent",
        "project_intent",
        "style",
        "material",
        "timeline",
        "budget",
        "room",
        "accepted_style",
        "accepted_material",
    ]:
        value = normalized.get(key)
        if isinstance(value, str):
            normalized[key] = value.strip().lower()

    size_value = normalized.get("size")
    if isinstance(size_value, str):
        size_match = re.search(r"(\d+)\s*[xX]\s*(\d+)", size_value)
        if size_match:
            normalized["size"] = f"{size_match.group(1)}x{size_match.group(2)}"

    count_value = normalized.get("count")
    if isinstance(count_value, str) and count_value.isdigit():
        normalized["count"] = int(count_value)

    return normalized


def merge_ai_analysis(context, analysis):
    analysis = normalize_ai_analysis(analysis)
    if not analysis:
        return {}

    data = context["data"]

    if analysis.get("project_intent") in {"replace", "install"}:
        data["project_intent"] = analysis["project_intent"]
        context["active_flow"] = "project"
        context["last_intent"] = "project"

    if isinstance(analysis.get("count"), int) and analysis["count"] > 0:
        data["count"] = analysis["count"]
    if analysis.get("size"):
        data["size"] = analysis["size"]
    if analysis.get("room"):
        data["room"] = analysis["room"]
    if analysis.get("budget") in {"low", "medium", "high"}:
        data["budget"] = analysis["budget"]
    if analysis.get("timeline"):
        data["timeline"] = analysis["timeline"]

    chosen_style = analysis.get("accepted_style") or analysis.get("style")
    if chosen_style in {"sliding", "casement", "picture", "double-hung"}:
        data["style"] = chosen_style
        context["pending_style_recommendation"] = None

    chosen_material = analysis.get("accepted_material") or analysis.get("material")
    if chosen_material in {"vinyl", "wood", "fiberglass", "aluminum"}:
        data["material"] = chosen_material
        context["pending_material_recommendation"] = None

    if analysis.get("wants_recommendation"):
        context["active_flow"] = "project"
    if analysis.get("wants_quote"):
        context["contact_offer_made"] = True

    if analysis.get("customer_goal"):
        context["customer_goal"] = analysis["customer_goal"]
    if analysis.get("next_best_question"):
        context["next_best_question"] = analysis["next_best_question"]

    context["message_analysis"] = analysis
    return analysis


def smart_project_follow_up(context, missing_field=None):
    data = context.get("data", {})
    next_best_question = context.get("next_best_question")
    if next_best_question:
        return next_best_question

    room = data.get("room")
    customer_goal = context.get("customer_goal") or ""

    if missing_field == "size":
        if room:
            return f"Do you know the rough size for those {room} windows? A format like 8x3 works perfectly."
        return ask_for_field("size", context)

    if missing_field == "style":
        if room == "living room" and data.get("material") == "fiberglass":
            return "For your living room, would you rather prioritize a more open view or stronger everyday ventilation? That will tell me whether to lean picture or casement."
        if "premium" in customer_goal:
            return "Are you leaning more toward a sleek modern look or maximum glass and light? That will help me narrow the best style quickly."
        return ask_for_field("style", context)

    if missing_field == "material":
        if "premium" in customer_goal:
            return "Is the premium feel more important to you, or would you rather balance look with budget? That will help me narrow the right material."
        return ask_for_field("material", context)

    if missing_field == "timeline":
        return "Are you trying to move on this soon, in a few weeks, or just planning ahead right now?"

    if missing_field == "budget":
        return "Should I keep this closer to budget-friendly, balanced, or premium?"

    return ask_for_field(missing_field, context) if missing_field else "Tell me a bit more about the project and I’ll narrow it down."


def extract_count(message, allow_loose=False):
    count_match = re.search(r"\b(\d+)\s*(window|windows)\b", message)
    if count_match:
        return int(count_match.group(1))

    for word, number in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\s+(window|windows)\b", message):
            return number

    if allow_loose:
        cleaned = re.sub(r"^(ok|okay|yes|yeah|yep|sure|about|around)\s+", "", message.strip(), flags=re.IGNORECASE)
        if re.fullmatch(r"\d+", cleaned):
            return int(cleaned)
        for word, number in NUMBER_WORDS.items():
            if re.fullmatch(rf"{word}", cleaned):
                return number
    return None


def extract_size(message):
    patterns = [
        r"(\d+)\s*[xX*]\s*(\d+)",
        r"(\d+)\s*by\s*(\d+)",
        r"(\d+)\s*feet?\s*by\s*(\d+)\s*feet?",
    ]
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            return f"{match.group(1)}x{match.group(2)}"
    return None


def extract_budget(message):
    if any(term in message for term in ["budget", "affordable", "low cost", "low-budget", "cheap"]):
        return "low"
    if any(term in message for term in ["mid", "balanced", "moderate"]):
        return "medium"
    if any(term in message for term in ["premium", "high-end", "luxury", "best"]):
        return "high"
    return None


def extract_style(message):
    if "sliding" in message:
        return "sliding"
    if "casement" in message:
        return "casement"
    if "picture" in message:
        return "picture"
    if "double-hung" in message or "double hung" in message:
        return "double-hung"
    return None


def extract_material(message):
    if "vinyl" in message:
        return "vinyl"
    if "wood" in message:
        return "wood"
    if "fiberglass" in message:
        return "fiberglass"
    if "aluminum" in message:
        return "aluminum"
    return None


def extract_timeline(message):
    if "asap" in message or "soon as possible" in message:
        return "ASAP"
    if re.search(r"\b(a|1)\s+week\b", message):
        return "1 week"
    if re.search(r"\b(\d+)\s+weeks\b", message):
        return re.search(r"\b(\d+)\s+weeks\b", message).group(1) + " weeks"
    if "week" in message:
        return "2 weeks"
    if "month" in message:
        return "1 month"
    if "this year" in message:
        return "This year"
    return None


def is_affirmative(message):
    return bool(re.search(r"\b(yes|yeah|yep|sure|okay|ok|send it|please do)\b", message))


def is_valid_email(email):
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", (email or "").strip()))


def looks_like_email(message):
    return is_valid_email(message)


def extract_email_from_text(message):
    match = re.search(r"([^\s,;]+@[^\s,;]+\.[^\s,;]+)", message)
    if match:
        return match.group(1).strip()
    return None


def normalize_phone_number(message):
    digits = re.sub(r"\D", "", message or "")
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return None
    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"


def extract_phone_from_text(message):
    match = re.search(r"(\+?\d[\d\s().-]{8,}\d)", message)
    if not match:
        return None
    return normalize_phone_number(match.group(1))


def looks_like_phone(message):
    return bool(normalize_phone_number(message))


def extract_name_from_text(message):
    cleaned = message.strip()
    cleaned = re.sub(r"^(my name is|i am|i'm|this is)\s+", "", cleaned, flags=re.IGNORECASE)
    email = extract_email_from_text(cleaned)
    if email:
        cleaned = cleaned.replace(email, " ")
    cleaned = re.sub(r"[^A-Za-z\s'-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
    if len(cleaned) < 2:
        return None
    return cleaned.title()


def has_name_intro(message):
    return bool(re.search(r"^(my name is|i am|i'm|this is)\b", message.strip(), flags=re.IGNORECASE))


def looks_like_name_input(message):
    cleaned = (message or "").strip()
    if not cleaned:
        return False
    if extract_email_from_text(cleaned) or extract_phone_from_text(cleaned):
        return False
    if re.search(
        r"\b(quote|consult|consultation|window|windows|replace|install|need|want|get|send|schedule|book|call|contact|yes|please)\b",
        cleaned,
        flags=re.IGNORECASE,
    ):
        return False
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", cleaned)
    return 1 < len(tokens) <= 4 and len(" ".join(tokens)) >= 4


def should_capture_name_from_intro(message):
    if not has_name_intro(message):
        return False
    return looks_like_name_input(message)


def is_correction_message(message):
    return bool(re.search(r"\b(actually|instead|change|update|make that|correction)\b", message))


def is_style_recommendation_question(message, missing_field):
    if missing_field != "style":
        return False
    return bool(
        re.search(
            r"\b(which|what)\b.*\b(prefer|recommend|best|better|suggest)\b|\bwhat would you recommend\b|\bwhich one\b",
            message,
        )
    )


def is_material_recommendation_question(message, missing_field):
    if missing_field != "material":
        return False
    return bool(
        re.search(
            r"\b(which|what)\b.*\b(material|option|prefer|recommend|best|better|suggest)\b|\bwhat would you recommend\b|\bwhich one\b",
            message,
        )
    )


def infer_style_from_confirmation(message, context):
    if not is_affirmative(message) and not re.search(r"\b(go with|that one|that style|that works|i'll take that|i will go with)\b", message):
        return None

    recent_style = extract_recommended_style_from_reply(context.get("last_assistant_reply"))
    if recent_style:
        return recent_style

    pending_style = context.get("pending_style_recommendation")
    if pending_style:
        return pending_style
    return None


def infer_material_from_confirmation(message, context):
    if not is_affirmative(message) and not re.search(r"\b(go with|that one|that material|that works|i'll take that|i will go with)\b", message):
        return None

    recent_material = extract_recommended_material_from_reply(context.get("last_assistant_reply"))
    if recent_material:
        return recent_material

    pending_material = context.get("pending_material_recommendation")
    if pending_material:
        return pending_material
    return None


def extract_recommended_style_from_reply(reply_text):
    if not reply_text:
        return None
    lowered = reply_text.lower()
    style_patterns = [
        ("casement", [r"recommend casement", r"lean toward casement", r"go with casement"]),
        ("sliding", [r"recommend sliding", r"lean toward sliding", r"go with sliding"]),
        ("picture", [r"recommend picture", r"lean toward picture", r"go with picture"]),
        ("double-hung", [r"recommend double-hung", r"recommend double hung", r"lean toward double-hung", r"lean toward double hung"]),
    ]
    for style, patterns in style_patterns:
        if any(re.search(pattern, lowered) for pattern in patterns):
            return style
    return None


def extract_recommended_material_from_reply(reply_text):
    if not reply_text:
        return None
    lowered = reply_text.lower()
    material_patterns = [
        ("fiberglass", [r"recommend fiberglass", r"lean toward fiberglass", r"go with fiberglass"]),
        ("vinyl", [r"recommend vinyl", r"lean toward vinyl", r"go with vinyl"]),
        ("wood", [r"recommend wood", r"lean toward wood", r"go with wood"]),
        ("aluminum", [r"recommend aluminum", r"lean toward aluminum", r"go with aluminum"]),
    ]
    for material, patterns in material_patterns:
        if any(re.search(pattern, lowered) for pattern in patterns):
            return material
    return None


def is_domain_follow_up_question(message):
    return bool(
        re.search(
            r"\b(competitor|competitors|compare|comparison|vs\.?|versus|warranty|energy|efficient|durability|durable|material|materials|style|styles|model|models|door|doors|window|windows|price|pricing|cost)\b",
            message,
        )
    )


def recommend_style_for_context(message, data):
    size = data.get("size")
    room = data.get("room") or ""
    if "living room" in message or room == "living room":
        budget = data.get("budget")
        material = data.get("material")
        if budget == "high" or material == "fiberglass":
            return (
                "For a living room with a more premium feel, I’d usually lean toward casement. "
                "It gives you cleaner sightlines, a more upscale look, and strong ventilation. "
                "If that direction works for you, say casement and I’ll lock it in."
            )
        return (
            "For a living room, I’d usually lean toward picture if the goal is light and a focal point, "
            "or sliding if you want a cleaner modern everyday option. "
            "If you want my default everyday recommendation, I’d go with sliding. "
            "If one fits, tell me and I’ll lock it in."
        )
    if "bedroom" in message or room == "bedroom":
        if size:
            return (
                f"For a bedroom with a {size} window size, I’d usually lean toward sliding if you want a clean look and easy operation, "
                "or casement if airflow matters more. My default recommendation for most bedrooms is sliding because it feels simple and practical. "
                "If that works for you, say sliding and I’ll lock it in."
            )
        return (
            "For a bedroom, I’d usually lean toward sliding for the cleaner look and easy everyday use, "
            "or casement if ventilation is your top priority. My default recommendation would be sliding. "
            "If you want, say sliding and I’ll use that."
        )

    return (
        "If you want my recommendation, I’d usually lean toward sliding for a modern, practical everyday choice, "
        "and casement if airflow is the bigger priority. If one of those feels right, tell me and I’ll factor it in."
    )


def recommended_style_value(data):
    room = data.get("room")
    if room == "living room" and (data.get("budget") == "high" or data.get("material") == "fiberglass"):
        return "casement"
    if room == "bedroom":
        return "sliding"
    if data.get("budget") == "high":
        return "casement"
    return "sliding"


def update_project_data(data, message, missing_field=None):
    if "replace" in message:
        data["project_intent"] = "replace"
    elif "install" in message or "installation" in message:
        data["project_intent"] = "install"

    extracted_count = extract_count(
        message,
        allow_loose=missing_field == "count" or is_correction_message(message),
    )
    if extracted_count:
        data["count"] = extracted_count

    extracted_size = extract_size(message)
    if extracted_size:
        data["size"] = extracted_size

    extracted_style = extract_style(message)
    if extracted_style:
        data["style"] = extracted_style

    extracted_material = extract_material(message)
    if extracted_material:
        data["material"] = extracted_material

    extracted_timeline = extract_timeline(message)
    if extracted_timeline:
        data["timeline"] = extracted_timeline

    extracted_budget = extract_budget(message)
    if extracted_budget:
        data["budget"] = extracted_budget

    return data


def update_contact_data(data, message):
    extracted_email = extract_email_from_text(message)
    extracted_phone = extract_phone_from_text(message)
    extracted_name = extract_name_from_text(message) if should_capture_name_from_intro(message) else None

    if extracted_email and not data.get("customer_email"):
        data["customer_email"] = extracted_email
    if extracted_phone and not data.get("customer_phone"):
        data["customer_phone"] = extracted_phone
    if extracted_name and not data.get("customer_name"):
        data["customer_name"] = extracted_name
    return data


def next_missing_field(data):
    ordered_fields = [
        "project_intent",
        "count",
        "size",
        "style",
        "material",
        "timeline",
        "budget",
    ]
    for field in ordered_fields:
        if not data.get(field):
            return field
    return None


def ask_for_field(field, context=None):
    options = FIELD_PROMPT_VARIATIONS.get(field) or [f"Could you share {field}?"]
    if not context:
        return options[0]

    asked_fields = context.setdefault("asked_fields", {})
    asked_count = asked_fields.get(field, 0)
    asked_fields[field] = asked_count + 1
    return options[asked_count % len(options)]


def recommend_model(data):
    if data["material"] == "aluminum" or data["budget"] == "high":
        return "E-Series"
    if data["budget"] == "low":
        return "100 Series"
    if data["style"] in {"picture", "casement"} or data["material"] == "wood":
        return "400 Series"
    return "400 Series"


def calculate_lead_score(data, context=None):
    score = 1
    engagement = len((context or {}).get("conversation", []))

    if data.get("timeline") == "ASAP":
        score += 3
    elif data.get("timeline") in {"1 week", "2 weeks"}:
        score += 2
    elif data.get("timeline"):
        score += 1

    filled_fields = sum(
        1
        for field in ["project_intent", "count", "size", "style", "material", "timeline", "budget"]
        if data.get(field)
    )
    if filled_fields >= 6:
        score += 2
    elif filled_fields >= 4:
        score += 1

    if data.get("customer_name"):
        score += 1
    if data.get("customer_email"):
        score += 1
    if data.get("customer_phone"):
        score += 1

    if data.get("count") and data["count"] >= 3:
        score += 1

    if engagement >= 10:
        score += 1

    return max(1, min(score, 10))


def lead_status_from_score(score):
    if score >= 8:
        return "Hot"
    if score >= 5:
        return "Warm"
    return "Cold"


def build_conversation_transcript(context):
    lines = []
    for turn in context.get("conversation", []):
        role = "User" if turn.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {turn.get('message')}")
    return "\n".join(lines)


def build_conversation_summary(data):
    recommended_model = data.get("recommended_model") or recommend_model(data)
    model_reason = MODEL_GUIDANCE.get(recommended_model, "Chosen based on your current preferences.")

    return (
        "Addison Windows Consultation Summary\n\n"
        f"Name: {data.get('customer_name') or 'Not provided'}\n"
        f"Email: {data.get('customer_email') or 'Not provided'}\n"
        f"Phone: {data.get('customer_phone') or 'Not provided'}\n"
        f"Project Type: {data.get('project_intent')}\n"
        f"Window Count: {data.get('count')}\n"
        f"Size: {data.get('size')}\n"
        f"Room: {data.get('room')}\n"
        f"Style Preference: {data.get('style')}\n"
        f"Material Preference: {data.get('material')}\n"
        f"Timeline: {data.get('timeline')}\n"
        f"Budget Direction: {data.get('budget')}\n"
        f"Recommended Model: {recommended_model}\n\n"
        f"Recommendation Reason: {model_reason}\n\n"
        "This summary can be used as a reference for the next consultation or quote step."
    )


def build_email_summary(data, context=None):
    summary = build_conversation_summary(data)
    conversation_text = build_conversation_transcript(context or {})
    if conversation_text:
        summary += "\n\nFULL CONVERSATION\n-----------------\n"
        summary += conversation_text
    return summary


def generate_recommendation_response(data, include_contact_prompt=True):
    recommended_model = data.get("recommended_model") or recommend_model(data)
    model_reason = MODEL_GUIDANCE.get(recommended_model, "Chosen based on your current preferences.")

    reply = (
        "Perfect. I’ve got enough to make a recommendation.\n\n"
        f"- Project: {data.get('project_intent')}\n"
        f"- Window count: {data.get('count')}\n"
        f"- Size: {data.get('size')}\n"
        f"- Style: {data.get('style')}\n"
        f"- Material: {data.get('material')}\n"
        f"- Timeline: {data.get('timeline')}\n"
        f"- Budget direction: {data.get('budget')}\n\n"
        f"My best fit right now is {recommended_model}. {model_reason}"
    )

    if include_contact_prompt:
        reply += "\n\nWould you like me to schedule a free consultation or help you get a quote?"

    return reply


def save_lead(user_id, data, context):
    ensure_recommended_model(data)
    augment_lead_with_qualification(data, context)
    score = calculate_lead_score(data, context)
    summary = build_conversation_summary(data)
    conversation_text = build_conversation_transcript(context)
    lead_status = lead_status_from_score(score)
    recommendation_reason = MODEL_GUIDANCE.get(
        data.get("recommended_model") or recommend_model(data),
        "Chosen based on your current preferences.",
    )
    data["recommendation_reason"] = recommendation_reason
    if lead_ready_for_sync(data):
        logger.info("Syncing new lead to Salesforce for %s", data.get("customer_email"))
        sf_lead_id, sf_status = sync_lead_to_salesforce(
            data,
            recommendation_reason,
            summary,
            score,
            lead_status,
            conversation_text,
        )
    else:
        sf_lead_id, sf_status = data.get("salesforce_lead_id"), "pending_contact"
    data["salesforce_lead_id"] = sf_lead_id
    data["salesforce_sync_status"] = sf_status
    logger.info("Lead save sync status=%s lead_id=%s", sf_status, sf_lead_id)
    cursor = conn.execute(
        """
        INSERT INTO leads (
            user_id,
            project_intent,
            lead_data,
            score,
            status,
            customer_name,
            customer_email,
            customer_phone,
            conversation_summary,
            email_sent,
            salesforce_lead_id,
            salesforce_sync_status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            data.get("project_intent"),
            json.dumps(data),
            score,
            lead_status,
            data.get("customer_name"),
            data.get("customer_email"),
            data.get("customer_phone"),
            summary,
            1 if data.get("email_sent") else 0,
            data.get("salesforce_lead_id"),
            data.get("salesforce_sync_status"),
        ),
    )
    conn.commit()
    return cursor.lastrowid, score


def update_lead_record(data, context=None):
    lead_id = data.get("lead_id")
    if not lead_id:
        return

    ensure_recommended_model(data)
    augment_lead_with_qualification(data, context)
    summary = build_conversation_summary(data)
    score = calculate_lead_score(data, context or {})
    lead_status = lead_status_from_score(score)
    conversation_text = build_conversation_transcript(context or {"conversation": []})
    recommendation_reason = MODEL_GUIDANCE.get(
        data.get("recommended_model") or recommend_model(data),
        "Chosen based on your current preferences.",
    )
    data["recommendation_reason"] = recommendation_reason
    if lead_ready_for_sync(data):
        logger.info("Updating Salesforce lead for %s", data.get("customer_email"))
        sf_lead_id, sf_status = sync_lead_to_salesforce(
            data,
            recommendation_reason,
            summary,
            score,
            lead_status,
            conversation_text,
        )
    else:
        sf_lead_id, sf_status = data.get("salesforce_lead_id"), "pending_contact"
    data["salesforce_lead_id"] = sf_lead_id
    data["salesforce_sync_status"] = sf_status
    logger.info("Lead update sync status=%s lead_id=%s", sf_status, sf_lead_id)
    conn.execute(
        """
        UPDATE leads
        SET lead_data = ?,
            customer_name = ?,
            customer_email = ?,
            customer_phone = ?,
            conversation_summary = ?,
            email_sent = ?,
            status = ?,
            salesforce_lead_id = ?,
            salesforce_sync_status = ?
        WHERE id = ?
        """,
        (
            json.dumps(data),
            data.get("customer_name"),
            data.get("customer_email"),
            data.get("customer_phone"),
            summary,
            1 if data.get("email_sent") else 0,
            lead_status,
            data.get("salesforce_lead_id"),
            data.get("salesforce_sync_status"),
            lead_id,
        ),
    )
    conn.commit()


def send_summary_email(recipient_name, recipient_email, data, context=None):
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    smtp_sender = os.environ.get("SMTP_SENDER", BUSINESS_FROM_EMAIL or smtp_user or "noreply@example.com")
    smtp_use_tls = os.environ.get("SMTP_USE_TLS", "true").lower() != "false"

    summary = build_email_summary(data, context)

    if not smtp_host or not smtp_user or not smtp_password:
        if platform.system() == "Darwin":
            mail_sent, mail_reply = send_summary_via_mail_app(recipient_name, recipient_email, data, context)
            if mail_sent:
                return True, mail_reply
        return False, (
            f"I’ve captured your details, {recipient_name}, and prepared the consultation summary. "
            "Email delivery isn’t configured yet on this server, so I saved everything for Addison Windows follow-up."
        )

    message = EmailMessage()
    message["Subject"] = "Your Addison Windows Consultation Summary"
    message["From"] = smtp_sender
    message["To"] = recipient_email
    message.set_content(
        f"Hi {recipient_name},\n\n"
        "Thanks for speaking with Addison Windows. Here is your conversation summary for future reference:\n\n"
        f"{summary}\n\n"
        "If you would like a formal quote or consultation next, reply to this email or continue the conversation with our team."
    )

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
            if smtp_use_tls:
                server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(message)
    except Exception:
        return False, (
            f"I captured your details, {recipient_name}, but the email could not be sent just yet. "
            "Your consultation summary is still saved internally for the Addison Windows team to follow up."
        )

    return True, (
        f"Perfect. I’ve sent your consultation summary to {recipient_email}, and the Addison Windows team can use your details for quote or consultation follow-up."
    )


def escape_applescript_string(value):
    return value.replace("\\", "\\\\").replace('"', '\\"')


def send_summary_via_mail_app(recipient_name, recipient_email, data, context=None):
    summary = build_email_summary(data, context)
    subject = "Your Addison Windows Consultation Summary"
    body = (
        f"Hi {recipient_name},\n\n"
        "Thanks for speaking with Addison Windows. Here is your conversation summary for future reference:\n\n"
        f"{summary}\n\n"
        "If you would like a formal quote or consultation next, reply to this email or continue the conversation with our team."
    )

    applescript = f'''
    tell application "Mail"
        set newMessage to make new outgoing message with properties {{subject:"{escape_applescript_string(subject)}", content:"{escape_applescript_string(body)}", visible:false}}
        tell newMessage
            set sender to "{escape_applescript_string(BUSINESS_FROM_EMAIL)}"
            make new to recipient at end of to recipients with properties {{address:"{escape_applescript_string(recipient_email)}"}}
            send
        end tell
    end tell
    '''

    try:
        subprocess.run(
            ["osascript", "-e", applescript],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False, (
            f"I captured your details, {recipient_name}, but the Mail app could not send the summary yet. "
            "Your consultation summary is still saved internally for the Addison Windows team to follow up."
        )

    return True, (
        f"Perfect. I’ve sent your consultation summary to {recipient_email}, and the Addison Windows team can use your details for quote or consultation follow-up."
    )


def complete_contact_capture(context, data):
    mark_completed(context, data)
    context["awaiting_contact_field"] = None

    if should_send_email(data):
        logger.info("Sending consultation summary email to %s", data.get("customer_email"))
        sent, message = send_summary_email(
            data.get("customer_name") or "there",
            data.get("customer_email"),
            data,
            context,
        )
        data["email_sent"] = sent
        update_lead_record(data, context)
        if sent:
            return "Great — everything is set. Our team will contact you shortly. If you need anything else, feel free to ask."
        return message

    update_lead_record(data, context)
    return "Great — everything is set. Our team will contact you shortly. If you need anything else, feel free to ask."


def start_contact_capture(context):
    context["intent"] = "lead_capture"
    context["contact_offer_made"] = True
    context["awaiting_contact_field"] = "customer_name"
    context["stage"] = "contact_collection"
    return "Absolutely. I can help with that. What’s your name?"


def handle_contact_capture(context, message):
    context["intent"] = "lead_capture"
    data = context["data"]
    ensure_recommended_model(data)
    awaiting = context.get("awaiting_contact_field")
    extracted_email = extract_email_from_text(message)
    extracted_phone = extract_phone_from_text(message)
    extracted_name = extract_name_from_text(message)
    explicit_name = extract_name_from_text(message) if has_name_intro(message) else None
    mark_completed(context, data)

    if not data.get("recommended_model"):
        return "I can absolutely help with that once I understand the project a bit better. Tell me about your windows or doors, and I’ll guide you from there."

    if extracted_email and context.get("stage") in {"contact_complete", "completed"}:
        data["customer_email"] = extracted_email
        if not data.get("customer_name"):
            context["awaiting_contact_field"] = "customer_name"
            context["stage"] = "contact_collection"
            update_lead_record(data, context)
            return "I’ve captured the email. What name should I use for the consultation request?"
        if not data.get("customer_phone"):
            context["awaiting_contact_field"] = "customer_phone"
            context["stage"] = "contact_collection"
            update_lead_record(data, context)
            return "I’ve got the email. What phone number should our team use for follow-up?"
        return complete_contact_capture(context, data)

    if not awaiting:
        if (explicit_name or (looks_like_name_input(message) and extracted_name)) and not data.get("customer_name"):
            data["customer_name"] = explicit_name or extracted_name
            update_lead_record(data, context)
        if extracted_email and not data.get("customer_email"):
            data["customer_email"] = extracted_email
        if extracted_phone and not data.get("customer_phone"):
            data["customer_phone"] = extracted_phone

        if not data.get("customer_name"):
            context["awaiting_contact_field"] = "customer_name"
            context["stage"] = "contact_collection"
            return "Absolutely. What’s your name?"
        if not data.get("customer_email"):
            context["awaiting_contact_field"] = "customer_email"
            context["stage"] = "contact_collection"
            return f"Great, {data['customer_name']}. What email should I use for the quote or consultation details?"
        if not data.get("customer_phone"):
            context["awaiting_contact_field"] = "customer_phone"
            context["stage"] = "contact_collection"
            update_lead_record(data, context)
            return "And what phone number should our team use to reach you?"
        if data.get("customer_name") and data.get("customer_email") and data.get("customer_phone"):
            return complete_contact_capture(context, data)

    if awaiting == "customer_name":
        if extracted_email and not extracted_name:
            return "I’ve got the email, but I still need your name before I can set up the consultation request. What name should I use?"

        cleaned_name = extracted_name
        if not cleaned_name:
            return "I want to make sure I get your name right. Please type it the way you’d like our team to use it."

        data["customer_name"] = cleaned_name
        if extracted_email and looks_like_email(extracted_email):
            data["customer_email"] = extracted_email
        if extracted_phone and looks_like_phone(extracted_phone):
            data["customer_phone"] = extracted_phone

        if not data.get("customer_email"):
            context["awaiting_contact_field"] = "customer_email"
            context["stage"] = "contact_collection"
            update_lead_record(data, context)
            return f"Nice to meet you, {cleaned_name}. What email should I use for the quote or consultation details?"

        if not data.get("customer_phone"):
            context["awaiting_contact_field"] = "customer_phone"
            context["stage"] = "contact_collection"
            update_lead_record(data, context)
            return f"Thanks, {cleaned_name}. What phone number should our team use for follow-up?"

        if data.get("customer_email") and data.get("customer_phone"):
            return complete_contact_capture(context, data)

    if awaiting == "customer_email":
        email_value = extracted_email or message.strip()
        if not is_valid_email(email_value):
            return "Please enter a valid email address."

        data["customer_email"] = email_value
        if data.get("customer_phone"):
            return complete_contact_capture(context, data)

        context["awaiting_contact_field"] = "customer_phone"
        context["stage"] = "contact_collection"
        update_lead_record(data, context)
        return "Perfect. What phone number should our team use for follow-up?"

    if awaiting == "customer_phone":
        phone_value = extracted_phone or normalize_phone_number(message)
        if not phone_value:
            return "That phone number looks a little off. Please enter a valid number with area code."

        data["customer_phone"] = phone_value
        return complete_contact_capture(context, data)

    return "I can help with that. Start by sharing your name, and I’ll take it one step at a time."


def handle_project_intent(context, message, user_id):
    context["intent"] = "project"
    context["active_flow"] = "project"
    context["last_intent"] = "project"
    context["stage"] = "collecting"

    missing_before_update = next_missing_field(context["data"])
    data = update_project_data(context["data"], message, missing_before_update)
    inferred_style = infer_style_from_confirmation(message, context)
    if inferred_style and not data.get("style"):
        data["style"] = inferred_style
    inferred_material = infer_material_from_confirmation(message, context)
    if inferred_material and not data.get("material"):
        data["material"] = inferred_material
    data = update_contact_data(data, message)
    mark_completed(context, data)
    missing = next_missing_field(data)
    analysis = context.get("message_analysis") or {}

    if missing_before_update == "style" and (
        is_style_recommendation_question(message, missing_before_update)
        or analysis.get("wants_recommendation")
    ):
        context["data"] = data
        context["pending_style_recommendation"] = recommended_style_value(data)
        return recommend_style_for_context(message, data)

    if missing_before_update == "material" and (
        is_material_recommendation_question(message, missing_before_update)
        or analysis.get("wants_recommendation")
    ):
        context["data"] = data
        context["pending_material_recommendation"] = "fiberglass" if data.get("budget") == "high" else "vinyl"
        if data.get("budget") == "high":
            return "For a more premium result, I’d lean toward fiberglass. It gives you a cleaner upscale finish, strong durability, and good long-term performance. If that works, say fiberglass and I’ll lock it in."
        return "For most projects, I’d usually lean toward vinyl if you want value and low maintenance, or fiberglass if you want a more premium finish. If one feels right, tell me and I’ll lock it in."

    if missing:
        context["data"] = data
        return smart_project_follow_up(context, missing)

    recommended_model = recommend_model(data)
    data["recommended_model"] = recommended_model
    context["data"] = data
    context["stage"] = "completed"
    context["pending_style_recommendation"] = None
    context["pending_material_recommendation"] = None

    if not data.get("lead_captured"):
        lead_id, _score = save_lead(user_id, data, context)
        data["lead_captured"] = True
        data["lead_id"] = lead_id
    else:
        update_lead_record(data, context)

    if is_correction_message(message):
        reply = (
            "Got it — I updated your project details.\n\n"
            + generate_recommendation_response(data, include_contact_prompt=not data.get("customer_email"))
        )
    else:
        reply = generate_recommendation_response(data, include_contact_prompt=True)

    context["contact_offer_made"] = True
    return reply


def handle_continue(context, user_id):
    context["intent"] = "continue"
    if context.get("awaiting_contact_field"):
        if context["awaiting_contact_field"] == "customer_name":
            return "Sure. Let’s finish the follow-up details. What’s your name?"
        if context["awaiting_contact_field"] == "customer_email":
            return "Sure. What email should I use for your quote or consultation follow-up?"
        return "Sure. What phone number should our team use to reach you?"

    if context["active_flow"] == "project":
        missing = next_missing_field(context["data"])
        if missing:
            context["last_intent"] = "project"
            return "Absolutely. Let’s pick this back up. " + smart_project_follow_up(context, missing)
        if context["contact_offer_made"] and not context["data"].get("customer_email"):
            return start_contact_capture(context)
        return handle_project_intent(context, "", user_id)

    return "We can keep going. Tell me if you want help with your project, styles, materials, pricing, or quote follow-up."


def handle_style_query(context):
    context["intent"] = "ask_about_products"
    context["last_intent"] = "style_query"
    style = context["data"].get("style")
    known_styles = PRODUCT_KNOWLEDGE.get("types") or PRODUCT_DATA.get("types") or []

    if style and style in STYLE_GUIDANCE:
        reply = (
            f"Based on what you’ve shared, {STYLE_GUIDANCE[style]} "
            "If you want, say continue and I’ll resume the project flow where we left off."
        )
        context["last_assistant_reply"] = reply
        return reply

    reply = (
        "We offer several popular window types, including:\n\n"
        "- Sliding windows\n"
        "- Casement windows\n"
        "- Double-hung windows\n"
        "- Picture windows\n"
        "- Bay and bow windows\n\n"
        "Each works a little differently depending on airflow, aesthetics, and how much glass you want. "
        "If you're looking for something specific for your home, tell me the room or the look you want and I’ll narrow it down."
    )
    if known_styles:
        reply += "\n\nAvailable product terms in the current knowledge base include: " + ", ".join(known_styles[:8]) + "."
    context["last_assistant_reply"] = reply
    return reply


def handle_model_query(context):
    context["intent"] = "ask_about_products"
    context["last_intent"] = "model_query"
    current_recommendation = context["data"].get("recommended_model")
    known_series = PRODUCT_KNOWLEDGE.get("series") or []

    if current_recommendation:
        return (
            f"Right now your strongest match is {current_recommendation}. "
            f"{MODEL_GUIDANCE[current_recommendation]} "
            "If you want a comparison against the other series, I can do that too."
        )

    reply = (
        "The main recommendation tiers I’m using are:\n\n"
        "- 100 Series: more budget-conscious.\n"
        "- 400 Series: balanced premium choice.\n"
        "- E-Series: best for high-end design and aluminum-driven durability.\n\n"
        "If you continue the intake, I’ll match one of these to your project instead of giving a generic answer."
    )
    if known_series:
        reply += "\n\nKnown series from your product guides include: " + ", ".join(known_series[:8]) + "."
    return reply


def handle_pricing(context):
    context["intent"] = "ask_about_pricing"
    context["last_intent"] = "pricing"
    model = context["data"].get("recommended_model")

    if model:
        return (
            f"For planning purposes, {model} sits in the "
            f"{'premium' if model == 'E-Series' else 'mid-to-premium' if model == '400 Series' else 'entry-friendly'} range. "
            "Exact pricing depends on count, size, installation complexity, and finish choices. "
            "If you want, I can also email your summary once we capture your name and email."
        )

    return (
        "Pricing usually comes down to window count, size, style, material, and installation scope. "
        "A budget-friendly project often points toward 100 Series, a balanced premium project toward 400 Series, "
        "and a luxury spec toward E-Series. If you share your project details, I can narrow the likely range."
    )


def handle_help(context=None):
    if context is not None:
        context["intent"] = "general_inquiry"
    return (
        "I can help in a few ways:\n\n"
        "- Start a replacement or installation project\n"
        "- Compare window and door styles\n"
        "- Talk through materials, efficiency, and durability\n"
        "- Compare options and pricing direction\n"
        "- Capture your name, email, and phone for quote follow-up\n\n"
        "You can also say continue anytime and I’ll resume the right part of the conversation."
    )


def handle_objection(context, user_id):
    context["intent"] = "general_inquiry"
    data = context.get("data", {})
    missing = next_missing_field(data) if context.get("active_flow") == "project" else None

    ai_reply = generate_chat_reply(
        conversation_history=get_recent_messages(user_id),
        context=context,
        product_knowledge=PRODUCT_KNOWLEDGE,
        product_data=PRODUCT_DATA,
    )
    if ai_reply:
        if missing:
            return f"{ai_reply}\n\nIf it helps, we can keep this simple. {ask_for_field(missing, context)}"
        return ai_reply

    reply = (
        "Totally fair, and I appreciate you saying that. We can keep this simple and low-pressure. "
        "I can give a quick recommendation first, then you can decide if it’s worth moving forward."
    )
    if missing:
        reply += f" To keep it useful, {ask_for_field(missing, context)}"
    return reply


def handle_general(context, message, user_id):
    context["intent"] = "general_inquiry"
    normalized_message = message.lower()

    if normalized_message in {"ok", "okay", "thanks", "thank you"} and context.get("stage") == "completed":
        return "You're all set. We'll be in touch shortly."

    if context.get("contact_offer_made") and not context["data"].get("customer_email"):
        if is_affirmative(normalized_message) or re.search(
            r"\b(quote|consult|consultation|call me|contact me|reach me)\b",
            normalized_message,
        ):
            return start_contact_capture(context)

    if context["stage"] == "collecting" and context["active_flow"] == "project":
        inferred_style = infer_style_from_confirmation(normalized_message, context)
        inferred_material = infer_material_from_confirmation(normalized_message, context)
        if inferred_style or inferred_material:
            return handle_project_intent(context, normalized_message, user_id)

        missing = next_missing_field(context["data"])
        if missing:
            if is_domain_follow_up_question(normalized_message):
                ai_reply = generate_chat_reply(
                    conversation_history=get_recent_messages(user_id),
                    context=context,
                    product_knowledge=PRODUCT_KNOWLEDGE,
                    product_data=PRODUCT_DATA,
                )
                if ai_reply:
                    if missing == "style":
                        if "casement" in ai_reply.lower():
                            context["pending_style_recommendation"] = "casement"
                        elif "sliding" in ai_reply.lower():
                            context["pending_style_recommendation"] = "sliding"
                        elif "picture" in ai_reply.lower():
                            context["pending_style_recommendation"] = "picture"
                        elif "double-hung" in ai_reply.lower() or "double hung" in ai_reply.lower():
                            context["pending_style_recommendation"] = "double-hung"
                    if missing == "material":
                        if "fiberglass" in ai_reply.lower():
                            context["pending_material_recommendation"] = "fiberglass"
                        elif "vinyl" in ai_reply.lower():
                            context["pending_material_recommendation"] = "vinyl"
                        elif "wood" in ai_reply.lower():
                            context["pending_material_recommendation"] = "wood"
                        elif "aluminum" in ai_reply.lower():
                            context["pending_material_recommendation"] = "aluminum"
                        return f"{ai_reply}\n\nTo keep your recommendation moving: {smart_project_follow_up(context, missing)}"

                    return "I’m still tracking your project details in the background. " + smart_project_follow_up(context, missing)

    ai_reply = generate_chat_reply(
        conversation_history=get_recent_messages(user_id),
        context=context,
        product_knowledge=PRODUCT_KNOWLEDGE,
        product_data=PRODUCT_DATA,
    )
    if ai_reply:
        context["fallback_count"] = 0
        return ai_reply

    context["fallback_count"] = context.get("fallback_count", 0) + 1
    if context["fallback_count"] >= 2:
        missing = next_missing_field(context.get("data", {}))
        if missing and context.get("active_flow") == "project":
            return (
                "I want to make this easy and avoid wasting your time. "
                f"Could you share one quick detail so I can give a better recommendation: {ask_for_field(missing, context)}"
            )

    return (
        "I’m here to help with windows and doors. Tell me about your project, ask about styles or materials, or let me know if you want a quote or consultation."
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    raw_message = payload.get("message", "").strip()
    normalized_message = raw_message.lower()

    if not raw_message:
        return jsonify({"reply": "Please enter a message."})

    user_id = ensure_user_id()
    context = get_context()
    if normalized_message in {"ok", "okay", "thanks", "thank you"} and context.get("stage") == "completed":
        return jsonify(
            {
                "reply": "You're all set. We'll be in touch shortly.",
                "context": build_context_payload(context),
                "lead_data": build_lead_payload(context),
            }
        )

    if not context.get("geo_data"):
        user_ip = get_request_ip(request)
        if user_ip and user_ip not in {"127.0.0.1", "::1"}:
            context["geo_data"] = enrich_ip(user_ip)
            log_event("IP Enrichment", {"ip": user_ip, "geo_data": context.get("geo_data")})
    add_context_message(context, "user", raw_message)
    context["data"] = update_contact_data(context["data"], raw_message)
    mark_completed(context, context["data"])
    if not context["data"].get("customer_ip"):
        context["data"] = enrich_request(request, context["data"], existing_geo=context.get("geo_data"))

    save_message(user_id, "user", raw_message)

    message_analysis = analyze_chat_message(
        latest_message=raw_message,
        conversation_history=get_recent_messages(user_id),
        context=context,
    )
    analysis = merge_ai_analysis(context, message_analysis)

    intent = detect_intent(normalized_message, context)
    if intent not in {"contact_capture", "continue"} and not context.get("awaiting_contact_field") and analysis.get("intent") in {
        "project",
        "style_query",
        "model_query",
        "pricing",
        "contact_capture",
        "continue",
        "help",
        "objection",
        "general",
    }:
        intent = analysis["intent"]
    logger.info(
        "Chat message received user_id=%s intent=%s stage=%s awaiting=%s analysis=%s",
        user_id,
        intent,
        context.get("stage"),
        context.get("awaiting_contact_field"),
        analysis,
    )
    intent_map = {
        "project": "project",
        "continue": "continue",
        "style_query": "ask_about_products",
        "model_query": "ask_about_products",
        "pricing": "ask_about_pricing",
        "contact_capture": "lead_capture",
        "objection": "general_inquiry",
        "help": "general_inquiry",
        "general": "general_inquiry",
    }
    context["intent"] = intent_map.get(intent, context.get("intent"))
    ai_first_intents = {"general", "style_query", "model_query", "pricing", "help"}
    should_use_direct_ai = intent in ai_first_intents and not (
        intent == "general"
        and context.get("stage") == "collecting"
        and context.get("active_flow") == "project"
    )

    if should_use_direct_ai:
        ai_reply = generate_chat_reply(
            conversation_history=get_recent_messages(user_id),
            context=context,
            product_knowledge=PRODUCT_KNOWLEDGE,
            product_data=PRODUCT_DATA,
        )
        if ai_reply:
            if should_block_repeat(context, ai_reply):
                ai_reply = "You're all set. Let me know if you need anything else."
            context["last_assistant_reply"] = ai_reply
            extracted_style = extract_recommended_style_from_reply(ai_reply)
            if extracted_style:
                context["pending_style_recommendation"] = extracted_style
            extracted_material = extract_recommended_material_from_reply(ai_reply)
            if extracted_material:
                context["pending_material_recommendation"] = extracted_material
            session["context"] = context
            save_message(user_id, "ai", ai_reply)
            add_context_message(context, "assistant", ai_reply)
            session["context"] = context
            if context["data"].get("lead_id"):
                update_lead_record(context["data"], context)
            return jsonify(
                {
                    "reply": ai_reply,
                    "context": build_context_payload(context),
                    "lead_data": build_lead_payload(context),
                }
            )

    if intent == "project":
        reply = handle_project_intent(context, normalized_message, user_id)
    elif intent == "continue":
        reply = handle_continue(context, user_id)
    elif intent == "style_query":
        reply = handle_style_query(context)
    elif intent == "model_query":
        reply = handle_model_query(context)
    elif intent == "pricing":
        reply = handle_pricing(context)
    elif intent == "contact_capture":
        reply = handle_contact_capture(context, raw_message)
    elif intent == "objection":
        reply = handle_objection(context, user_id)
    elif intent == "help":
        reply = handle_help(context)
    else:
        reply = handle_general(context, raw_message, user_id)

    if context.get("stage") == "completed" and normalized_message in {"ok", "okay", "thanks", "thank you"}:
        reply = "You're all set. We'll be in touch shortly."
    elif should_block_repeat(context, reply):
        reply = "You're all set. Let me know if you need anything else."

    context["last_assistant_reply"] = reply
    add_context_message(context, "assistant", reply)
    session["context"] = context
    save_message(user_id, "ai", reply)
    return jsonify(
        {
            "reply": reply,
            "context": build_context_payload(context),
            "lead_data": build_lead_payload(context),
        }
    )


@app.route("/reset", methods=["POST"])
def reset():
    user_id = session.get("user_id")
    if user_id:
        conn.execute("DELETE FROM chats WHERE user_id = ?", (user_id,))
        conn.commit()
    session.clear()
    return jsonify({"status": "cleared"})


@app.route("/history")
def history():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify([])

    rows = conn.execute(
        "SELECT role, message FROM chats WHERE user_id = ? ORDER BY id ASC",
        (user_id,),
    ).fetchall()
    return jsonify([[row["role"], row["message"]] for row in rows])


@app.route("/dashboard")
def dashboard():
    leads = conn.execute(
        """
        SELECT id, project_intent, lead_data, score, status, created_at,
             customer_name, customer_email, customer_phone, conversation_summary, email_sent,
               salesforce_lead_id, salesforce_sync_status
        FROM leads
        ORDER BY id DESC
        """
    ).fetchall()
    formatted = []
    for row in leads:
        lead_data = json.loads(row["lead_data"]) if row["lead_data"] else {}
        formatted.append(
            {
                "id": row["id"],
                "intent": row["project_intent"],
                "summary": lead_data,
                "conversation_summary": row["conversation_summary"],
                "score": row["score"],
                "status": row["status"],
                "created_at": row["created_at"],
                "customer_name": row["customer_name"],
                "customer_email": row["customer_email"],
                "customer_phone": row["customer_phone"],
                "email_sent": bool(row["email_sent"]),
                "salesforce_lead_id": row["salesforce_lead_id"],
                "salesforce_sync_status": row["salesforce_sync_status"],
            }
        )
    return render_template("dashboard.html", leads=formatted)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
