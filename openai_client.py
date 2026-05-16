import json
import os
from env_loader import load_env_files
from openai import OpenAI


load_env_files()


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_URL = os.environ.get("OPENAI_API_URL")


def openai_is_configured():
    return bool(OPENAI_API_KEY)


def _build_client():
    if not openai_is_configured():
        return None

    client_kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_API_URL:
        client_kwargs["base_url"] = OPENAI_API_URL
    return OpenAI(**client_kwargs)


def _trim_history(conversation_history, limit=8):
    return conversation_history[-limit:] if conversation_history else []


def _serialize_reference(product_knowledge, product_data, context):
    data = (context or {}).get("data", {})
    reference = {
        "intent": (context or {}).get("intent"),
        "current_stage": (context or {}).get("stage"),
        "active_flow": (context or {}).get("active_flow"),
        "awaiting_contact_field": (context or {}).get("awaiting_contact_field"),
        "project_data": {
            "project_intent": data.get("project_intent"),
            "count": data.get("count"),
            "size": data.get("size"),
            "style": data.get("style"),
            "material": data.get("material"),
            "timeline": data.get("timeline"),
            "budget": data.get("budget"),
            "recommended_model": data.get("recommended_model"),
            "customer_name": data.get("customer_name"),
            "customer_email": data.get("customer_email"),
            "customer_phone": data.get("customer_phone"),
        },
        "recent_context_conversation": (context or {}).get("conversation", [])[-8:],
        "known_series": (product_knowledge or {}).get("series") or [],
        "known_types": (product_knowledge or {}).get("types") or (product_data or {}).get("types") or [],
    }
    return json.dumps(reference)


def _extract_output_text(response_data):
    if response_data.get("output_text"):
        return response_data["output_text"].strip()

    output_blocks = response_data.get("output", [])
    collected = []
    for block in output_blocks:
        for content in block.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                collected.append(content["text"])
    return "\n".join(collected).strip()


def _parse_json_text(text):
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None


def analyze_chat_message(latest_message, conversation_history, context):
    client = _build_client()
    if not client or not latest_message:
        return None

    system_prompt = (
        "You analyze customer messages for an Addison Windows sales assistant. "
        "Infer useful structured meaning from messy, non-sequential, natural conversation. "
        "Use the latest user message plus recent context. "
        "Return strict JSON only with these keys: "
        "intent, project_intent, count, size, style, material, timeline, budget, room, "
        "accepted_style, accepted_material, wants_recommendation, wants_quote, customer_goal, next_best_question, confidence. "
        "Valid intent values: project, style_query, model_query, pricing, contact_capture, continue, help, objection, general. "
        "Valid project_intent values: replace, install, or null. "
        "Valid style values: sliding, casement, picture, double-hung, or null. "
        "Valid material values: vinyl, wood, fiberglass, aluminum, or null. "
        "Valid budget values: low, medium, high, or null. "
        "Return size in a normalized NxN format like 8x3 when possible. "
        "Use accepted_style or accepted_material only if the user is clearly accepting or confirming a recommendation. "
        "Set customer_goal to a short phrase like premium look, more airflow, larger view, energy efficiency, fast replacement, or null. "
        "Set next_best_question to one concise natural-language clarifying question only if it would help the sales assistant understand the customer better. Otherwise use null. "
        "If unknown, use null. Do not explain anything outside JSON."
    )

    reference = {
        "latest_message": latest_message,
        "last_assistant_reply": (context or {}).get("last_assistant_reply"),
        "project_data": (context or {}).get("data", {}),
        "active_flow": (context or {}).get("active_flow"),
        "stage": (context or {}).get("stage"),
        "recent_history": _trim_history(conversation_history, limit=6),
    }

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(reference)},
            ],
            max_output_tokens=300,
        )
        output_text = getattr(response, "output_text", None)
        if not output_text:
            output_text = _extract_output_text(response.model_dump())
        parsed = _parse_json_text(output_text)
        if not isinstance(parsed, dict):
            return None
        return parsed
    except Exception:
        return None


def generate_chat_reply(conversation_history, context, product_knowledge, product_data):
    client = _build_client()
    if not client:
        return None

    system_prompt = (
        "You are the production-grade AI sales assistant for Addison Windows. "
        "Behave like a real conversational consultant, not a scripted bot or form. "
        "Always answer the user's question first before asking anything else. "
        "Never repeat a question that has already been answered in the provided context. "
        "Never repeat the same recommendation twice. "
        "Never ask for email if it is already present in context. "
        "Never restart the flow after completion. "
        "Avoid robotic repetition. "
        "Use existing context naturally and only ask the single next most relevant question when needed. "
        "Be concise, confident, helpful, and commercially aware. "
        "Handle hesitation, confusion, and objections with empathy and a low-pressure tone. "
        "Focus only on windows and doors, including styles, materials, installation, replacement, energy efficiency, durability, pricing guidance, and product comparisons. "
        "If the user asks about competitors or comparisons, treat that as a valid window-industry question. Acknowledge competitors professionally, highlight Addison Windows strengths like build quality, energy efficiency, customization, warranty positioning, and long-term value, and avoid negative language. "
        "If the user asks what kinds of windows are available, explain the options clearly before guiding them forward. "
        "If the user is showing buying intent, naturally move toward a quote or consultation and collect name, email, and phone one at a time. "
        "Do not force a rigid intake flow if the user wants advice first. "
        "Do not invent exact prices, warranties, or unsupported technical claims. "
        "If the user asks something unrelated, respond exactly: I’m here to help with windows and doors. Let me know if you’d like help choosing the right option for your home. "
        "Once name, email, and phone are collected, confirm submission and do not ask further intake questions or repeat the recommendation. "
        "If the user says ok, okay, thanks, thank you, or sure after completion, respond naturally and do not restart the flow. "
        "Return plain conversational text only, with no JSON or markup.\n\n"
        "Reference context:\n"
        f"{_serialize_reference(product_knowledge, product_data, context)}"
    )

    input_items = [{"role": "system", "content": system_prompt}]
    for role, message in _trim_history(conversation_history):
        mapped_role = "assistant" if role == "ai" else "user"
        input_items.append(
            {
                "role": mapped_role,
                "content": message,
            }
        )

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=input_items,
            max_output_tokens=300,
        )
        if getattr(response, "output_text", None):
            return response.output_text.strip() or None
        response_data = response.model_dump()
        return _extract_output_text(response_data) or None
    except Exception:
        return None
