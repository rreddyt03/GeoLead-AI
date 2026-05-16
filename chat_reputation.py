def should_block_repeat(context, reply):
    last_reply = context.get("last_assistant_reply", "")

    if not last_reply or not reply:
        return False

    if reply.strip() == last_reply.strip():
        return True

    if "My best fit right now is" in reply and "My best fit right now is" in last_reply:
        return True

    return False


def mark_completed(context, data):
    if data.get("customer_name") and data.get("customer_email") and data.get("customer_phone"):
        context["stage"] = "completed"