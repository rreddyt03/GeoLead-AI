def should_send_email(data):
    if not (data.get("customer_name") and data.get("customer_email") and data.get("customer_phone")):
        return False

    if data.get("email_sent"):
        return False

    return True