def log_event(event, data):
    try:
        print(f"[{event}] -> {data}")
    except Exception:
        pass