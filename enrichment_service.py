import json
import urllib.error
import urllib.request


def enrich_ip(ip):
    if not ip:
        return {}

    try:
        with urllib.request.urlopen(f"http://ip-api.com/json/{ip}", timeout=2) as response:
            data = json.loads(response.read().decode())

        return {
            "city": data.get("city"),
            "region": data.get("regionName"),
            "zip": data.get("zip"),
            "isp": data.get("isp"),
            "lat": data.get("lat"),
            "lon": data.get("lon"),
        }
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        print(f"[IP Enrichment Error]: {exc}")
        return {}