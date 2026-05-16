import ipaddress
import json
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_TIMEOUT = 2
USER_AGENT = "smart-cta-geo-intelligence/1.0"


def _fetch_json(url, headers=None, timeout=DEFAULT_TIMEOUT):
    request_obj = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request_obj, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _is_public_ip(ip):
    if not ip:
        return False
    try:
        parsed_ip = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not (
        parsed_ip.is_loopback
        or parsed_ip.is_private
        or parsed_ip.is_link_local
        or parsed_ip.is_multicast
        or parsed_ip.is_reserved
        or parsed_ip.is_unspecified
    )


def get_real_ip(request):
    headers = [
        "X-Forwarded-For",
        "X-Real-IP",
        "CF-Connecting-IP",
        "True-Client-IP",
    ]

    for header_name in headers:
        raw_value = request.headers.get(header_name)
        if not raw_value:
            continue
        ip = raw_value.split(",")[0].strip()
        if _is_public_ip(ip):
            return ip

    remote_addr = (request.remote_addr or "").strip()
    if _is_public_ip(remote_addr):
        return remote_addr

    if remote_addr in {"127.0.0.1", "localhost", ""}:
        return "DEV_IP"

    return remote_addr


def merge_sources(ip, sources):
    def pick(current, new_value):
        return current if current not in {None, ""} else new_value

    result = {
        "ip": ip,
        "city": None,
        "region": None,
        "zip": None,
        "country": None,
        "lat": None,
        "lon": None,
        "isp": None,
    }

    for source in sources:
        if not isinstance(source, dict):
            continue
        result["city"] = pick(result["city"], source.get("city"))
        result["region"] = pick(result["region"], source.get("region") or source.get("regionName"))
        result["zip"] = pick(result["zip"], source.get("postal") or source.get("zip"))
        result["country"] = pick(result["country"], source.get("country_name") or source.get("country"))
        result["lat"] = pick(result["lat"], source.get("latitude") or source.get("lat"))
        result["lon"] = pick(result["lon"], source.get("longitude") or source.get("lon"))
        result["isp"] = pick(result["isp"], source.get("org") or source.get("isp"))

    return result


def enrich_ip_multi(ip, existing_geo=None):
    if ip == "DEV_IP":
        return {
            "ip": ip,
            "city": "Development",
            "region": "Local",
            "zip": "00000",
            "country": "Local",
            "lat": None,
            "lon": None,
            "isp": "Localhost",
        }

    if not _is_public_ip(ip):
        return merge_sources(ip, [existing_geo or {}])

    sources = []
    if existing_geo:
        sources.append(existing_geo)

    encoded_ip = urllib.parse.quote(ip, safe="")

    try:
        source_one = _fetch_json(f"http://ip-api.com/json/{encoded_ip}")
        if source_one.get("status") == "success":
            sources.append(source_one)
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        pass

    try:
        source_two = _fetch_json(
            f"https://ipapi.co/{encoded_ip}/json/",
            headers={"User-Agent": USER_AGENT},
        )
        if not source_two.get("error"):
            sources.append(source_two)
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        pass

    return merge_sources(ip, sources)


def reverse_lookup(lat, lon):
    if lat in {None, ""} or lon in {None, ""}:
        return None

    query = urllib.parse.urlencode({"lat": lat, "lon": lon, "format": "jsonv2"})
    url = f"https://nominatim.openstreetmap.org/reverse?{query}"
    try:
        result = _fetch_json(url, headers={"User-Agent": USER_AGENT})
        return result.get("display_name")
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return None


def geo_confidence(data):
    if data.get("ip") == "DEV_IP":
        return 20

    score = 0
    if data.get("city"):
        score += 25
    if data.get("region"):
        score += 20
    if data.get("zip"):
        score += 25
    if data.get("lat") not in {None, ""}:
        score += 15
    if data.get("lon") not in {None, ""}:
        score += 15
    return score


def enrich_request(request, data, existing_geo=None):
    try:
        print("[DEBUG] Raw Headers:", dict(request.headers))
        print("[DEBUG] Remote Addr:", request.remote_addr)

        ip = get_real_ip(request)
        print("[DEBUG] Final IP:", ip)
        if not ip:
            return data

        geo = enrich_ip_multi(ip, existing_geo=existing_geo)
        approx_address = reverse_lookup(geo.get("lat"), geo.get("lon"))
        confidence = geo_confidence(geo)

        data["customer_ip"] = ip
        data["geo_city"] = geo.get("city")
        data["geo_region"] = geo.get("region")
        data["geo_zip"] = geo.get("zip")
        data["geo_country"] = geo.get("country")
        data["geo_isp"] = geo.get("isp")
        data["geo_lat"] = geo.get("lat")
        data["geo_lon"] = geo.get("lon")
        data["geo_address"] = approx_address
        data["geo_confidence"] = confidence
    except Exception:
        return data

    return data