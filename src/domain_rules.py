from urllib.parse import urlparse


TRUSTED_DOMAINS = set()

with open(
    "data/trusted_domains.txt",
    "r",
    encoding="utf-8"
) as file:

    for line in file:
        domain = line.strip().lower()

        if domain:
            TRUSTED_DOMAINS.add(domain)


def get_hostname(url):

    url = url.strip()

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)

    return parsed.hostname.lower() if parsed.hostname else ""


def is_trusted_domain(url):

    hostname = get_hostname(url)

    if not hostname:
        return False

    # Exact domain
    if hostname in TRUSTED_DOMAINS:
        return True

    # Subdomain of trusted domain
    for domain in TRUSTED_DOMAINS:

        if hostname.endswith("." + domain):
            return True

    return False