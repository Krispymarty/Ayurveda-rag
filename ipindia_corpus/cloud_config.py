"""Operator-only public research configuration; never read from source documents."""
from dataclasses import dataclass
import os
import re
from urllib.parse import urlsplit

LOCAL_HOST = re.compile(r"(?:localhost|127\.0\.0\.1|api|testserver)(?::([0-9]{1,5}))?", re.I)
DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", re.I)


def _authority(value):
    if not isinstance(value, str) or not value or len(value) > 259 or not value.isascii():
        raise ValueError("invalid_deployment_host")
    if value != value.strip() or value.count(":") > 1:
        raise ValueError("invalid_deployment_host")
    hostname, colon, port = value.lower().partition(":")
    if len(hostname) > 253 or not all(DNS_LABEL.fullmatch(label) for label in hostname.split(".")):
        raise ValueError("invalid_deployment_host")
    if colon and (not port.isdigit() or len(port) > 5 or not 1 <= int(port) <= 65535):
        raise ValueError("invalid_deployment_port")
    return hostname + (":" + str(int(port)) if colon else "")


def _origin(value):
    try:
        if not isinstance(value, str) or any(ord(char) <= 32 or ord(char) > 126 for char in value):
            raise ValueError("invalid_public_origin")
        parsed = urlsplit(value)
        if (parsed.scheme != "https" or not parsed.netloc or parsed.username is not None
                or parsed.password is not None or parsed.path or parsed.query or parsed.fragment):
            raise ValueError("invalid_public_origin")
        authority = _authority(parsed.netloc)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_public_origin") from exc
    if authority.endswith(":443"):
        authority = authority[:-4]
    return "https://" + authority


@dataclass(frozen=True)
class DeploymentConfig:
    mode: str
    public_origin: str | None = None
    allowed_hosts: tuple[str, ...] = ()

    @property
    def public_research(self):
        return self.mode == "public_research"

    def allows_host(self, value):
        if not self.public_research:
            match = LOCAL_HOST.fullmatch(value)
            return bool(match and (not match[1] or 1 <= int(match[1]) <= 65535))
        try:
            return _authority(value) in self.allowed_hosts
        except ValueError:
            return False

    def allows_origin(self, value):
        if not self.public_research:
            return True
        try:
            return _origin(value) == self.public_origin
        except ValueError:
            return False


def load_deployment_config(environ=None):
    env = os.environ if environ is None else environ
    mode = env.get("AYURIP_DEPLOYMENT_MODE", "local_internal")
    if mode not in {"local_internal", "public_research"}:
        raise ValueError("invalid_deployment_mode")
    if mode == "local_internal":
        if env.get("AYURIP_PUBLIC_ORIGIN") or env.get("AYURIP_ALLOWED_HOSTS"):
            raise ValueError("public_configuration_requires_public_mode")
        return DeploymentConfig(mode)
    origin = _origin(env.get("AYURIP_PUBLIC_ORIGIN", ""))
    primary = origin.removeprefix("https://")
    allowed = [primary]
    if ":" not in primary:
        allowed.append(primary + ":443")
    extras = env.get("AYURIP_ALLOWED_HOSTS", "")
    if extras:
        values = extras.split(",")
        if len(values) > 16 or any(not value.strip() for value in values):
            raise ValueError("invalid_deployment_host_list")
        allowed.extend(_authority(value.strip()) for value in values)
    return DeploymentConfig(mode, origin, tuple(dict.fromkeys(allowed)))
