"""Avisa que uma sessão do estudo ficou gravada.

O Space carrega o log de cada sessão para um dataset privado da Hugging Face.
Um webhook do Hub aponta para ``/api/session-recorded`` e, quando o dataset
muda, este módulo reenvia um aviso para o Discord e por email.

Vive dentro do serviço principal de propósito. Um Space só para isto
adormeceria ao fim de 48 horas sem visitas e perderia o aviso; aqui o processo
acabou de servir a sessão que produziu o log, portanto está acordado.

O aviso **não leva o código do participante**. O caminho do ficheiro no
dataset é ``interaction_logs/966F/...`` e mandar isso para o Discord ou para
uma caixa de correio seria enviar pseudónimos do estudo para fora. Dizer que
chegou um log, e quantos existem, chega para confirmar que a sessão ficou
gravada.

Ambiente (Settings do Space):

  ATLAS_NOTIFY_SECRET   segredo do webhook; sem ele o endpoint fica desligado
  ATLAS_LOG_HF_REPO     dataset dos logs, já usado pelo registo de interação
  HF_TOKEN              token com leitura nesse dataset
  RESEND_API_KEY        chave da API do Resend, o caminho que funciona no Space
  MAIL_TO               destinatário do aviso
  MAIL_FROM             opcional, remetente. Por omissão onboarding@resend.dev
  DISCORD_WEBHOOK_URL   opcional, mas o Space não alcança o Discord
  SMTP_USER             opcional, só serve numa execução local
  SMTP_PASS             opcional, só serve numa execução local

Nota sobre a rede do Space: o egress é filtrado. As portas de SMTP são
recusadas e o TLS para o Discord nunca completa, portanto o email tem de sair
por uma API HTTP.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
import threading
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any, Dict

_logger = logging.getLogger(__name__)


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def notify_enabled() -> bool:
    """Sem segredo definido não há endpoint: evita um POST anónimo a disparar avisos."""
    return bool(_env("ATLAS_NOTIFY_SECRET"))


def _count_logs() -> str:
    """Quantos logs o dataset tem. Devolve texto e nunca levanta."""
    repo = _env("ATLAS_LOG_HF_REPO")
    if not repo:
        return "?"
    try:
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(
            repo, repo_type="dataset", token=_env("HF_TOKEN") or None)
        return str(sum(1 for f in files
                       if f.startswith("interaction_logs/") and f.endswith(".json")))
    except Exception:
        _logger.exception("Could not count logs in %s", repo)
        return "?"


def _send_discord(text: str) -> None:
    """POST to the Discord webhook using the standard library.

    Deliberately not httpx or requests: neither is a declared dependency, and
    a notification must never be the reason the Space fails to build.
    """
    url = _env("DISCORD_WEBHOOK_URL")
    if not url:
        _logger.warning("Discord skipped: DISCORD_WEBHOOK_URL is not set")
        return
    try:
        import json
        import urllib.request

        request = urllib.request.Request(
            url,
            data=json.dumps({"content": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            _logger.info("Discord notified, HTTP %s", response.status)
    except Exception:
        _logger.exception("Discord notification failed")


def _send_email_resend(subject: str, text: str) -> bool:
    """Send through Resend's HTTP API. Returns True if it went out.

    The Space cannot open SMTP at all (ports 465 and 587 are refused) and its
    egress filter drops TLS to most hosts, but api.resend.com is reachable, so
    this is the only email path that works from inside the Space.
    """
    key, to = _env("RESEND_API_KEY"), _env("MAIL_TO")
    if not (key and to):
        return False
    sender = _env("MAIL_FROM") or "Attention Atlas <onboarding@resend.dev>"
    try:
        import json
        import urllib.request

        request = urllib.request.Request(
            "https://api.resend.com/emails",
            data=json.dumps({"from": sender, "to": [to],
                             "subject": subject, "text": text}).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            _logger.info("Email sent through Resend, HTTP %s", response.status)
        return True
    except Exception:
        _logger.exception("Resend email failed")
        return False


def _send_email(subject: str, text: str) -> None:
    if _send_email_resend(subject, text):
        return

    # Fallback for a local run, where SMTP is not blocked.
    user, password, to = _env("SMTP_USER"), _env("SMTP_PASS"), _env("MAIL_TO")
    if not (user and password and to):
        missing = [n for n, v in (("SMTP_USER", user), ("SMTP_PASS", password),
                                  ("MAIL_TO", to)) if not v]
        _logger.warning(
            "Email skipped: no RESEND_API_KEY, and SMTP is missing %s",
            ", ".join(missing))
        return
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to
        msg.set_content(text)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465,
                              context=ssl.create_default_context()) as server:
            server.login(user, password)
            server.send_message(msg)
        _logger.info("Email sent to %s", to)
    except Exception:
        _logger.exception("Email notification failed")


def channel_status() -> Dict[str, bool]:
    """Which channels are configured. Booleans only, never the values."""
    return {
        "discord": bool(_env("DISCORD_WEBHOOK_URL")),
        "email_resend": bool(_env("RESEND_API_KEY") and _env("MAIL_TO")),
        "email_smtp": bool(_env("SMTP_USER") and _env("SMTP_PASS") and _env("MAIL_TO")),
    }


def connectivity() -> Dict[str, str]:
    """What the Space can actually reach.

    A Space sits behind an egress filter that accepts the TCP connection but
    drops the TLS handshake for destinations it does not allow, and refuses
    non-web ports outright. A plain socket test is therefore not enough: each
    host is probed with a real HTTPS request, and huggingface.co acts as the
    control that proves the probe itself works.
    """
    import socket
    import urllib.request

    results: Dict[str, str] = {}

    for label, host, port in (("smtp_gmail_465", "smtp.gmail.com", 465),
                              ("smtp_gmail_587", "smtp.gmail.com", 587)):
        try:
            with socket.create_connection((host, port), timeout=8):
                results[label] = "ok"
        except Exception as exc:
            results[label] = type(exc).__name__

    for label, url in (("control_huggingface", "https://huggingface.co/api/whoami-v2"),
                       ("discord", "https://discord.com/api/v10/gateway"),
                       ("resend", "https://api.resend.com/"),
                       ("telegram", "https://api.telegram.org/")):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "atlas-probe"})
            with urllib.request.urlopen(request, timeout=8) as response:
                results[label] = f"ok HTTP {response.status}"
        except urllib.error.HTTPError as exc:
            # An HTTP error still means the TLS handshake completed.
            results[label] = f"reachable HTTP {exc.code}"
        except Exception as exc:
            results[label] = type(exc).__name__

    return results


def _dispatch(text: str) -> None:
    _logger.info("Dispatching notification. Channels: %s", channel_status())
    _send_discord(text)
    _send_email("Attention Atlas: sessão gravada", text)


def handle_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Trata um evento do webhook. As entregas correm fora do pedido."""
    scope = ((payload.get("event") or {}).get("scope") or "repo.content")
    if scope != "repo.content":
        return {"ignored": scope}

    when = datetime.now(timezone.utc).strftime("%H:%M UTC de %d/%m/%Y")
    total = _count_logs()
    text = (f"Sessão gravada. Chegou um log novo ao dataset às {when}. "
            f"Total de logs: {total}.")

    # Uma entrega lenta não deve fazer o Hub considerar o webhook falhado.
    threading.Thread(target=_dispatch, args=(text,), daemon=True).start()
    _logger.info(text)
    # The channel flags travel back in the response so a test event says
    # straight away whether Discord and email are configured at all, instead
    # of leaving a silent no-op to be guessed at.
    return {"ok": True, "total": total, **channel_status()}
