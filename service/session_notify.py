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
  DISCORD_WEBHOOK_URL   opcional
  SMTP_USER             opcional, endereço Gmail que envia
  SMTP_PASS             opcional, palavra-passe de aplicação desse Gmail
  MAIL_TO               opcional, destinatário do aviso
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


def _send_email(subject: str, text: str) -> None:
    user, password, to = _env("SMTP_USER"), _env("SMTP_PASS"), _env("MAIL_TO")
    if not (user and password and to):
        missing = [n for n, v in (("SMTP_USER", user), ("SMTP_PASS", password),
                                  ("MAIL_TO", to)) if not v]
        _logger.warning("Email skipped: %s not set", ", ".join(missing))
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
        "email": bool(_env("SMTP_USER") and _env("SMTP_PASS") and _env("MAIL_TO")),
    }


def connectivity() -> Dict[str, str]:
    """Can the Space actually reach Discord and Gmail?

    A Space restricts outbound traffic, and a blocked destination shows up as
    an SSL handshake that never completes. Testing the TCP connection tells
    the two failures apart: a wrong secret from a destination the platform
    will not let us reach at all.
    """
    import socket

    out = {}
    for label, host, port in (("discord", "discord.com", 443),
                              ("gmail_smtp", "smtp.gmail.com", 465)):
        try:
            with socket.create_connection((host, port), timeout=8):
                out[label] = "ok"
        except Exception as exc:
            out[label] = f"{type(exc).__name__}: {exc}"
    return out


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
