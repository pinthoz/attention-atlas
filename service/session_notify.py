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
        with urllib.request.urlopen(request, timeout=10):
            pass
    except Exception:
        _logger.exception("Discord notification failed")


def _send_email(subject: str, text: str) -> None:
    user, password, to = _env("SMTP_USER"), _env("SMTP_PASS"), _env("MAIL_TO")
    if not (user and password and to):
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
    except Exception:
        _logger.exception("Email notification failed")


def _dispatch(text: str) -> None:
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
    return {"ok": True, "total": total}
