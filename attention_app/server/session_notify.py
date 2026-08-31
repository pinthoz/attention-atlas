"""Avisa que uma sessão do estudo ficou gravada.

Chamado por :mod:`attention_app.server.interaction_log` no fim de uma sessão,
depois de o log subir para o dataset privado.

**Só avisa sessões com participante.** O Space é público, portanto qualquer
visita gera um log, e antes isto vinha de um webhook do Hub que disparava a
cada alteração do dataset, sem forma de distinguir uma sessão real de alguém
a espreitar: o payload do webhook nem traz o caminho do ficheiro. Aqui o
código do participante é conhecido, e a distinção é exata.

O aviso **não leva o código do participante**. O caminho do ficheiro no
dataset é ``interaction_logs/966F/...`` e mandar isso para uma caixa de
correio seria enviar pseudónimos do estudo para fora. Dizer que
chegou um log, e quantos existem, chega para confirmar que a sessão ficou
gravada.

Ambiente (Settings do Space):

  ATLAS_NOTIFY_SECRET   segredo do webhook; sem ele o endpoint fica desligado
  ATLAS_LOG_HF_REPO     dataset dos logs, já usado pelo registo de interação
  HF_TOKEN              token com leitura nesse dataset
  RESEND_API_KEY        chave da API do Resend, o caminho que funciona no Space
  MAIL_TO               destinatário do aviso
  MAIL_FROM             opcional, remetente. Por omissão onboarding@resend.dev
  SMTP_USER             opcional, só serve numa execução local
  SMTP_PASS             opcional, só serve numa execução local

Nota sobre a rede do Space: o egress é filtrado. As portas de SMTP são
recusadas, portanto o email tem de sair por uma API HTTP. O Discord também
não é alcançável, e foi por isso abandonado.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
import threading
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Dict, Optional

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
                     "Content-Type": "application/json",
                     # Cloudflare sits in front of the Resend API and answers
                     # 403 "error code: 1010" to urllib's default agent, which
                     # it reads as a bot signature.
                     "User-Agent": "attention-atlas/1.0",
                     "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            _logger.info("Email sent through Resend, HTTP %s", response.status)
        return True
    except urllib.error.HTTPError as exc:
        # Resend explains the refusal in the body, and without it a 403 is
        # indistinguishable from a bad key. The usual cause is the free plan:
        # with no verified domain it only accepts the account's own address.
        try:
            detail = exc.read().decode("utf-8", "replace")[:400]
        except Exception:
            detail = "(no body)"
        _logger.error("Resend refused the email: HTTP %s - %s", exc.code, detail)
        return False
    except Exception:
        _logger.exception("Resend email failed")
        return False


def _send_email(subject: str, text: str) -> None:
    if _send_email_resend(subject, text):
        return
    if _env("RESEND_API_KEY"):
        # Resend was configured and refused: the reason is already logged, and
        # falling through to SMTP would only add a misleading second message.
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
    _send_email("Attention Atlas: sessão gravada", text)


def notify_session_saved(participant: Optional[str], n_events: int) -> None:
    """Avisa que a sessão de um participante ficou gravada.

    Uma sessão sem participante é uma visita ao Space, não um dado do estudo,
    e não gera aviso. Nunca levanta: um aviso perdido não pode estragar o
    fecho de uma sessão.
    """
    if not participant:
        _logger.info("No notification: session has no participant code.")
        return
    try:
        when = datetime.now(timezone.utc).strftime("%H:%M UTC de %d/%m/%Y")
        text = (f"Sessão gravada às {when}, com {n_events} eventos "
                f"registados. Total de logs no dataset: {_count_logs()}.")
        # Fora do fecho da sessão: uma entrega lenta não deve prender o
        # participante numa janela que não fecha.
        threading.Thread(target=_dispatch, args=(text,), daemon=True).start()
    except Exception:
        _logger.exception("Could not send the session notification")
