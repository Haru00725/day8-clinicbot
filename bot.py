import os
import sys
import json
import re
import sqlite3
import threading
from datetime import datetime, timedelta

import numpy as np
import requests
from flask import Flask, request
from google import genai
from google.genai import types

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------- CONFIG ----------------
CLINIC_NAME = "GlowDerm Skin & Hair Clinic"
GOOGLE_REVIEW_LINK = os.getenv("GOOGLE_REVIEW_LINK", "https://g.page/r/YOUR_LINK/review")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")        # Meta access token
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")      # Meta test number's Phone Number ID
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "clinicbot123")
ADMIN_KEY = os.getenv("ADMIN_KEY", "demo123")

GRAPH_VERSION = "v21.0"  # change if Meta shows a newer version in API Setup

# If you get a "model not found" error, change these to current Gemini models
CHAT_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"

SLOTS = ["10:00", "11:00", "12:00", "14:00", "15:00", "16:00", "17:00"]

client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------- DATABASE ----------------
db = sqlite3.connect(os.getenv("DB_PATH", "clinic.db"), check_same_thread=False)
db_lock = threading.Lock()
db.execute(
    """CREATE TABLE IF NOT EXISTS appointments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phone TEXT, name TEXT, service TEXT,
        date TEXT, time TEXT, status TEXT DEFAULT 'booked')"""
)
db.commit()


def free_slots(date):
    rows = db.execute(
        "SELECT time FROM appointments WHERE date=? AND status!='cancelled'", (date,)
    ).fetchall()
    taken = {r[0] for r in rows}
    return [s for s in SLOTS if s not in taken]


def availability_text():
    lines = []
    today = datetime.now()
    for i in range(0, 8):
        d = today + timedelta(days=i)
        if d.weekday() == 6:  # Sunday closed
            continue
        ds = d.strftime("%Y-%m-%d")
        lines.append(f"{d.strftime('%A')} {ds}: {', '.join(free_slots(ds)) or 'fully booked'}")
    return "\n".join(lines)


# ---------------- RAG ----------------
def embed(texts):
    res = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    return np.array([e.values for e in res.embeddings])


try:
    with open("clinic_knowledge.txt", encoding="utf-8") as f:
        CHUNKS = [c.strip() for c in f.read().split("\n\n") if c.strip()]
except Exception:
    CHUNKS = []

CHUNK_VECS = embed(CHUNKS) if CHUNKS else np.array([])


def retrieve(query, k=3):
    if not CHUNKS or CHUNK_VECS.size == 0:
        return []
    qv = embed([query])[0]
    sims = (CHUNK_VECS @ qv) / (
        np.linalg.norm(CHUNK_VECS, axis=1) * np.linalg.norm(qv)
    )
    top = np.argsort(sims)[::-1][:k]
    return [CHUNKS[i] for i in top]



# ---------------- BOT BRAIN ----------------
HISTORY = {}


def parse_json(raw):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


def ask_bot(phone, user_msg):
    history = HISTORY.setdefault(phone, [])
    search_text = user_msg + " " + " ".join(t for _, t in history[-2:])
    context = "\n\n".join(retrieve(search_text))
    today = datetime.now().strftime("%A, %Y-%m-%d")

    system = f"""You are the WhatsApp assistant for {CLINIC_NAME}. Today is {today}.

Answer ONLY using the clinic information below. If the answer is not there, say you
will have the clinic team confirm it, and share nothing invented. Never give medical
advice or diagnose; suggest booking a consultation instead. Keep replies short, warm
and WhatsApp-style (2-4 sentences, no markdown).

CLINIC INFORMATION:
{context}

AVAILABLE SLOTS (next 7 days):
{availability_text()}

BOOKING: To book, you need the customer's name, service, date and time. Ask for
missing details one at a time. Once you have all four AND the customer has confirmed,
include a booking object. Only offer slots from the available list.

Return ONLY JSON:
{{"reply": "message to customer", "booking": null}}
or, when confirmed:
{{"reply": "confirmation message", "booking": {{"name": "...", "service": "...", "date": "YYYY-MM-DD", "time": "HH:MM"}}}}
"""

    convo = "\n".join(f"{role}: {text}" for role, text in history[-10:])
    prompt = f"Conversation so far:\n{convo}\n\nCustomer: {user_msg}"

    try:
        resp = client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                temperature=0.3,
            ),
        )
        data = parse_json(resp.text)
        reply = data.get("reply", "Sorry, could you say that again?")
        booking = data.get("booking")

        if booking:
            date, time = booking.get("date"), booking.get("time")
            with db_lock:
                if time in free_slots(date):
                    db.execute(
                        "INSERT INTO appointments (phone, name, service, date, time) VALUES (?,?,?,?,?)",
                        (phone, booking["name"], booking["service"], date, time),
                    )
                    db.commit()
                else:
                    reply = (
                        f"Sorry, {time} on {date} just got taken. "
                        f"Free slots that day: {', '.join(free_slots(date)) or 'none'}."
                    )
    except Exception as e:
        print("Bot error:", e)
        reply = "Sorry, something went wrong. Please try again in a moment."

    history.append(("Customer", user_msg))
    history.append(("Assistant", reply))
    return reply


# ---------------- WHATSAPP (META CLOUD API) ----------------
def send_whatsapp(to, body):
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        print(f"[WhatsApp to {to}] {body}")
        return
    url = f"https://graph.facebook.com/{GRAPH_VERSION}/{PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body},
    }
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=15)
        if r.status_code >= 400:
            print("Meta send error:", r.status_code, r.text)
    except Exception as e:
        print("Meta send exception:", e)


def handle_message(phone, text):
    reply = ask_bot(phone, text)
    send_whatsapp(phone, reply)


def complete_appointment(appt_id):
    row = db.execute(
        "SELECT phone, name FROM appointments WHERE id=?", (appt_id,)
    ).fetchone()
    if not row:
        return False
    with db_lock:
        db.execute("UPDATE appointments SET status='completed' WHERE id=?", (appt_id,))
        db.commit()
    phone, name = row
    send_whatsapp(
        phone,
        f"Hi {name}, thank you for visiting {CLINIC_NAME}! We hope you had a great "
        f"experience. If you have a minute, a Google review would mean a lot to us: "
        f"{GOOGLE_REVIEW_LINK}",
    )
    return True


# ---------------- FLASK APP ----------------
app = Flask(__name__)
PROCESSED = set()  # Meta retries webhooks, so skip messages we've already handled


@app.route("/")
def home():
    return "Clinic bot is running"


@app.route("/webhook", methods=["GET"])
def verify():
    if (
        request.args.get("hub.mode") == "subscribe"
        and request.args.get("hub.verify_token") == VERIFY_TOKEN
    ):
        return request.args.get("hub.challenge", ""), 200
    return "Forbidden", 403


@app.route("/webhook", methods=["POST"])
def incoming():
    data = request.get_json(silent=True) or {}
    try:
        value = data["entry"][0]["changes"][0]["value"]
        messages = value.get("messages")
        if messages:
            m = messages[0]
            if m.get("type") == "text" and m["id"] not in PROCESSED:
                PROCESSED.add(m["id"])
                threading.Thread(
                    target=handle_message,
                    args=(m["from"], m["text"]["body"]),
                    daemon=True,
                ).start()
    except (KeyError, IndexError):
        pass
    return "OK", 200  # reply fast so Meta doesn't retry


@app.route("/appointments")
def appointments():
    if request.args.get("key") != ADMIN_KEY:
        return "Forbidden", 403
    rows = db.execute(
        "SELECT id, name, service, date, time, status FROM appointments"
    ).fetchall()
    if not rows:
        return "No appointments yet"
    return "<br>".join(
        f"#{r[0]} | {r[1]} | {r[2]} | {r[3]} {r[4]} | {r[5]}" for r in rows
    )


@app.route("/complete/<int:appt_id>")
def complete(appt_id):
    if request.args.get("key") != ADMIN_KEY:
        return "Forbidden", 403
    ok = complete_appointment(appt_id)
    return "Review request sent" if ok else ("Not found", 404)


# ---------------- TERMINAL TEST MODE ----------------
def terminal_test():
    print("Chat with the bot. Type '/done 1' to mark appointment #1 complete, 'quit' to exit.\n")
    while True:
        msg = input("You: ").strip()
        if msg.lower() == "quit":
            break
        if msg.startswith("/done"):
            complete_appointment(int(msg.split()[1]))
            continue
        print("Bot:", ask_bot("test-user", msg), "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        terminal_test()
    else:
        app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))