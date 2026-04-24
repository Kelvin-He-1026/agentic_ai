import os
import re
import time
import json
import requests
from html import unescape
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
STATE_FILE = "last_seen_trump_post.txt"
ARCHIVE_URL = "https://ix.cnn.io/data/truth-social/truth_archive.json"

EST = timezone(timedelta(hours=-5), name="EST")


def clean_html(text: str) -> str:
    return unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


def fetch_latest_post() -> dict:
    resp = requests.get(ARCHIVE_URL, timeout=30)
    resp.raise_for_status()

    posts = resp.json()
    if not posts:
        raise ValueError("No posts returned from archive")

    latest = posts[0]  # archive is newest-first in the scraper output

    dt_utc = datetime.fromisoformat(latest["created_at"].replace("Z", "+00:00"))
    dt_est = dt_utc.astimezone(EST)

    return {
        "id": str(latest["id"]),
        "created_at_est": dt_est.strftime("%Y-%m-%d %I:%M:%S %p %Z"),
        "content": clean_html(latest.get("content", "")) or "[media-only post]",
        "url": latest.get("url"),
    }


def send_to_discord(post: dict) -> None:
    payload = {
        "content": "New Truth Social post from @realDonaldTrump",
        "embeds": [
            {
                "title": "Latest post",
                "description": post["content"][:4000],
                "url": post["url"],
                "fields": [
                    {"name": "Time (EST)", "value": post["created_at_est"], "inline": False},
                    {"name": "Post ID", "value": post["id"], "inline": False},
                ],
            }
        ],
        "allowed_mentions": {"parse": []},
    }

    resp = requests.post(
        DISCORD_WEBHOOK_URL,
        params={"wait": "true"},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()


def load_last_seen_id():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


def save_last_seen_id(post_id: str):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        f.write(post_id)


if __name__ == "__main__":
    if not DISCORD_WEBHOOK_URL:
        raise ValueError("DISCORD_WEBHOOK_URL not found in environment or .env")

    last_seen_id = load_last_seen_id()

    while True:
        try:
            latest = fetch_latest_post()

            if last_seen_id is None:
                last_seen_id = latest["id"]
                save_last_seen_id(last_seen_id)
                print(f"Initialized with latest post: {latest['created_at_est']}")
            elif latest["id"] != last_seen_id:
                send_to_discord(latest)
                save_last_seen_id(latest["id"])
                last_seen_id = latest["id"]
                print("Sent new post to Discord:")
                print(json.dumps(latest, indent=2))
            else:
                print(f"No new post. Latest still: {latest['created_at_est']}")

        except Exception as e:
            print("Error:", e)

        time.sleep(60)