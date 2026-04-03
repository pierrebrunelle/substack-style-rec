"""PixelTable schema + data setup.

Creates tables, indexes, computed columns, and loads video data from
the Twelve Labs index. Run once — everything is idempotent.

Usage:
    uv run setup_pixeltable.py
    uv run setup_pixeltable.py --with-videos   # include local video files for segment embeddings
"""

import argparse
import csv
import logging
import re
from pathlib import Path

import httpx
import pixeltable as pxt
from pixeltable.functions.twelvelabs import embed
from pixeltable.functions.video import video_splitter

import config
from functions import analyze_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

marengo = embed.using(model_name="marengo3.0")

VIDEOS_DIR = Path(__file__).resolve().parent / "data" / "videos"
CSV_PATH = Path(__file__).resolve().parent.parent / "scripts" / "videos_metadata.csv"


def strip_extension(filename: str) -> str:
    return re.sub(r"\.(mp4|webm|mkv|mov)$", "", filename, flags=re.IGNORECASE).strip()


def setup(with_videos: bool = False):
    logger.info("Setting up PixelTable under '%s'...", config.APP_NAMESPACE)
    pxt.create_dir(config.APP_NAMESPACE, if_exists="ignore")

    # -- Schema ---------------------------------------------------------------

    creators = pxt.create_table(
        f"{config.APP_NAMESPACE}.creators",
        {
            "id": pxt.Required[pxt.String],
            "name": pxt.String,
            "avatar_url": pxt.String,
            "description": pxt.String,
        },
        primary_key=["id"],
        if_exists="ignore",
    )

    videos = pxt.create_table(
        f"{config.APP_NAMESPACE}.videos",
        {
            "id": pxt.Required[pxt.String],
            "title": pxt.String,
            "creator_id": pxt.String,
            "category": pxt.String,
            "duration": pxt.Int,
            "thumbnail_url": pxt.String,
            "hls_url": pxt.String,
            "upload_date": pxt.String,
            "video": pxt.Video,
        },
        primary_key=["id"],
        if_exists="ignore",
    )

    videos.add_embedding_index(
        "title", string_embed=marengo, idx_name="title_marengo", if_exists="ignore"
    )
    videos.add_computed_column(
        raw_attributes=analyze_video(videos.id), if_exists="ignore"
    )
    videos.add_computed_column(topic=videos.raw_attributes["topic"], if_exists="ignore")
    videos.add_computed_column(style=videos.raw_attributes["style"], if_exists="ignore")
    videos.add_computed_column(tone=videos.raw_attributes["tone"], if_exists="ignore")
    logger.info("  Schema ready (creators + videos + title index + analyze columns)")

    # -- Data ingestion from Twelve Labs index --------------------------------

    logger.info(
        "  Fetching videos from Twelve Labs index %s ...", config.TWELVELABS_INDEX_ID
    )
    tl_videos = _fetch_tl_videos()
    logger.info("  Found %d videos in TL index", len(tl_videos))

    yt_file_map = _build_video_file_map() if with_videos else {}
    if yt_file_map:
        logger.info("  Found %d local video files", len(yt_file_map))

    # Insert creators
    seen: set[str] = set()
    creator_rows = []
    for tlv in tl_videos:
        meta = tlv.get("user_metadata") or {}
        cid, cname = meta.get("creatorId"), meta.get("creatorName")
        if cid and cname and cid not in seen:
            seen.add(cid)
            creator_rows.append(
                {
                    "id": cid,
                    "name": cname,
                    "avatar_url": "",
                    "description": config.CREATOR_DESCRIPTIONS.get(cid, ""),
                }
            )
    if creator_rows:
        status = creators.insert(creator_rows, on_error="ignore")
        logger.info("  Creators: %d inserted", status.num_rows)

    # Insert videos
    video_rows = []
    for tlv in tl_videos:
        meta = tlv.get("user_metadata") or {}
        sys_meta = tlv.get("system_metadata", {})
        hls = tlv.get("hls") or {}
        cid = meta.get("creatorId")
        if not cid or not meta.get("creatorName"):
            continue
        row = {
            "id": tlv["_id"],
            "title": strip_extension(sys_meta.get("filename", "")),
            "creator_id": cid,
            "category": meta.get("category", "interview"),
            "duration": round(sys_meta.get("duration", 0)),
            "thumbnail_url": (hls.get("thumbnail_urls") or [""])[0],
            "hls_url": hls.get("video_url", ""),
            "upload_date": meta.get("uploadDate", ""),
        }
        yt_id = meta.get("youtubeId")
        if with_videos and yt_id and yt_id in yt_file_map:
            row["video"] = yt_file_map[yt_id]
        video_rows.append(row)

    if video_rows:
        n_with_files = sum(1 for r in video_rows if "video" in r)
        logger.info(
            "  Inserting %d videos (%d with files)...", len(video_rows), n_with_files
        )
        status = videos.insert(video_rows, on_error="ignore")
        logger.info(
            "  Videos: %d inserted, %d errors", status.num_rows, status.num_excs
        )

    # Backfill video paths for existing rows
    if with_videos and yt_file_map:
        tl_to_yt = {
            tlv["_id"]: (tlv.get("user_metadata") or {}).get("youtubeId")
            for tlv in tl_videos
        }
        backfilled = 0
        for row in videos.select(videos.id).collect():
            yt_id = tl_to_yt.get(row["id"])
            if yt_id and yt_id in yt_file_map:
                videos.update(
                    {"video": yt_file_map[yt_id]}, where=(videos.id == row["id"])
                )
                backfilled += 1
        if backfilled:
            logger.info("  Backfilled video paths for %d rows", backfilled)

    # -- Video segment view + embedding index ---------------------------------

    view_name = f"{config.APP_NAMESPACE}.video_segments"
    has_files = False
    try:
        has_files = videos.where(videos.video != None).count() > 0  # noqa: E711
    except Exception:
        pass

    if has_files:
        pxt.create_view(
            view_name,
            videos.where(videos.video != None),  # noqa: E711
            iterator=video_splitter(videos.video, duration=30),
            if_exists="ignore",
        )
        segments = pxt.get_table(view_name)
        segments.add_embedding_index(
            "video_segment",
            embedding=marengo,
            idx_name="segment_marengo",
            if_exists="ignore",
        )
        logger.info("  Video segments view + Marengo index ready")
    else:
        logger.info(
            "  Skipping video_segments (no video files — run download_videos.py first)"
        )

    logger.info("\nSetup complete.")


def _fetch_tl_videos() -> list[dict]:
    result: list[dict] = []
    page = 1
    while True:
        resp = httpx.get(
            f"{config.TWELVELABS_BASE_URL}/indexes/{config.TWELVELABS_INDEX_ID}/videos",
            params={"page": page, "page_limit": 50},
            headers={"x-api-key": config.TWELVELABS_API_KEY},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        result.extend(data.get("data", []))
        if page >= data.get("page_info", {}).get("total_page", 1):
            break
        page += 1
    return result


def _build_video_file_map() -> dict[str, str]:
    if not CSV_PATH.exists():
        return {}
    result: dict[str, str] = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yt_id = row.get("video_id", "").strip()
            if yt_id:
                fp = VIDEOS_DIR / f"{yt_id}.mp4"
                if fp.exists():
                    result[yt_id] = str(fp)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Setup PixelTable schema + ingest data"
    )
    parser.add_argument(
        "--with-videos",
        action="store_true",
        help="Include local video files for segment embeddings",
    )
    args = parser.parse_args()
    setup(with_videos=args.with_videos)
