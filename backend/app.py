"""Pixeltable schema for Substack TV-style video recommendations.

Hierarchy:
    creators (table)       creator profiles
    videos (table)         source videos + Twelve Labs Analyze attributes
    -> video_scenes (view) one row per scene (video_splitter), Marengo embeddings for semantic search

Create or update the schema with:
    pxt app update app.py substack_rec
"""

from __future__ import annotations

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.catalog.model import Column, EmbeddingIndex

from functions import analyze_video

marengo = pxtf.twelvelabs.embed.using(model_name="marengo3.0")
TableModel = pxt.model_base()


class Creators(TableModel, name="creators"):
    id = Column(type=pxt.Required[pxt.String], primary_key=True)
    name: pxt.String
    avatar_url: pxt.String
    description: pxt.String


class Videos(TableModel, name="videos"):
    id = Column(type=pxt.Required[pxt.String], primary_key=True)
    title: pxt.String
    creator_id: pxt.String
    category: pxt.String
    duration: pxt.Int
    thumbnail_url: pxt.String
    hls_url: pxt.String
    upload_date: pxt.String
    video: pxt.Video
    raw_attributes = analyze_video(id)
    topic = raw_attributes["topic"]
    style = raw_attributes["style"]
    tone = raw_attributes["tone"]
    scenes = video.scene_detect_histogram(fps=1, threshold=0.9, min_scene_len=900)
    __indexes__ = [EmbeddingIndex(title, string_embed=marengo, name="title_marengo")]


class VideoScenes(
    TableModel,
    name="video_scenes",
    base=Videos,
    iterator=pxtf.video.video_splitter(
        video=Videos.video, segment_times=Videos.scenes[1:].start_time, mode="fast"
    ),
):
    __indexes__ = [
        EmbeddingIndex(video_segment, embedding=marengo, name="scene_marengo")  # type: ignore[name-defined]
    ]
