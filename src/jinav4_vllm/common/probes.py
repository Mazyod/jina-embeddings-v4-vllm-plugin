"""Probe inputs (not a benchmark corpus): enough to exercise text + image paths."""
from __future__ import annotations
from dataclasses import dataclass

IMAGE_TEMPLATE = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "Describe the image.<|im_end|>\n"
)


@dataclass(frozen=True)
class TextProbe:
    id: str
    text: str
    kind: str          # "query" | "passage"
    lang: str


@dataclass(frozen=True)
class ImageProbe:
    id: str
    path: str


def build_text_prompt(text: str, kind: str) -> str:
    prefix = {"query": "Query", "passage": "Passage"}[kind]
    return f"{prefix}: {text}"


def build_image_prompt() -> str:
    return IMAGE_TEMPLATE


TEXT_PROBES: list[TextProbe] = [
    TextProbe("text_query_en_short", "Overview of climate change impacts on coastal cities", "query", "en"),
    TextProbe("text_passage_en_long",
              "The impacts of climate change on coastal cities are significant and far-reaching. "
              "Rising sea levels threaten infrastructure, increase flooding frequency, and force "
              "costly adaptation measures across transport, housing, and water systems. Storm "
              "surges compound the damage, while saltwater intrusion degrades freshwater supplies "
              "and agricultural land in low-lying delta regions worldwide.", "passage", "en"),
    TextProbe("text_query_ar", "تأثير تغير المناخ على المدن الساحلية", "query", "ar"),
    TextProbe("text_query_ja", "浜辺に沈む美しい夕日", "query", "ja"),
    TextProbe("text_passage_symbols", "Δ-encoding: cost ≈ $1,234.56 (≤2% error) — see §3.1 & Fig. 2.", "passage", "en"),
]

IMAGE_PROBES: list[ImageProbe] = [
    ImageProbe("image_cat", "data/probes/cat.png"),
    ImageProbe("image_chart", "data/probes/chart.png"),
]
