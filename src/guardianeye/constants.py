"""Shared labels. Empty images are negatives, not a box class."""

BOX_CLASSES: tuple[str, ...] = ("human", "vehicle", "animal")
IMAGE_CLASSES: tuple[str, ...] = ("human", "vehicle", "animal", "empty")
AUDIO_LABELS: tuple[str, ...] = ("background", "gunshot")
DOMAINS: tuple[str, ...] = ("aerial_thermal", "camera_trap_rgb")
SITES: tuple[str, ...] = ("belize", "vietnam")
VISUAL_INDICATORS: tuple[str, ...] = ("human", "vehicle")

# COCO names used only if a demo checkpoint with those class names is loaded.
COCO_TO_GUARDIAN: dict[str, str] = {
    "person": "human",
    "bicycle": "vehicle",
    "car": "vehicle",
    "motorcycle": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "boat": "vehicle",
    "bird": "animal",
    "cat": "animal",
    "dog": "animal",
    "horse": "animal",
    "sheep": "animal",
    "cow": "animal",
    "elephant": "animal",
    "bear": "animal",
    "zebra": "animal",
    "giraffe": "animal",
}
