#!/usr/bin/env python3
"""A small, keyboard-first dataset triage tool built with PyQt6.

Usage:
    python dataset_rlhf.py /path/to/dataset good bad trash

Images are moved from the dataset root into one of the category directories.
Press the category number again (or Backspace) to move the current image back
to the dataset root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import (
    QEasingCurve,
    QObject,
    QPointF,
    QRectF,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    QVariantAnimation,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QIcon,
    QImage,
    QImageReader,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


IMAGE_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

CATEGORY_COLORS = [
    "#6ee7b7",
    "#60a5fa",
    "#fbbf24",
    "#f472b6",
    "#a78bfa",
    "#fb7185",
    "#34d399",
    "#22d3ee",
    "#facc15",
    "#fb923c",
]

MAX_DECODE_DIMENSION = 3200
MAX_CACHED_IMAGES = 12
FILTER_ALL = -2
FILTER_UNCLASSIFIED = -1


def supported_extensions() -> set[str]:
    formats = {
        "." + bytes(fmt).decode("ascii", errors="ignore").lower()
        for fmt in QImageReader.supportedImageFormats()
    }
    return IMAGE_EXTENSIONS | formats


def natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name.casefold())
    return [int(part) if part.isdigit() else part for part in parts]


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size} B"


def rgba(color: QColor | str, alpha: int) -> str:
    c = QColor(color)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {alpha})"


def category_color(index: int) -> QColor:
    return QColor(CATEGORY_COLORS[index % len(CATEGORY_COLORS)])


def resolve_category_dir(dataset_dir: Path, category: str) -> Path:
    path = Path(category).expanduser()
    if not path.is_absolute():
        path = dataset_dir / path
    return path.resolve()


def session_state_path(dataset_dir: Path) -> Path:
    resolved = dataset_dir.expanduser().resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^\w.-]+", "_", resolved.name, flags=re.UNICODE).strip("._-")
    filename = f"dataset_rlhf_{slug[:48] or 'dataset'}_{digest}.json"
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base_dir = (Path(cache_home) if cache_home else Path.home() / ".cache") / "dataset_rlhf"
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / filename
    except OSError:
        return Path(tempfile.gettempdir()) / filename


def sidecar_mappers() -> tuple[Callable[[Path], Path], ...]:
    return (
        lambda p: p.with_suffix(".txt"),
        lambda p: p.with_name(p.name + ".txt"),
        lambda p: p.with_name(p.stem + ".mask.png"),
        lambda p: p.with_suffix(".json"),
        lambda p: p.with_name(p.name + ".json"),
        lambda p: p.with_suffix(".caption"),
        lambda p: p.with_name(p.name + ".caption"),
    )


def sidecar_paths(image_path: Path) -> list[Path]:
    paths: list[Path] = []
    for mapper in sidecar_mappers():
        path = mapper(image_path)
        if path != image_path and path not in paths:
            paths.append(path)
    return paths


def unique_image_destination(destination: Path) -> Path:
    """Return a collision-free image path, including its potential sidecars."""
    if not destination.exists() and not any(p.exists() for p in sidecar_paths(destination)):
        return destination

    index = 1
    while True:
        candidate = destination.with_name(f"{destination.stem}_{index}{destination.suffix}")
        if not candidate.exists() and not any(p.exists() for p in sidecar_paths(candidate)):
            return candidate
        index += 1


@dataclass
class ImageRecord:
    uid: int
    original_name: str
    path: Path
    category: int | None = None


class LoadSignals(QObject):
    loaded = pyqtSignal(int, QImage, int, int)
    failed = pyqtSignal(int, str)


class LoadTask(QRunnable):
    def __init__(self, uid: int, path: Path, signals: LoadSignals) -> None:
        super().__init__()
        self.uid = uid
        self.path = path
        self.signals = signals
        self.setAutoDelete(True)

    def run(self) -> None:
        reader = QImageReader(str(self.path))
        reader.setAutoTransform(True)
        size = reader.size()
        original_width = size.width() if size.isValid() else 0
        original_height = size.height() if size.isValid() else 0

        if size.isValid() and max(size.width(), size.height()) > MAX_DECODE_DIMENSION:
            reader.setScaledSize(
                size.scaled(
                    QSize(MAX_DECODE_DIMENSION, MAX_DECODE_DIMENSION),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
            )

        image = reader.read()
        if image.isNull():
            self.signals.failed.emit(self.uid, reader.errorString() or "未知图片读取错误")
            return
        self.signals.loaded.emit(self.uid, image, original_width, original_height)


class ElideLabel(QLabel):
    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full_text = text
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setToolTip(text)

    def setFullText(self, text: str) -> None:
        self._full_text = text
        self.setToolTip(text)
        self._refresh()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        width = max(0, self.width() - 2)
        super().setText(
            QFontMetrics(self.font()).elidedText(
                self._full_text, Qt.TextElideMode.ElideMiddle, width
            )
        )


class CategoryCard(QFrame):
    clicked = pyqtSignal(int)

    def __init__(
        self,
        index: int,
        key_label: str,
        label: str,
        destination: Path,
        color: QColor,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.index = index
        self._color = color
        self._active = False
        self._count = 0
        self._ratio = 0.0
        self.setObjectName("categoryCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"将当前图片移动到 {destination}")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(11, 10, 12, 10)
        layout.setSpacing(10)

        self.key_badge = QLabel(key_label)
        self.key_badge.setObjectName("keyBadge")
        self.key_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.key_badge.setFixedSize(28, 28)
        self.key_badge.setStyleSheet(
            f"background: {rgba(color, 35)}; color: {color.name()};"
            f"border: 1px solid {rgba(color, 75)}; border-radius: 9px;"
            "font-weight: 700; font-size: 12px;"
        )
        layout.addWidget(self.key_badge)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(1)
        self.name_label = ElideLabel(label)
        self.name_label.setObjectName("categoryName")
        self.path_label = ElideLabel(str(destination))
        self.path_label.setObjectName("categoryPath")
        text_layout.addWidget(self.name_label)
        text_layout.addWidget(self.path_label)
        layout.addLayout(text_layout, 1)

        self.count_label = QLabel("0")
        self.count_label.setObjectName("categoryCount")
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.count_label)

        for child in (self.key_badge, self.name_label, self.path_label, self.count_label):
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def set_count(self, count: int, ratio: float = 0.0) -> None:
        self._count = count
        self._ratio = ratio
        if self._active:
            self.count_label.setText(f"当前 · {ratio:.0f}%")
            self.count_label.setStyleSheet(f"color: {self._color.name()}; font-weight: 600;")
        else:
            self.count_label.setText(f"{count} · {ratio:.0f}%")
            self.count_label.setStyleSheet("")

    def set_active(self, active: bool) -> None:
        self._active = active
        if active:
            self.setStyleSheet(
                f"QFrame#categoryCard {{ background: {rgba(self._color, 18)};"
                f" border: 1px solid {rgba(self._color, 120)}; border-radius: 14px; }}"
            )
        else:
            self.setStyleSheet("")
        self.set_count(self._count, self._ratio)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(
            event.position().toPoint()
        ):
            self.clicked.emit(self.index)
        super().mouseReleaseEvent(event)


class ClassificationRatioBar(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(10)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._counts: list[int] = []
        self._denominator = 0

    def set_data(self, denominator: int, counts: list[int]) -> None:
        self._denominator = denominator
        self._counts = counts
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#202731"))
        painter.drawRoundedRect(rect, 5, 5)
        if self._denominator <= 0 or not self._counts:
            return

        clip = QPainterPath()
        clip.addRoundedRect(rect, 5, 5)
        painter.setClipPath(clip)
        x = rect.x()
        for index, count in enumerate(self._counts):
            if count <= 0:
                continue
            width = rect.width() * count / self._denominator
            painter.fillRect(QRectF(x, rect.y(), width + 0.35, rect.height()), category_color(index))
            x += width


class ClassificationPositionSlider(QSlider):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setObjectName("positionSlider")
        self.setFixedHeight(28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("蓝色：已分类；红色：未分类")
        self._classified = 0
        self._total = 0

    def set_classification_state(self, classified: int, total: int) -> None:
        self._classified = classified
        self._total = total
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if not self.isEnabled():
            painter.setOpacity(0.42)

        margin = 7.0
        track_height = 6.0
        track = QRectF(
            margin,
            (self.height() - track_height) / 2.0,
            max(0.0, self.width() - margin * 2.0),
            track_height,
        )
        painter.setPen(Qt.PenStyle.NoPen)

        if self._total > 0:
            clip = QPainterPath()
            clip.addRoundedRect(track, 3, 3)
            painter.save()
            painter.setClipPath(clip)
            painter.fillRect(track, QColor("#ef4444"))
            classified_width = track.width() * self._classified / self._total
            painter.fillRect(
                QRectF(track.x(), track.y(), classified_width, track.height()),
                QColor("#3b82f6"),
            )
            painter.restore()
        else:
            painter.setBrush(QColor("#303846"))
            painter.drawRoundedRect(track, 3, 3)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 24), 1))
        painter.drawRoundedRect(track, 3, 3)

        span = self.maximum() - self.minimum()
        fraction = (self.value() - self.minimum()) / span if span > 0 else 0.0
        handle_x = track.left() + track.width() * fraction
        painter.setPen(QPen(QColor("#0a0d12"), 2))
        painter.setBrush(QColor("#f4f7fb"))
        painter.drawEllipse(QPointF(handle_x, track.center().y()), 6.5, 6.5)

    def _value_at(self, x: float) -> int:
        margin = 7.0
        width = max(1.0, self.width() - margin * 2.0)
        fraction = min(1.0, max(0.0, (x - margin) / width))
        span = self.maximum() - self.minimum()
        return round(self.minimum() + span * fraction)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self.setSliderDown(True)
            self.setValue(self._value_at(event.position().x()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.isSliderDown():
            self.setValue(self._value_at(event.position().x()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.isSliderDown():
            self.setValue(self._value_at(event.position().x()))
            self.setSliderDown(False)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class FilterComboBox(QComboBox):
    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor("#8e98a5"), 1.7)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        center_x = self.width() - 15.0
        center_y = self.height() / 2.0
        painter.drawLine(
            QPointF(center_x - 4, center_y - 2),
            QPointF(center_x, center_y + 2),
        )
        painter.drawLine(
            QPointF(center_x, center_y + 2),
            QPointF(center_x + 4, center_y - 2),
        )


class ImageCanvas(QWidget):
    wheel_requested = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("imageCanvas")
        self.setMinimumSize(420, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._pixmap: QPixmap | None = None
        self._previous_pixmap: QPixmap | None = None
        self._fade = 1.0
        self._loading = False
        self._error = ""
        self._empty_text = "目录中未找到图片"
        self._stamp_index: int | None = None
        self._stamp_name = ""
        self._stamp_color = QColor(CATEGORY_COLORS[0])
        self._stamp_scale = 1.0
        self._wheel_accumulator = 0

        self._fade_animation = QVariantAnimation(self)
        self._fade_animation.setDuration(170)
        self._fade_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade_animation.valueChanged.connect(self._on_fade_changed)

        self._stamp_animation = QVariantAnimation(self)
        self._stamp_animation.setDuration(320)
        self._stamp_animation.setEasingCurve(QEasingCurve.Type.OutBack)
        self._stamp_animation.valueChanged.connect(self._on_stamp_scale_changed)

    def set_pixmap(self, pixmap: QPixmap, animate: bool = True) -> None:
        self._loading = False
        self._error = ""
        if self._pixmap is None or not animate or self._pixmap.cacheKey() == pixmap.cacheKey():
            self._pixmap = pixmap
            self._previous_pixmap = None
            self._fade = 1.0
        else:
            self._previous_pixmap = self._pixmap
            self._pixmap = pixmap
            self._fade = 0.0
            self._fade_animation.stop()
            self._fade_animation.setStartValue(0.0)
            self._fade_animation.setEndValue(1.0)
            self._fade_animation.start()
        self.update()

    def clear_image(self) -> None:
        self._pixmap = None
        self._previous_pixmap = None
        self._loading = False
        self._error = ""
        self.update()

    def set_loading(self, loading: bool) -> None:
        self._loading = loading
        self.update()

    def set_error(self, message: str) -> None:
        self._loading = False
        self._error = message
        self._pixmap = None
        self._previous_pixmap = None
        self._fade = 1.0
        self.update()

    def set_empty_text(self, text: str) -> None:
        self._empty_text = text
        self.update()

    def set_stamp(self, category_index: int | None, name: str = "") -> None:
        self._stamp_index = category_index
        self._stamp_name = name
        if category_index is not None:
            self._stamp_color = category_color(category_index)
            self._stamp_animation.stop()
            self._stamp_animation.setStartValue(0.72)
            self._stamp_animation.setEndValue(1.0)
            self._stamp_animation.start()
        self.update()

    def _on_fade_changed(self, value: object) -> None:
        self._fade = float(value)
        self.update()

    def _on_stamp_scale_changed(self, value: object) -> None:
        self._stamp_scale = float(value)
        self.update()

    def _image_target_rect(self, pixmap: QPixmap) -> QRectF:
        area = QRectF(self.rect()).adjusted(26, 26, -26, -26)
        if pixmap.isNull() or area.width() <= 0 or area.height() <= 0:
            return QRectF()
        scaled = pixmap.size()
        scaled.scale(QSize(int(area.width()), int(area.height())), Qt.AspectRatioMode.KeepAspectRatio)
        x = area.x() + (area.width() - scaled.width()) / 2.0
        y = area.y() + (area.height() - scaled.height()) / 2.0
        return QRectF(x, y, scaled.width(), scaled.height())

    def _draw_pixmap(self, painter: QPainter, pixmap: QPixmap, rect: QRectF) -> None:
        if rect.isEmpty():
            return
        clip = QPainterPath()
        clip.addRoundedRect(rect, 14, 14)
        painter.save()
        painter.setClipPath(clip)

        checker = 18
        painter.fillRect(rect, QColor("#0a0c0f"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#111419"))
        rows = int(rect.height() // checker) + 2
        cols = int(rect.width() // checker) + 2
        for row in range(rows):
            for col in range(cols):
                if (row + col) % 2 == 0:
                    painter.drawRect(QRectF(
                        rect.x() + col * checker,
                        rect.y() + row * checker,
                        checker,
                        checker,
                    ))

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(rect, pixmap, QRectF(pixmap.rect()))
        painter.restore()

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 28), 1))
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 14, 14)

    def _draw_stamp(self, painter: QPainter, image_rect: QRectF) -> None:
        if self._stamp_index is None or image_rect.isEmpty():
            return

        key = str(self._stamp_index + 1) if self._stamp_index < 9 else (
            "0" if self._stamp_index == 9 else "·"
        )
        title_font = QFont(self.font())
        title_font.setPointSizeF(max(10.0, self.font().pointSizeF() * 0.88))
        title_font.setWeight(QFont.Weight.DemiBold)
        title_metrics = QFontMetrics(title_font)
        name_font = QFont(self.font())
        name_font.setPointSizeF(max(11.0, self.font().pointSizeF() * 1.03))
        name_font.setWeight(QFont.Weight.Bold)
        name_metrics = QFontMetrics(name_font)

        title = "✓ 已分类"
        subtitle = f"{key}  {self._stamp_name}"
        content_width = max(title_metrics.horizontalAdvance(title), name_metrics.horizontalAdvance(subtitle))
        width = min(max(158.0, content_width + 48.0), max(170.0, image_rect.width() * 0.48))
        height = 68.0
        margin = min(28.0, max(12.0, image_rect.width() * 0.025))
        stamp_rect = QRectF(
            image_rect.right() - width - margin,
            image_rect.top() + margin,
            width,
            height,
        )

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        center = stamp_rect.center()
        painter.translate(center)
        painter.rotate(-6.5)
        painter.scale(self._stamp_scale, self._stamp_scale)
        painter.translate(-center)

        shadow = stamp_rect.translated(0, 5)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 85))
        painter.drawRoundedRect(shadow, 12, 12)

        painter.setBrush(QColor(13, 16, 21, 224))
        painter.setPen(QPen(QColor(self._stamp_color.red(), self._stamp_color.green(), self._stamp_color.blue(), 205), 2))
        painter.drawRoundedRect(stamp_rect, 12, 12)

        inner = stamp_rect.adjusted(5, 5, -5, -5)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(self._stamp_color.red(), self._stamp_color.green(), self._stamp_color.blue(), 70), 1))
        painter.drawRoundedRect(inner, 9, 9)

        painter.setPen(self._stamp_color)
        painter.setFont(title_font)
        painter.drawText(
            QRectF(stamp_rect.x() + 16, stamp_rect.y() + 10, stamp_rect.width() - 32, 18),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            title,
        )
        painter.setPen(QColor("#f4f7fb"))
        painter.setFont(name_font)
        painter.drawText(
            QRectF(stamp_rect.x() + 16, stamp_rect.y() + 31, stamp_rect.width() - 32, 25),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            name_metrics.elidedText(
                subtitle, Qt.TextElideMode.ElideRight, int(stamp_rect.width() - 32)
            ),
        )
        painter.restore()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor("#0d1117"))
        gradient.setColorAt(1.0, QColor("#090b0e"))
        painter.fillRect(self.rect(), gradient)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 8))
        spacing = 36
        for y in range(spacing, self.height(), spacing):
            for x in range(spacing, self.width(), spacing):
                painter.drawEllipse(QPointF(x, y), 0.8, 0.8)

        if self._pixmap is None:
            title = self._empty_text if not self._error else "图片读取失败"
            subtitle = self._error
            painter.setPen(QColor("#8b93a1"))
            title_font = QFont(self.font())
            title_font.setPointSizeF(self.font().pointSizeF() * 1.18)
            title_font.setWeight(QFont.Weight.DemiBold)
            painter.setFont(title_font)
            painter.drawText(
                self.rect().adjusted(40, 40, -40, -52),
                int(Qt.AlignmentFlag.AlignCenter),
                title,
            )
            if subtitle:
                painter.setFont(self.font())
                painter.setPen(QColor("#626b78"))
                painter.drawText(
                    self.rect().adjusted(60, 74, -60, -30),
                    int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
                    subtitle,
                )
            return

        if self._previous_pixmap is not None and self._fade < 1.0:
            old_rect = self._image_target_rect(self._previous_pixmap)
            painter.setOpacity(1.0 - self._fade)
            self._draw_pixmap(painter, self._previous_pixmap, old_rect)
            painter.setOpacity(self._fade)
            self._draw_pixmap(painter, self._pixmap, self._image_target_rect(self._pixmap))
            painter.setOpacity(1.0)
        else:
            self._draw_pixmap(painter, self._pixmap, self._image_target_rect(self._pixmap))

        self._draw_stamp(painter, self._image_target_rect(self._pixmap))

        if self._loading:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            pill = QRectF(self.width() / 2 - 52, self.height() - 60, 104, 30)
            painter.setPen(QPen(QColor(255, 255, 255, 24), 1))
            painter.setBrush(QColor(16, 20, 26, 225))
            painter.drawRoundedRect(pill, 15, 15)
            painter.setPen(QColor("#a8b0bd"))
            painter.drawText(pill, int(Qt.AlignmentFlag.AlignCenter), "正在加载")

    def wheelEvent(self, event: QWheelEvent) -> None:
        self._wheel_accumulator += event.angleDelta().y()
        if abs(self._wheel_accumulator) >= 120:
            direction = 1 if self._wheel_accumulator < 0 else -1
            self._wheel_accumulator = 0
            self.wheel_requested.emit(direction)
            event.accept()
            return
        event.ignore()


class RankingConfigWidget(QWidget):
    ranking_confirmed = pyqtSignal(list)

    def __init__(
        self,
        categories: list[str],
        initial_ranking: list[int] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.categories = categories
        self.category_count = len(categories)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title = QLabel("配置类别排序")
        title.setObjectName("sideTitle")
        layout.addWidget(title)

        desc = QLabel("排在前面的类别代表质量更优（从高到低）。\n支持拖拽或使用按钮调整顺序：")
        desc.setObjectName("sideSubtitle")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        self.list_widget = QListWidget()
        self.list_widget.setObjectName("rankingList")
        self.list_widget.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list_widget.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        ranking = initial_ranking
        if (
            ranking is None
            or len(ranking) != self.category_count
            or set(ranking) != set(range(self.category_count))
        ):
            ranking = list(range(self.category_count))

        for idx in ranking:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, idx)
            self.list_widget.addItem(item)

        self.list_widget.model().rowsMoved.connect(self._refresh_display)
        layout.addWidget(self.list_widget, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.up_btn = QPushButton("▲ 上移")
        self.up_btn.setObjectName("secondaryButton")
        self.up_btn.clicked.connect(self._move_current_up)
        self.down_btn = QPushButton("▼ 下移")
        self.down_btn.setObjectName("secondaryButton")
        self.down_btn.clicked.connect(self._move_current_down)
        btn_row.addWidget(self.up_btn)
        btn_row.addWidget(self.down_btn)
        layout.addLayout(btn_row)

        self.preview_label = QLabel("")
        self.preview_label.setObjectName("decision")
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)

        self.confirm_btn = QPushButton("🚀 确认并开始校验")
        self.confirm_btn.setObjectName("navButton")
        self.confirm_btn.setStyleSheet(
            "font-size: 13px; font-weight: 700; min-height: 38px; background: #2563eb; color: #ffffff; border: none; border-radius: 11px;"
        )
        self.confirm_btn.clicked.connect(self._on_confirm)
        layout.addWidget(self.confirm_btn)

        self._refresh_display()

    def get_current_ranking(self) -> list[int]:
        ranking: list[int] = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            idx = item.data(Qt.ItemDataRole.UserRole)
            ranking.append(idx)
        return ranking

    def set_ranking(self, ranking: list[int]) -> None:
        if len(ranking) != self.category_count or set(ranking) != set(range(self.category_count)):
            return
        self.list_widget.clear()
        for idx in ranking:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, idx)
            self.list_widget.addItem(item)
        self._refresh_display()

    def _refresh_display(self) -> None:
        ranking = self.get_current_ranking()
        for pos, idx in enumerate(ranking):
            item = self.list_widget.item(pos)
            color = category_color(idx)
            cat_name = self.categories[idx]
            prefix = "★ 最高" if pos == 0 else ("✘ 最低" if pos == len(ranking) - 1 else f"  第 {pos + 1} 档")
            item.setText(f" {prefix}  ·  {cat_name}")
            item.setForeground(color)

        chain_str = " > ".join(self.categories[i] for i in ranking)
        self.preview_label.setText(f"当前规则（优 > 劣）：\n{chain_str}")

    def _move_current_up(self) -> None:
        row = self.list_widget.currentRow()
        if row > 0:
            item = self.list_widget.takeItem(row)
            self.list_widget.insertItem(row - 1, item)
            self.list_widget.setCurrentRow(row - 1)
            self._refresh_display()

    def _move_current_down(self) -> None:
        row = self.list_widget.currentRow()
        if 0 <= row < self.list_widget.count() - 1:
            item = self.list_widget.takeItem(row)
            self.list_widget.insertItem(row + 1, item)
            self.list_widget.setCurrentRow(row + 1)
            self._refresh_display()

    def _on_confirm(self) -> None:
        ranking = self.get_current_ranking()
        self.ranking_confirmed.emit(ranking)


class CompareControlWidget(QWidget):
    reconfigure_requested = pyqtSignal()
    skip_requested = pyqtSignal()

    def __init__(self, categories: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.categories = categories
        self._ranking: list[int] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title = QLabel("校验控制")
        title.setObjectName("sideTitle")
        subtitle = QLabel("盲选对决：选择前不显示真实类别")
        subtitle.setObjectName("sideSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        rule_box = QVBoxLayout()
        rule_box.setSpacing(4)
        rule_label = QLabel("当前优劣规则：")
        rule_label.setObjectName("ratioTitle")
        self.rule_text = QLabel("未设定")
        self.rule_text.setObjectName("decision")
        self.rule_text.setWordWrap(True)
        rule_box.addWidget(rule_label)
        rule_box.addWidget(self.rule_text)
        layout.addLayout(rule_box)

        self.reconfig_btn = QPushButton("⚙️ 修改类别排序规则")
        self.reconfig_btn.setObjectName("secondaryButton")
        self.reconfig_btn.clicked.connect(self.reconfigure_requested.emit)
        layout.addWidget(self.reconfig_btn)

        stats_box = QVBoxLayout()
        stats_box.setSpacing(6)
        stats_title = QLabel("校验统计")
        stats_title.setObjectName("sideTitle")
        stats_title.setStyleSheet("font-size: 13px; font-weight: 600; padding-top: 4px;")
        stats_box.addWidget(stats_title)

        self.stats_summary = QLabel(
            "总比对: 0 次\n一致率: 0.0%\n(✓ 一致: 0 · ⚠️ 交换: 0)\n(🌟 俱佳: 0 · 🗑️ 俱劣: 0)"
        )
        self.stats_summary.setObjectName("decision")
        stats_box.addWidget(self.stats_summary)
        layout.addLayout(stats_box)

        log_box = QVBoxLayout()
        log_box.setSpacing(4)
        log_title = QLabel("近期对决动态")
        log_title.setObjectName("ratioTitle")
        log_box.addWidget(log_title)
        self.log_text = QLabel("尚未开始比对")
        self.log_text.setObjectName("decision")
        self.log_text.setStyleSheet("color: #8b96a5; font-size: 10px; line-height: 1.4;")
        self.log_text.setWordWrap(True)
        log_box.addWidget(self.log_text)
        layout.addLayout(log_box)

        layout.addStretch(1)

        self.skip_btn = QPushButton("🔄 换一组 / 跳过 [空格]")
        self.skip_btn.setObjectName("secondaryButton")
        self.skip_btn.clicked.connect(self.skip_requested.emit)
        layout.addWidget(self.skip_btn)

    def set_ranking(self, ranking: list[int]) -> None:
        self._ranking = ranking
        chain = " > ".join(self.categories[i] for i in ranking)
        self.rule_text.setText(chain)

    def update_stats(
        self,
        total: int,
        consistent: int,
        conflict: int,
        both_good: int,
        both_bad: int,
        logs: list[str],
    ) -> None:
        rate = (consistent / total * 100.0) if total > 0 else 0.0
        self.stats_summary.setText(
            f"总比对: {total} 次  (一致率: {rate:.1f}%)\n"
            f"✓ 一致: {consistent} · ⚠️ 交换: {conflict}\n"
            f"🌟 俱佳: {both_good} · 🗑️ 俱劣: {both_bad}"
        )
        if logs:
            self.log_text.setText("\n".join(logs[-6:]))
        else:
            self.log_text.setText("尚未开始比对")


class DatasetRLHFWindow(QMainWindow):
    def __init__(self, dataset_dir: Path, categories: list[str]) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir.expanduser().resolve()
        self.category_dirs = [
            resolve_category_dir(self.dataset_dir, category) for category in categories
        ]
        self.categories = [
            directory.name or str(directory) for directory in self.category_dirs
        ]
        self.extensions = supported_extensions()
        self.records: list[ImageRecord] = []
        self.visible_records: list[ImageRecord] = []
        self.filter_mode = FILTER_ALL
        self.current_index = -1
        self.pixmaps: OrderedDict[int, QPixmap] = OrderedDict()
        self.image_sizes: dict[int, tuple[int, int]] = {}
        self.loading_uids: set[int] = set()
        self.failed_uids: dict[int, str] = {}
        self.failed_paths: dict[int, Path] = {}
        self.tasks: dict[int, LoadTask] = {}
        self.category_counts = [0] * len(categories)
        self.state_path = session_state_path(self.dataset_dir)
        self._category_index_by_path = {
            str(directory): index for index, directory in enumerate(self.category_dirs)
        }
        self._restored_decisions: list[tuple[str, Path, int]] = []
        self._carryover_decisions: list[dict[str, str]] = []
        self._resume_original_name: str | None = None
        self.category_ranking: list[int] | None = None
        self.compare_total_count = 0
        self.compare_consistent_count = 0
        self.compare_conflict_count = 0
        self.compare_both_good_count = 0
        self.compare_both_bad_count = 0
        self.compare_left_record: ImageRecord | None = None
        self.compare_right_record: ImageRecord | None = None
        self.compare_recent_logs: list[str] = []
        self._load_state()

        self.thread_pool = QThreadPool(self)
        self.thread_pool.setMaxThreadCount(2)
        self.load_signals = LoadSignals(self)
        self.load_signals.loaded.connect(self._on_image_loaded)
        self.load_signals.failed.connect(self._on_image_failed)

        self.preload_timer = QTimer(self)
        self.preload_timer.setSingleShot(True)
        self.preload_timer.setInterval(110)
        self.preload_timer.timeout.connect(self._preload_neighbors)

        self.setWindowTitle(f"Dataset RLHF · {self.dataset_dir.name}")
        self.setMinimumSize(940, 640)
        self._build_ui()
        self._prepare_directories()
        self._scan_dataset()
        self._refresh_stats()
        self._apply_filter(resume_original=self._resume_original_name)
        self._save_state()
        self._center_on_screen()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(20, 20, 20, 22)
        outer.setSpacing(0)

        body = QHBoxLayout()
        body.setSpacing(16)

        # ------------------ 中央面板 1：单图分类视图 ------------------
        viewer_panel = QFrame()
        viewer_panel.setObjectName("panel")
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(14, 12, 14, 14)
        viewer_layout.setSpacing(10)

        info_row = QHBoxLayout()
        info_row.setContentsMargins(6, 2, 6, 0)
        info_row.setSpacing(12)
        self.filename_label = ElideLabel("")
        self.filename_label.setObjectName("filename")
        self.meta_label = QLabel("")
        self.meta_label.setObjectName("meta")
        self.meta_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        info_row.addWidget(self.filename_label, 1)
        info_row.addWidget(self.meta_label)
        viewer_layout.addLayout(info_row)

        self.canvas = ImageCanvas()
        self.canvas.wheel_requested.connect(self._wheel_navigate)
        viewer_layout.addWidget(self.canvas, 1)

        footer = QHBoxLayout()
        footer.setSpacing(10)
        self.prev_button = QPushButton("‹")
        self.prev_button.setObjectName("navButton")
        self.prev_button.setToolTip("上一张（← / A / PageUp）")
        self.prev_button.clicked.connect(lambda: self.go_to(self.current_index - 1))

        self.position_label = QLabel("0 / 0")
        self.position_label.setObjectName("position")
        self.position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.position_label.setMinimumWidth(76)

        self.slider = ClassificationPositionSlider()
        self.slider.setRange(0, 0)
        self.slider.setTracking(False)
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderMoved.connect(self._on_slider_moved)

        self.next_button = QPushButton("›")
        self.next_button.setObjectName("navButton")
        self.next_button.setToolTip("下一张（→ / D / PageDown / 空格）")
        self.next_button.clicked.connect(lambda: self.go_to(self.current_index + 1))

        footer.addWidget(self.prev_button)
        footer.addWidget(self.position_label)
        footer.addWidget(self.slider, 1)
        footer.addWidget(self.next_button)
        viewer_layout.addLayout(footer)

        hint = QLabel("蓝 = 已分类 · 红 = 未分类    滚轮 / ← → 翻页    数字键分类    Backspace 移回")
        hint.setObjectName("hint")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        viewer_layout.addWidget(hint)

        # ------------------ 中央面板 2：双图对比校验视图 ------------------
        self.compare_panel = QFrame()
        self.compare_panel.setObjectName("panel")
        compare_layout = QVBoxLayout(self.compare_panel)
        compare_layout.setContentsMargins(14, 12, 14, 14)
        compare_layout.setSpacing(10)

        # 顶部提示状态栏
        self.compare_status_bar = QLabel(
            "👀 盲选对决：类别信息已隐藏。请仅凭视觉质量判断，按 [A / ←] 选左图，按 [D / →] 选右图"
        )
        self.compare_status_bar.setObjectName("decision")
        self.compare_status_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.compare_status_bar.setStyleSheet(
            "background: #141920; border: 1px solid #2b3542; border-radius: 11px; "
            "color: #a5b4fc; font-weight: 600; font-size: 12px; padding: 8px 12px;"
        )
        compare_layout.addWidget(self.compare_status_bar)

        # 中间双图并排
        vs_layout = QHBoxLayout()
        vs_layout.setSpacing(12)

        # 左卡片 (图 A)
        left_card = QFrame()
        left_card.setObjectName("categoryCard")
        left_card_layout = QVBoxLayout(left_card)
        left_card_layout.setContentsMargins(10, 10, 10, 10)
        left_card_layout.setSpacing(8)

        left_header = QHBoxLayout()
        left_header.setSpacing(8)
        self.left_badge = QLabel("图 A")
        self.left_badge.setObjectName("keyBadge")
        self.left_badge.setFixedSize(85, 25)
        self.left_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_filename = ElideLabel("")
        self.left_filename.setObjectName("filename")
        left_header.addWidget(self.left_badge)
        left_header.addWidget(self.left_filename, 1)
        left_card_layout.addLayout(left_header)

        self.canvas_left = ImageCanvas()
        left_card_layout.addWidget(self.canvas_left, 1)

        self.choose_left_btn = QPushButton("👈 左图更好 [A / ←]")
        self.choose_left_btn.setObjectName("navButton")
        self.choose_left_btn.setStyleSheet(
            "font-size: 13px; font-weight: 700; min-height: 42px; background: #1a2332; "
            "color: #38bdf8; border: 1px solid #2563eb; border-radius: 11px;"
        )
        self.choose_left_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.choose_left_btn.clicked.connect(lambda: self._handle_compare_choice("left"))
        left_card_layout.addWidget(self.choose_left_btn)
        vs_layout.addWidget(left_card, 1)

        # 中间 VS 分隔
        vs_label = QLabel("VS")
        vs_label.setStyleSheet("font-size: 22px; font-weight: 900; color: #475569;")
        vs_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vs_layout.addWidget(vs_label)

        # 右卡片 (图 B)
        right_card = QFrame()
        right_card.setObjectName("categoryCard")
        right_card_layout = QVBoxLayout(right_card)
        right_card_layout.setContentsMargins(10, 10, 10, 10)
        right_card_layout.setSpacing(8)

        right_header = QHBoxLayout()
        right_header.setSpacing(8)
        self.right_badge = QLabel("图 B")
        self.right_badge.setObjectName("keyBadge")
        self.right_badge.setFixedSize(85, 25)
        self.right_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_filename = ElideLabel("")
        self.right_filename.setObjectName("filename")
        right_header.addWidget(self.right_badge)
        right_header.addWidget(self.right_filename, 1)
        right_card_layout.addLayout(right_header)

        self.canvas_right = ImageCanvas()
        right_card_layout.addWidget(self.canvas_right, 1)

        self.choose_right_btn = QPushButton("👉 右图更好 [D / →]")
        self.choose_right_btn.setObjectName("navButton")
        self.choose_right_btn.setStyleSheet(
            "font-size: 13px; font-weight: 700; min-height: 42px; background: #1a2332; "
            "color: #38bdf8; border: 1px solid #2563eb; border-radius: 11px;"
        )
        self.choose_right_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.choose_right_btn.clicked.connect(lambda: self._handle_compare_choice("right"))
        right_card_layout.addWidget(self.choose_right_btn)
        vs_layout.addWidget(right_card, 1)

        compare_layout.addLayout(vs_layout, 1)

        # 评判操作栏（都很不错 / 都是垃圾 / 跳过）
        action_row = QHBoxLayout()
        action_row.setSpacing(10)

        self.both_good_btn = QPushButton("🌟 都很不错 · 归入好类 [W / ↑]")
        self.both_good_btn.setObjectName("navButton")
        self.both_good_btn.setStyleSheet(
            "font-size: 12px; font-weight: 700; min-height: 40px; background: #064e3b; "
            "color: #6ee7b7; border: 1px solid #059669; border-radius: 11px;"
        )
        self.both_good_btn.setToolTip("两张图片均符合高标准，一同归入本次对比中更高的类别")
        self.both_good_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.both_good_btn.clicked.connect(lambda: self._handle_compare_choice("both_good"))

        self.both_bad_btn = QPushButton("🗑️ 都是垃圾 · 归入差类 [S / ↓]")
        self.both_bad_btn.setObjectName("navButton")
        self.both_bad_btn.setStyleSheet(
            "font-size: 12px; font-weight: 700; min-height: 40px; background: #4c0519; "
            "color: #fda4af; border: 1px solid #be123c; border-radius: 11px;"
        )
        self.both_bad_btn.setToolTip("两张图片均较差，一同归入本次对比中更差的类别")
        self.both_bad_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.both_bad_btn.clicked.connect(lambda: self._handle_compare_choice("both_bad"))

        self.compare_skip_btn = QPushButton("🔄 跳过 (空格)")
        self.compare_skip_btn.setObjectName("secondaryButton")
        self.compare_skip_btn.setToolTip("保持原样，直接换一组进行对比")
        self.compare_skip_btn.setMinimumHeight(40)
        self.compare_skip_btn.setMinimumWidth(100)
        self.compare_skip_btn.clicked.connect(self._sample_compare_pair)

        action_row.addWidget(self.both_good_btn, 1)
        action_row.addWidget(self.both_bad_btn, 1)
        action_row.addWidget(self.compare_skip_btn)
        compare_layout.addLayout(action_row)

        # 底部提示栏
        compare_footer = QHBoxLayout()
        compare_hint = QLabel(
            "快捷键：A 或 ← 选左图 · D 或 → 选右图 · W 或 ↑ 都很不错 · S 或 ↓ 都是垃圾 · 空格 跳过"
        )
        compare_hint.setObjectName("hint")
        compare_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        compare_footer.addWidget(compare_hint, 1)
        compare_layout.addLayout(compare_footer)

        # 中央 StackedWidget
        self.main_stack = QStackedWidget()
        self.main_stack.addWidget(viewer_panel)
        self.main_stack.addWidget(self.compare_panel)
        body.addWidget(self.main_stack, 1)

        # ------------------ 右侧选项卡栏 ------------------
        # Tab 0: 单图分类侧边栏
        triage_sidebar = QWidget()
        side_layout = QVBoxLayout(triage_sidebar)
        side_layout.setContentsMargins(12, 14, 12, 12)
        side_layout.setSpacing(12)

        side_title = QLabel("目标分类")
        side_title.setObjectName("sideTitle")
        side_subtitle = QLabel("按数字键将当前图片移入对应目录")
        side_subtitle.setObjectName("sideSubtitle")
        side_subtitle.setWordWrap(True)
        side_layout.addWidget(side_title)
        side_layout.addWidget(side_subtitle)

        ratio_header = QHBoxLayout()
        ratio_header.setSpacing(8)
        self.ratio_title = QLabel("分类比例")
        self.ratio_title.setObjectName("ratioTitle")
        self.ratio_summary = QLabel("已分类 0 / 0 · 0%")
        self.ratio_summary.setObjectName("ratioSummary")
        self.ratio_summary.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        ratio_header.addWidget(self.ratio_title)
        ratio_header.addWidget(self.ratio_summary, 1)
        self.ratio_bar = ClassificationRatioBar()
        state_tip = f"分类比例仅统计已分类样本；断点文件：{self.state_path}"
        self.ratio_title.setToolTip(state_tip)
        self.ratio_summary.setToolTip(state_tip)
        self.ratio_bar.setToolTip(state_tip)
        side_layout.addLayout(ratio_header)
        side_layout.addWidget(self.ratio_bar)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(8)
        filter_label = QLabel("显示")
        filter_label.setObjectName("filterLabel")
        self.filter_combo = FilterComboBox()
        self.filter_combo.setObjectName("filterCombo")
        self.filter_combo.setMinimumHeight(32)
        self.filter_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.filter_combo.addItem("全部样本", FILTER_ALL)
        self.filter_combo.addItem("未分类", FILTER_UNCLASSIFIED)
        for index, name in enumerate(self.categories):
            key_label = str(index + 1) if index < 9 else ("0" if index == 9 else "·")
            self.filter_combo.addItem(f"{key_label} · {name}", index)
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        filter_row.addWidget(filter_label)
        filter_row.addWidget(self.filter_combo, 1)
        side_layout.addLayout(filter_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        card_container = QWidget()
        card_container.setObjectName("cardContainer")
        self.cards_layout = QVBoxLayout(card_container)
        self.cards_layout.setContentsMargins(0, 2, 0, 2)
        self.cards_layout.setSpacing(8)
        self.category_cards: list[CategoryCard] = []
        for index, (name, destination) in enumerate(
            zip(self.categories, self.category_dirs)
        ):
            key_label = str(index + 1) if index < 9 else ("0" if index == 9 else "·")
            card = CategoryCard(
                index, key_label, name, destination, category_color(index)
            )
            card.clicked.connect(self._classify_current)
            self.category_cards.append(card)
            self.cards_layout.addWidget(card)
        self.cards_layout.addStretch(1)
        scroll.setWidget(card_container)
        side_layout.addWidget(scroll, 1)

        self.decision_label = QLabel("当前图片：未分类")
        self.decision_label.setObjectName("decision")
        self.decision_label.setWordWrap(True)
        side_layout.addWidget(self.decision_label)

        self.clear_button = QPushButton("移回待筛选")
        self.clear_button.setObjectName("secondaryButton")
        self.clear_button.setToolTip("将当前图片移回数据集根目录（Backspace）")
        self.clear_button.clicked.connect(self._unclassify_current)
        side_layout.addWidget(self.clear_button)

        tip = QLabel("结果自动暂存 · 同一分类再按一次移回待筛选")
        tip.setObjectName("tip")
        tip.setWordWrap(True)
        side_layout.addWidget(tip)

        # Tab 1: 对比校验侧边栏（包含配置排序与对决控制）
        self.compare_sidebar_stack = QStackedWidget()
        self.ranking_config_widget = RankingConfigWidget(
            self.categories, initial_ranking=self.category_ranking
        )
        self.ranking_config_widget.ranking_confirmed.connect(self._on_ranking_confirmed)

        self.compare_control_widget = CompareControlWidget(self.categories)
        self.compare_control_widget.reconfigure_requested.connect(self._on_reconfigure_ranking)
        self.compare_control_widget.skip_requested.connect(self._sample_compare_pair)

        self.compare_sidebar_stack.addWidget(self.ranking_config_widget)
        self.compare_sidebar_stack.addWidget(self.compare_control_widget)

        # 右侧 TabWidget
        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setFixedWidth(340)
        self.sidebar_tabs.addTab(triage_sidebar, "🏷️ 单图分类")
        self.sidebar_tabs.addTab(self.compare_sidebar_stack, "⚔️ 排序校验")
        self.sidebar_tabs.currentChanged.connect(self._on_mode_tab_changed)

        body.addWidget(self.sidebar_tabs)
        outer.addLayout(body, 1)

        for widget in (
            self.prev_button,
            self.next_button,
            self.clear_button,
            self.choose_left_btn,
            self.choose_right_btn,
            self.both_good_btn,
            self.both_bad_btn,
            self.compare_skip_btn,
        ):
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

        if self.category_ranking is not None:
            self.compare_control_widget.set_ranking(self.category_ranking)
            self.compare_sidebar_stack.setCurrentIndex(1)
        else:
            self.compare_sidebar_stack.setCurrentIndex(0)

    def _on_mode_tab_changed(self, index: int) -> None:
        self.main_stack.setCurrentIndex(index)
        if index == 1:
            if self.category_ranking is not None:
                self.compare_sidebar_stack.setCurrentIndex(1)
                if self.compare_left_record is None or self.compare_right_record is None:
                    self._sample_compare_pair()
            else:
                self.compare_sidebar_stack.setCurrentIndex(0)
        self.setFocus()

    def _on_ranking_confirmed(self, ranking: list[int]) -> None:
        self.category_ranking = ranking
        self.compare_control_widget.set_ranking(ranking)
        self.compare_sidebar_stack.setCurrentIndex(1)
        self._save_state()
        self._sample_compare_pair()
        self.setFocus()

    def _on_reconfigure_ranking(self) -> None:
        if self.category_ranking is not None:
            self.ranking_config_widget.set_ranking(self.category_ranking)
        self.compare_sidebar_stack.setCurrentIndex(0)
        self.setFocus()

    def _sample_compare_pair(self) -> None:
        if not self.category_ranking:
            return

        records_by_category: dict[int, list[ImageRecord]] = {
            i: [] for i in range(len(self.categories))
        }
        for r in self.records:
            if r.category is not None:
                records_by_category[r.category].append(r)

        active_categories = [
            cat for cat, recs in records_by_category.items() if len(recs) > 0
        ]
        if len(active_categories) < 2:
            self.compare_left_record = None
            self.compare_right_record = None
            self.canvas_left.clear_image()
            self.canvas_right.clear_image()
            self.canvas_left.set_empty_text("已分类类别不足 2 个")
            self.canvas_right.set_empty_text("请先在「单图分类」模式分类")
            self.left_badge.setText("无数据")
            self.right_badge.setText("无数据")
            self.left_filename.setFullText("请先为至少两个类别分类图片")
            self.right_filename.setFullText("请先为至少两个类别分类图片")
            self.choose_left_btn.setEnabled(False)
            self.choose_right_btn.setEnabled(False)
            self.both_good_btn.setEnabled(False)
            self.both_bad_btn.setEnabled(False)
            self._set_compare_status(
                "当前包含图片的类别不足 2 个，无法对决！请切换到「单图分类」继续分类。",
                is_consistent=False,
            )
            return

        self.choose_left_btn.setEnabled(True)
        self.choose_right_btn.setEnabled(True)
        self.both_good_btn.setEnabled(True)
        self.both_bad_btn.setEnabled(True)

        cat1, cat2 = random.sample(active_categories, 2)
        rec1 = random.choice(records_by_category[cat1])
        rec2 = random.choice(records_by_category[cat2])

        if random.random() < 0.5:
            self.compare_left_record = rec1
            self.compare_right_record = rec2
        else:
            self.compare_left_record = rec2
            self.compare_right_record = rec1

        self._update_compare_display()

    def _update_compare_display(self) -> None:
        left_rec = self.compare_left_record
        right_rec = self.compare_right_record
        if left_rec is None or right_rec is None:
            return

        # 盲选模式：在用户做出选择前不暴露任何类别名称与类别代表色
        neutral_style_a = (
            "background: rgba(56, 189, 248, 28); color: #38bdf8; "
            "border: 1px solid rgba(56, 189, 248, 80); border-radius: 6px; font-weight: 700; font-size: 11px;"
        )
        neutral_style_b = (
            "background: rgba(167, 139, 250, 28); color: #a78bfa; "
            "border: 1px solid rgba(167, 139, 250, 80); border-radius: 6px; font-weight: 700; font-size: 11px;"
        )

        self.left_badge.setText("图 A")
        self.left_badge.setStyleSheet(neutral_style_a)
        self.left_filename.setFullText(left_rec.path.name)
        pix_left = self._load_pixmap_for_record(left_rec)
        if pix_left is not None:
            self.canvas_left.set_pixmap(pix_left, animate=False)
            self.canvas_left.set_stamp(None)  # 盲选：不加分类印章
        else:
            self.canvas_left.set_error("读取左图失败")

        self.right_badge.setText("图 B")
        self.right_badge.setStyleSheet(neutral_style_b)
        self.right_filename.setFullText(right_rec.path.name)
        pix_right = self._load_pixmap_for_record(right_rec)
        if pix_right is not None:
            self.canvas_right.set_pixmap(pix_right, animate=False)
            self.canvas_right.set_stamp(None)  # 盲选：不加分类印章
        else:
            self.canvas_right.set_error("读取右图失败")

    def _load_pixmap_for_record(self, record: ImageRecord) -> QPixmap | None:
        if record.uid in self.pixmaps:
            return self.pixmaps[record.uid]
        if not record.path.is_file():
            return None
        reader = QImageReader(str(record.path))
        reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid() and max(size.width(), size.height()) > MAX_DECODE_DIMENSION:
            reader.setScaledSize(
                size.scaled(
                    QSize(MAX_DECODE_DIMENSION, MAX_DECODE_DIMENSION),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
            )
        img = reader.read()
        if img.isNull():
            return None
        pixmap = QPixmap.fromImage(img)
        self.pixmaps[record.uid] = pixmap
        self._trim_cache(keep=record.uid)
        return pixmap

    def _handle_compare_choice(self, chosen_side: str) -> None:
        if self.compare_left_record is None or self.compare_right_record is None:
            return
        if not self.category_ranking:
            return

        left_rec = self.compare_left_record
        right_rec = self.compare_right_record
        if left_rec.category is None or right_rec.category is None:
            return

        left_rank = self.category_ranking.index(left_rec.category)
        right_rank = self.category_ranking.index(right_rec.category)

        # 确定本次对比中较好与较差的类别
        better_rec = left_rec if left_rank < right_rank else right_rec
        worse_rec = right_rec if left_rank < right_rank else left_rec
        better_cat = better_rec.category
        worse_cat = worse_rec.category
        better_cat_name = self.categories[better_cat]
        worse_cat_name = self.categories[worse_cat]

        self.compare_total_count += 1

        # 选项 1：都很不错 -> 不再交换，而是一起扔进更好的类中
        if chosen_side == "both_good":
            self.compare_both_good_count += 1
            # 将原本在较差类别的图片晋升移动到较好类别
            self._move_record_to_category(worse_rec, better_cat)
            notice = (
                f"🌟 都很不错！已将两张图片一同归入更好的类别【{better_cat_name}】"
                f"（{worse_rec.path.name} 晋升至 {better_cat_name}）"
            )
            self._set_compare_status(notice, is_consistent=True)
            self._add_compare_log(f"🌟 俱佳: {worse_rec.path.name} ➜ {better_cat_name}")
            self._update_compare_stats()
            self._sample_compare_pair()
            return

        # 选项 2：都是垃圾 -> 不再交换，而是一起扔进更差的类中
        if chosen_side == "both_bad":
            self.compare_both_bad_count += 1
            # 将原本在较好类别的图片降级移动到较差类别
            self._move_record_to_category(better_rec, worse_cat)
            notice = (
                f"🗑️ 都是垃圾！已将两张图片一同归入更差的类别【{worse_cat_name}】"
                f"（{better_rec.path.name} 降级至 {worse_cat_name}）"
            )
            self._set_compare_status(notice, is_consistent=True)
            self._add_compare_log(f"🗑️ 俱劣: {better_rec.path.name} ➜ {worse_cat_name}")
            self._update_compare_stats()
            self._sample_compare_pair()
            return

        # 排在前面（index 越小）代表预设质量越好
        preset_left_is_better = left_rank < right_rank
        user_chose_left = chosen_side == "left"

        left_cat_name = self.categories[left_rec.category]
        right_cat_name = self.categories[right_rec.category]

        is_consistent = (preset_left_is_better and user_chose_left) or (
            not preset_left_is_better and not user_chose_left
        )

        if is_consistent:
            # 判断与预先分类一致
            self.compare_consistent_count += 1
            winner_side = "左图" if user_chose_left else "右图"
            winner_cat = left_cat_name if user_chose_left else right_cat_name
            loser_cat = right_cat_name if user_chose_left else left_cat_name
            notice = f"✓ 校验一致！用户判断【{winner_side}】更优（预设：{winner_cat} > {loser_cat}）"
            self._set_compare_status(notice, is_consistent=True)
            self._add_compare_log(f"✓ 一致: {winner_cat} > {loser_cat}")
            self._update_compare_stats()
            # 通知用户后重新抽取，不做别的事
            self._sample_compare_pair()
        else:
            # 判断与预先分类相反！
            self.compare_conflict_count += 1
            user_better_rec = left_rec if user_chose_left else right_rec
            user_worse_rec = right_rec if user_chose_left else left_rec
            user_better_side = "左图" if user_chose_left else "右图"

            better_side_cat = self.categories[user_better_rec.category]
            worse_side_cat = self.categories[user_worse_rec.category]

            notice = (
                f"⚠️ 发现分类冲突！用户选择【{user_better_side}】更优，"
                f"但预设为 {worse_side_cat} > {better_side_cat}"
            )
            self._set_compare_status(notice, is_consistent=False)
            self._add_compare_log(f"⚠️ 冲突: 认为 {better_side_cat} > {worse_side_cat}")
            self._update_compare_stats()

            # 通知用户，让用户选择是否将这两张图片的 class 交换
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("发现分类冲突 - 是否交换图片类别？")
            msg_box.setIcon(QMessageBox.Icon.Question)
            msg_box.setText(
                f"您的判断与当前预设分类相反：\n\n"
                f"• 您选择更优的图片：{user_better_rec.path.name}\n"
                f"  当前所在类别：{better_side_cat}\n\n"
                f"• 您选择较劣的图片：{user_worse_rec.path.name}\n"
                f"  当前所在类别：{worse_side_cat}\n\n"
                f"根据当前排序规则，{worse_side_cat} 应优于 {better_side_cat}。\n\n"
                f"是否将这两张图片的类别目录进行互换？"
            )
            swap_button = msg_box.addButton("🔄 交换两图类别", QMessageBox.ButtonRole.AcceptRole)
            keep_button = msg_box.addButton("保持原样（不交换）", QMessageBox.ButtonRole.RejectRole)
            msg_box.setDefaultButton(swap_button)
            msg_box.exec()

            if msg_box.clickedButton() == swap_button:
                self._swap_records(left_rec, right_rec)
                self._set_compare_status(
                    f"⇄ 已成功交换类别：{left_rec.path.name} ⮂ {right_rec.path.name}",
                    is_consistent=True,
                )
                self._add_compare_log(f"⇄ 已交换: {left_cat_name} ⮂ {right_cat_name}")
                self._update_compare_stats()

            # 然后再重抽
            self._sample_compare_pair()

    def _move_record_to_category(self, record: ImageRecord, target_cat: int) -> bool:
        if record.category == target_cat:
            return True
        source = record.path
        if not source.is_file():
            QMessageBox.warning(self, "文件不存在", f"找不到文件：\n{source}")
            return False

        destination_dir = self.category_dirs[target_cat]
        try:
            destination = unique_image_destination(destination_dir / record.original_name)
            shutil.move(str(source), str(destination))
        except OSError as exc:
            QMessageBox.critical(self, "移动失败", f"无法移动 {source.name}：\n{exc}")
            return False

        sidecar_errors: list[str] = []
        for mapper in sidecar_mappers():
            sidecar_source = mapper(source)
            if not sidecar_source.is_file():
                continue
            try:
                shutil.move(str(sidecar_source), str(mapper(destination)))
            except OSError as exc:
                sidecar_errors.append(f"{sidecar_source.name}: {exc}")

        record.path = destination
        record.category = target_cat
        self._reorder_records()
        self._refresh_stats()
        self._save_state()
        return True

    def _swap_records(self, rec_a: ImageRecord, rec_b: ImageRecord) -> None:
        cat_a = rec_a.category
        cat_b = rec_b.category
        if cat_a is None or cat_b is None or cat_a == cat_b:
            return

        dir_a = self.category_dirs[cat_a]
        dir_b = self.category_dirs[cat_b]

        dest_a = unique_image_destination(dir_b / rec_a.original_name)
        dest_b = unique_image_destination(dir_a / rec_b.original_name)

        try:
            shutil.move(str(rec_a.path), str(dest_a))
            for mapper in sidecar_mappers():
                src_sc = mapper(rec_a.path)
                if src_sc.is_file():
                    shutil.move(str(src_sc), str(mapper(dest_a)))
        except OSError as exc:
            QMessageBox.critical(self, "移动失败", f"无法移动 {rec_a.path.name}：\n{exc}")
            return

        try:
            shutil.move(str(rec_b.path), str(dest_b))
            for mapper in sidecar_mappers():
                src_sc = mapper(rec_b.path)
                if src_sc.is_file():
                    shutil.move(str(src_sc), str(mapper(dest_b)))
        except OSError as exc:
            QMessageBox.critical(self, "移动失败", f"无法移动 {rec_b.path.name}：\n{exc}")
            return

        rec_a.path = dest_a
        rec_a.category = cat_b

        rec_b.path = dest_b
        rec_b.category = cat_a

        self._reorder_records()
        self._refresh_stats()
        self._save_state()

    def _set_compare_status(self, text: str, is_consistent: bool) -> None:
        color = "#34d399" if is_consistent else "#f87171"
        self.compare_status_bar.setStyleSheet(
            f"background: #141920; border: 1px solid {color}; border-radius: 11px; "
            f"color: {color}; font-size: 12px; font-weight: 600; padding: 8px 12px;"
        )
        self.compare_status_bar.setText(text)

    def _add_compare_log(self, text: str) -> None:
        self.compare_recent_logs.append(text)
        if len(self.compare_recent_logs) > 20:
            self.compare_recent_logs.pop(0)

    def _update_compare_stats(self) -> None:
        self.compare_control_widget.update_stats(
            self.compare_total_count,
            self.compare_consistent_count,
            self.compare_conflict_count,
            self.compare_both_good_count,
            self.compare_both_bad_count,
            self.compare_recent_logs,
        )

    def _center_on_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        width = min(1340, max(980, int(available.width() * 0.84)))
        height = min(900, max(680, int(available.height() * 0.84)))
        self.resize(width, height)
        self.move(available.center() - self.rect().center())

    def _prepare_directories(self) -> None:
        for directory in self.category_dirs:
            directory.mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> None:
        state_file = self.state_path
        if not state_file.is_file():
            # 兼容读取 /tmp 下的历史旧断点文件
            resolved = self.dataset_dir.expanduser().resolve()
            digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
            slug = re.sub(r"[^\w.-]+", "_", resolved.name, flags=re.UNICODE).strip("._-")
            legacy_file = Path(tempfile.gettempdir()) / f"dataset_rlhf_{slug[:48] or 'dataset'}_{digest}.json"
            if legacy_file.is_file():
                state_file = legacy_file
            else:
                return

        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] 无法读取断点文件 {state_file}: {exc}", file=sys.stderr)
            return
        if not isinstance(data, dict):
            return
        if data.get("dataset") not in (None, str(self.dataset_dir)):
            print(f"[WARN] 忽略不匹配的断点文件：{state_file}", file=sys.stderr)
            return

        last_file = data.get("last_file")
        if isinstance(last_file, str) and last_file:
            self._resume_original_name = last_file

        decisions = data.get("decisions", [])
        if not isinstance(decisions, list):
            return
        seen_paths: set[Path] = set()
        for entry in decisions:
            if not isinstance(entry, dict):
                continue
            raw_path = entry.get("path")
            raw_category = entry.get("category")
            if not isinstance(raw_path, str) or not isinstance(raw_category, str):
                continue

            image_path = Path(raw_path).expanduser()
            if not image_path.is_absolute():
                image_path = self.dataset_dir / image_path
            image_path = image_path.resolve()
            if not image_path.is_file() or image_path in seen_paths:
                continue

            category_path = Path(raw_category).expanduser()
            if not category_path.is_absolute():
                category_path = self.dataset_dir / category_path
            category_path = category_path.resolve()
            category_index = self._category_index_by_path.get(str(category_path))
            original_name = entry.get("original_name")
            if not isinstance(original_name, str) or not original_name:
                original_name = image_path.name

            if category_index is None:
                self._carryover_decisions.append(
                    {
                        "original_name": original_name,
                        "path": str(image_path),
                        "category": str(category_path),
                    }
                )
                continue

            seen_paths.add(image_path)
            self._restored_decisions.append((original_name, image_path, category_index))

        ranking = data.get("ranking")
        if (
            isinstance(ranking, list)
            and len(ranking) == len(self.categories)
            and set(ranking) == set(range(len(self.categories)))
        ):
            self.category_ranking = ranking

    def _scan_dataset(self) -> None:
        paths = [
            path
            for path in self.dataset_dir.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in self.extensions
        ]
        paths.sort(key=natural_key)
        records = [
            ImageRecord(uid=index, original_name=path.name, path=path)
            for index, path in enumerate(paths)
        ]
        known_paths = {record.path.resolve() for record in records}

        # 1. 优先使用元数据记录（保留历史 original_name）
        for original_name, path, category_index in self._restored_decisions:
            resolved = path.resolve()
            if resolved in known_paths or not path.is_file():
                continue
            actual_cat = self._category_index_by_path.get(str(path.parent.resolve()))
            cat_idx = actual_cat if actual_cat is not None else category_index
            known_paths.add(resolved)
            records.append(
                ImageRecord(
                    uid=-1,
                    original_name=original_name,
                    path=path,
                    category=cat_idx,
                )
            )

        # 2. 直接扫描所有分类目录，即使无元数据也能恢复已分类图片
        recovered_from_dirs = 0
        for category_index, category_dir in enumerate(self.category_dirs):
            if not category_dir.is_dir():
                continue
            category_paths = [
                path
                for path in category_dir.iterdir()
                if path.is_file()
                and not path.name.startswith(".")
                and path.suffix.lower() in self.extensions
            ]
            category_paths.sort(key=natural_key)
            for path in category_paths:
                resolved = path.resolve()
                if resolved in known_paths:
                    continue
                known_paths.add(resolved)
                records.append(
                    ImageRecord(
                        uid=-1,
                        original_name=path.name,
                        path=path,
                        category=category_index,
                    )
                )
                recovered_from_dirs += 1

        if recovered_from_dirs > 0:
            print(
                f"[INFO] 已直接从分类目录恢复 {recovered_from_dirs} 张已分类图片",
                file=sys.stderr,
            )

        records.sort(key=self._record_sort_key)
        for uid, record in enumerate(records):
            record.uid = uid
        self.records = records

    def _refresh_stats(self) -> None:
        total = len(self.records)
        counts = [0] * len(self.categories)
        for record in self.records:
            if record.category is not None:
                counts[record.category] += 1
        self.category_counts = counts
        classified = sum(counts)
        self.ratio_summary.setText(f"已分类 {classified} / {total}")
        self.ratio_bar.set_data(classified, counts)
        for card, count in zip(self.category_cards, counts):
            ratio = count / classified * 100.0 if classified else 0.0
            card.set_count(count, ratio)
        self._update_filter_options()

    def _update_filter_options(self) -> None:
        total = len(self.records)
        unclassified = sum(record.category is None for record in self.records)
        self.filter_combo.blockSignals(True)
        self.filter_combo.setItemText(0, f"全部样本 ({total})")
        self.filter_combo.setItemData(0, "显示全部样本", Qt.ItemDataRole.ToolTipRole)
        self.filter_combo.setItemText(1, f"未分类 ({unclassified})")
        self.filter_combo.setItemData(1, "仅显示未分类样本", Qt.ItemDataRole.ToolTipRole)
        for index, count in enumerate(self.category_counts):
            key_label = str(index + 1) if index < 9 else ("0" if index == 9 else "·")
            combo_index = index + 2
            self.filter_combo.setItemText(
                combo_index, f"{key_label} · {self.categories[index]} ({count})"
            )
            self.filter_combo.setItemData(
                combo_index,
                f"仅显示 {self.categories[index]}\n{self.category_dirs[index]}",
                Qt.ItemDataRole.ToolTipRole,
            )
        combo_index = self.filter_combo.findData(self.filter_mode)
        if combo_index >= 0:
            self.filter_combo.setCurrentIndex(combo_index)
        self.filter_combo.blockSignals(False)

    @staticmethod
    def _record_sort_key(record: ImageRecord) -> tuple[bool, list[object]]:
        return record.category is None, natural_key(Path(record.original_name))

    def _reorder_records(self) -> None:
        self.records.sort(key=self._record_sort_key)

    def _apply_filter(
        self,
        preferred: ImageRecord | None = None,
        fallback_index: int | None = None,
        resume_original: str | None = None,
    ) -> None:
        previous = preferred if preferred is not None else self.current_record
        if self.filter_mode == FILTER_ALL:
            visible = list(self.records)
        elif self.filter_mode == FILTER_UNCLASSIFIED:
            visible = [record for record in self.records if record.category is None]
        else:
            visible = [
                record for record in self.records if record.category == self.filter_mode
            ]
        self.visible_records = visible

        selected = -1
        if resume_original is not None:
            for index, record in enumerate(visible):
                if record.original_name == resume_original:
                    selected = index
                    break
        if selected < 0 and previous is not None:
            for index, record in enumerate(visible):
                if record is previous:
                    selected = index
                    break
        if selected < 0 and visible:
            if previous is None and resume_original is None:
                selected = next(
                    (
                        index
                        for index, record in enumerate(visible)
                        if record.category is None
                    ),
                    0,
                )
            else:
                selected = min(max(0, fallback_index or 0), len(visible) - 1)

        self.slider.setRange(0, max(0, len(visible) - 1))
        self.slider.setEnabled(bool(visible))
        classified = sum(record.category is not None for record in visible)
        self.slider.set_classification_state(classified, len(visible))
        self._show_image(selected)

    def _on_filter_changed(self, combo_index: int) -> None:
        mode = self.filter_combo.itemData(combo_index)
        if not isinstance(mode, int) or mode == self.filter_mode:
            return
        self.filter_mode = mode
        self._apply_filter(preferred=self.current_record)
        self.setFocus()

    def _save_state(self) -> None:
        decisions: list[dict[str, str]] = []
        saved_paths: set[str] = set()
        for record in self.records:
            if record.category is None:
                continue
            path = str(record.path.resolve())
            decisions.append(
                {
                    "original_name": record.original_name,
                    "path": path,
                    "category": str(self.category_dirs[record.category]),
                }
            )
            saved_paths.add(path)

        for entry in self._carryover_decisions:
            raw_path = entry.get("path")
            if raw_path is None:
                continue
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file() or str(path) in saved_paths:
                continue
            decisions.append(
                {
                    "original_name": entry.get("original_name", path.name),
                    "path": str(path),
                    "category": entry.get("category", ""),
                }
            )
            saved_paths.add(str(path))

        current = self.current_record
        payload = {
            "version": 1,
            "dataset": str(self.dataset_dir),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "last_file": current.original_name if current is not None else None,
            "decisions": decisions,
        }
        if self.category_ranking is not None:
            payload["ranking"] = self.category_ranking
        temp_path = self.state_path.with_name(
            f".{self.state_path.name}.{os.getpid()}.tmp"
        )
        try:
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temp_path, self.state_path)
        except OSError as exc:
            print(f"[WARN] 无法保存断点文件 {self.state_path}: {exc}", file=sys.stderr)
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    @property
    def current_record(self) -> ImageRecord | None:
        if 0 <= self.current_index < len(self.visible_records):
            return self.visible_records[self.current_index]
        return None

    def _show_image(self, index: int) -> None:
        if not self.visible_records:
            self.current_index = -1
            self.canvas.clear_image()
            if not self.records:
                empty_text = "目录中未找到可筛选的图片"
                filename = "暂无图片"
            elif self.filter_mode == FILTER_UNCLASSIFIED:
                empty_text = "当前没有未分类样本"
                filename = "无匹配样本"
            elif self.filter_mode >= 0:
                empty_text = f"「{self.categories[self.filter_mode]}」分类下暂无样本"
                filename = "无匹配样本"
            else:
                empty_text = "当前筛选下没有样本"
                filename = "无匹配样本"
            self.canvas.set_empty_text(empty_text)
            self.filename_label.setFullText(filename)
            self.meta_label.setText("")
            self.position_label.setText("0 / 0")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.clear_button.setEnabled(False)
            self._sync_slider(0)
            self._update_decision_ui()
            return

        index = max(0, min(index, len(self.visible_records) - 1))
        changed = index != self.current_index
        self.current_index = index
        record = self.visible_records[index]

        self.filename_label.setFullText(record.path.name)
        self.position_label.setText(f"{index + 1} / {len(self.visible_records)}")
        self.prev_button.setEnabled(index > 0)
        self.next_button.setEnabled(index < len(self.visible_records) - 1)
        self.clear_button.setEnabled(record.category is not None)
        self._sync_slider(index)
        self._update_decision_ui()
        self.canvas.set_stamp(
            record.category,
            self.categories[record.category] if record.category is not None else "",
        )
        self._update_meta(record)

        pixmap = self.pixmaps.get(record.uid)
        if pixmap is not None:
            self.canvas.set_pixmap(pixmap, animate=changed)
        else:
            self.canvas.set_loading(True)
            self._request_image(record)

        self.preload_timer.start()
        if changed:
            self.setFocus()

    def _sync_slider(self, index: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)

    def _update_meta(self, record: ImageRecord) -> None:
        try:
            size = format_size(record.path.stat().st_size)
        except OSError:
            size = "文件不可用"
        dimensions = self.image_sizes.get(record.uid)
        dims = f"{dimensions[0]} × {dimensions[1]}  ·  " if dimensions else ""
        self.meta_label.setText(f"{dims}{size}")

    def _update_decision_ui(self) -> None:
        record = self.current_record
        for index, card in enumerate(self.category_cards):
            active = record is not None and record.category == index
            card.set_active(active)

        if record is None:
            self.decision_label.setText("当前图片：未分类")
            self.decision_label.setStyleSheet("")
        elif record.category is None:
            self.decision_label.setText("当前图片：未分类")
            self.decision_label.setStyleSheet("")
        else:
            color = category_color(record.category).name()
            self.decision_label.setText(
                f"当前图片：已移入 {self.categories[record.category]}/"
            )
            self.decision_label.setStyleSheet(f"color: {color}; font-weight: 600;")

    def _request_image(self, record: ImageRecord) -> None:
        if record.uid in self.pixmaps or record.uid in self.loading_uids:
            return
        if record.uid in self.failed_uids:
            if self.failed_paths.get(record.uid) != record.path:
                self.failed_uids.pop(record.uid, None)
                self.failed_paths.pop(record.uid, None)
            else:
                self.canvas.set_error(self.failed_uids[record.uid])
                return
        self.loading_uids.add(record.uid)
        task = LoadTask(record.uid, record.path, self.load_signals)
        self.tasks[record.uid] = task
        self.thread_pool.start(task)

    def _on_image_loaded(self, uid: int, image: QImage, width: int, height: int) -> None:
        self.loading_uids.discard(uid)
        self.tasks.pop(uid, None)
        self.failed_uids.pop(uid, None)
        self.failed_paths.pop(uid, None)
        pixmap = QPixmap.fromImage(image)
        self.pixmaps[uid] = pixmap
        self.image_sizes[uid] = (width or pixmap.width(), height or pixmap.height())
        self._trim_cache(keep=uid)
        record = self.current_record
        if record is not None and record.uid == uid:
            self.canvas.set_pixmap(pixmap, animate=True)
            self._update_meta(record)
        if uid in self.pixmaps:
            self.pixmaps.move_to_end(uid)

    def _on_image_failed(self, uid: int, message: str) -> None:
        self.loading_uids.discard(uid)
        task = self.tasks.pop(uid, None)
        self.failed_uids[uid] = message
        if task is not None:
            self.failed_paths[uid] = task.path
        record = self.current_record
        if record is not None and record.uid == uid:
            if task is not None and record.path != task.path and record.path.is_file():
                self.failed_uids.pop(uid, None)
                self.failed_paths.pop(uid, None)
                self.canvas.set_loading(True)
                self._request_image(record)
                return
            self.canvas.set_error(message)

    def _trim_cache(self, keep: int) -> None:
        while len(self.pixmaps) > MAX_CACHED_IMAGES:
            removable = next(
                (
                    uid
                    for uid in self.pixmaps
                    if uid != keep and (self.current_record is None or uid != self.current_record.uid)
                ),
                None,
            )
            if removable is None:
                break
            self.pixmaps.pop(removable, None)

    def _preload_neighbors(self) -> None:
        if self.current_record is None:
            return
        for offset in (-2, -1, 1, 2, 3):
            index = self.current_index + offset
            if 0 <= index < len(self.visible_records):
                record = self.visible_records[index]
                if record.uid not in self.pixmaps and record.uid not in self.loading_uids:
                    self._request_image(record)

    def go_to(self, index: int) -> None:
        if self.visible_records:
            self._show_image(index)

    def _wheel_navigate(self, direction: int) -> None:
        self.go_to(self.current_index + direction)

    def _on_slider_changed(self, value: int) -> None:
        if self.visible_records and value != self.current_index:
            self._show_image(value)

    def _on_slider_moved(self, value: int) -> None:
        if self.visible_records:
            self.position_label.setText(f"{value + 1} / {len(self.visible_records)}")

    def _classify_current(self, category_index: int) -> None:
        record = self.current_record
        if record is None:
            return
        if not 0 <= category_index < len(self.categories):
            return

        if record.category == category_index:
            self._move_record(record, self.dataset_dir, None)
            return

        destination_dir = self.category_dirs[category_index]
        self._move_record(record, destination_dir, category_index)

    def _unclassify_current(self) -> None:
        record = self.current_record
        if record is None or record.category is None:
            return
        self._move_record(record, self.dataset_dir, None)

    def _move_record(self, record: ImageRecord, destination_dir: Path, new_category: int | None) -> None:
        source = record.path
        if not source.is_file():
            QMessageBox.warning(self, "文件不存在", f"找不到文件：\n{source}")
            return

        try:
            destination = unique_image_destination(destination_dir / record.original_name)
            shutil.move(str(source), str(destination))
        except OSError as exc:
            QMessageBox.critical(self, "移动失败", f"无法移动 {source.name}：\n{exc}")
            return

        sidecar_errors: list[str] = []
        for mapper in sidecar_mappers():
            sidecar_source = mapper(source)
            if not sidecar_source.is_file():
                continue
            try:
                shutil.move(str(sidecar_source), str(mapper(destination)))
            except OSError as exc:
                sidecar_errors.append(f"{sidecar_source.name}: {exc}")

        record.path = destination
        record.category = new_category
        previous_index = self.current_index
        self._reorder_records()
        self._refresh_stats()
        self._apply_filter(preferred=record, fallback_index=previous_index)
        self._save_state()

        if sidecar_errors:
            QMessageBox.warning(
                self,
                "部分附属文件未移动",
                "图片已移动，但以下附属文件移动失败：\n" + "\n".join(sidecar_errors[:8]),
            )

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._save_state()
        super().closeEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.isAutoRepeat():
            super().keyPressEvent(event)
            return

        key = event.key()

        # 对比校验模式快捷键
        if hasattr(self, "sidebar_tabs") and self.sidebar_tabs.currentIndex() == 1:
            if key in (Qt.Key.Key_Left, Qt.Key.Key_A, Qt.Key.Key_1):
                self._handle_compare_choice("left")
                return
            if key in (Qt.Key.Key_Right, Qt.Key.Key_D, Qt.Key.Key_2):
                self._handle_compare_choice("right")
                return
            if key in (Qt.Key.Key_Up, Qt.Key.Key_W, Qt.Key.Key_3):
                self._handle_compare_choice("both_good")
                return
            if key in (Qt.Key.Key_Down, Qt.Key.Key_S, Qt.Key.Key_4):
                self._handle_compare_choice("both_bad")
                return
            if key == Qt.Key.Key_Space:
                self._sample_compare_pair()
                return
            super().keyPressEvent(event)
            return

        # 单图分类模式快捷键
        if key in (Qt.Key.Key_Left, Qt.Key.Key_A, Qt.Key.Key_PageUp):
            self.go_to(self.current_index - 1)
            return
        if key in (
            Qt.Key.Key_Right,
            Qt.Key.Key_D,
            Qt.Key.Key_PageDown,
            Qt.Key.Key_Space,
        ):
            self.go_to(self.current_index + 1)
            return
        if key == Qt.Key.Key_Home:
            self.go_to(0)
            return
        if key == Qt.Key.Key_End:
            self.go_to(len(self.visible_records) - 1)
            return
        if key == Qt.Key.Key_Backspace:
            self._unclassify_current()
            return

        digit_map = {
            Qt.Key.Key_1: 0,
            Qt.Key.Key_2: 1,
            Qt.Key.Key_3: 2,
            Qt.Key.Key_4: 3,
            Qt.Key.Key_5: 4,
            Qt.Key.Key_6: 5,
            Qt.Key.Key_7: 6,
            Qt.Key.Key_8: 7,
            Qt.Key.Key_9: 8,
            Qt.Key.Key_0: 9,
        }
        category_index = digit_map.get(key)
        if category_index is not None and category_index < len(self.categories):
            self._classify_current(category_index)
            return

        super().keyPressEvent(event)


APP_STYLESHEET = """
QMainWindow, QWidget#root {
    background: qradialgradient(cx: 0.18, cy: 0.0, radius: 1.2,
                                fx: 0.18, fy: 0.0,
                                stop: 0 #101820, stop: 0.45 #0b0f14,
                                stop: 1 #080a0d);
}
QWidget { color: #e7ebf1; }
QLabel { background: transparent; }
QFrame#panel {
    background-color: #11151b; border: 1px solid #232a34; border-radius: 18px;
}
QLabel#filename { color: #f1f4f8; font-size: 13px; font-weight: 600; }
QLabel#meta { color: #747e8c; font-size: 11px; }
QLabel#hint { color: #596371; font-size: 10px; padding-top: 1px; }
QLabel#position { color: #a5aeba; font-size: 11px; font-weight: 600; }
QPushButton#navButton {
    background: #191e25; color: #dbe2eb; border: 1px solid #2a323d;
    border-radius: 11px; font-size: 23px; padding-bottom: 3px;
    min-width: 44px; min-height: 36px;
}
QPushButton#navButton:hover { background: #222933; border-color: #3b4655; }
QPushButton#navButton:pressed { background: #151a20; }
QPushButton#navButton:disabled { color: #414a56; background: #12161b; border-color: #1d232b; }
QLabel#sideTitle { font-size: 16px; font-weight: 700; color: #f2f5f9; }
QLabel#sideSubtitle { color: #737d8a; font-size: 11px; }
QLabel#ratioTitle { color: #a7b0bc; font-size: 10px; font-weight: 600; }
QLabel#ratioSummary { color: #6d7784; font-size: 10px; }
QLabel#filterLabel { color: #8e98a5; font-size: 10px; font-weight: 600; }
QComboBox#filterCombo {
    background: #161c23; color: #dce3eb; border: 1px solid #29323d;
    border-radius: 10px; padding: 5px 28px 5px 9px; font-size: 11px;
}
QComboBox#filterCombo:hover { background: #1b222b; border-color: #3a4756; }
QComboBox#filterCombo::drop-down { border: none; width: 24px; }
QComboBox#filterCombo::down-arrow { image: none; }
QComboBox#filterCombo QAbstractItemView {
    background: #171d24; color: #dfe5ec; border: 1px solid #303a46;
    border-radius: 8px; selection-background-color: #25313d;
    selection-color: #f4f7fa; padding: 4px;
}
QLabel#categoryName { color: #e9edf3; font-size: 12px; font-weight: 600; }
QLabel#categoryPath { color: #66717f; font-size: 10px; }
QLabel#categoryCount { color: #7f8996; font-size: 11px; }
QFrame#categoryCard {
    background: #161b22; border: 1px solid #252d38; border-radius: 14px;
}
QFrame#categoryCard:hover { background: #1b212a; border-color: #394554; }
QLabel#decision {
    background: #141920; border: 1px solid #242c36; border-radius: 11px;
    color: #99a3b0; font-size: 11px; padding: 9px 11px;
}
QPushButton#secondaryButton {
    background: #192027; color: #c6ced8; border: 1px solid #2b3541;
    border-radius: 11px; padding: 8px 12px; font-size: 11px; font-weight: 600;
}
QPushButton#secondaryButton:hover { background: #202a33; border-color: #3a4858; }
QPushButton#secondaryButton:disabled { color: #505965; background: #12171c; border-color: #1e252d; }
QLabel#tip { color: #59636f; font-size: 10px; }
QScrollArea, QWidget#cardContainer { background: transparent; }
QScrollBar:vertical { width: 5px; background: transparent; margin: 2px 0; }
QScrollBar::handle:vertical { background: #343d48; border-radius: 2px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QToolTip {
    background-color: #202730; color: #dce3eb; border: 1px solid #394554;
    border-radius: 6px; padding: 5px 7px;
}
QTabWidget::pane {
    border: 1px solid #232a34; border-radius: 14px; background: #11151b;
}
QTabBar::tab {
    background: #141920; color: #8b96a5; border: 1px solid #232a34;
    border-top-left-radius: 9px; border-top-right-radius: 9px;
    padding: 7px 14px; font-weight: 600; font-size: 11px; margin-right: 4px;
}
QTabBar::tab:selected {
    background: #1a222d; color: #f1f4f8; border-color: #384658;
}
QTabBar::tab:hover:!selected {
    background: #171d26; color: #b4bfce;
}
QListWidget#rankingList {
    background: #151921; border: 1px solid #252e3b; border-radius: 10px;
    padding: 4px; font-size: 12px; outline: none;
}
QListWidget#rankingList::item {
    background: #19202a; border: 1px solid #2a3443; border-radius: 8px;
    padding: 8px 10px; margin-bottom: 4px; font-weight: 600;
}
QListWidget#rankingList::item:hover {
    background: #202936; border-color: #3e4d61;
}
QListWidget#rankingList::item:selected {
    background: #233144; border-color: #38bdf8;
}
"""


def make_app_icon() -> QIcon:
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    gradient = QLinearGradient(4, 4, 60, 60)
    gradient.setColorAt(0.0, QColor("#34d399"))
    gradient.setColorAt(1.0, QColor("#22d3ee"))
    painter.setBrush(gradient)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(QRectF(4, 4, 56, 56), 16, 16)
    painter.setPen(QPen(QColor("#081015"), 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    path = QPainterPath()
    path.moveTo(18, 33)
    path.lineTo(28, 43)
    path.lineTo(47, 21)
    painter.drawPath(path)
    painter.end()
    return QIcon(pixmap)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        usage="%(prog)s DATASET CATEGORY [CATEGORY ...]",
        description="用数字键把数据集图片快速移入不同分类目录。",
    )
    parser.add_argument("dataset", type=Path, help="待筛选图片所在的数据集目录")
    parser.add_argument(
        "categories",
        nargs="+",
        help="分类目录（相对数据集或绝对路径），按参数顺序映射到数字键 1、2、3...",
    )
    args = parser.parse_args(argv)

    dataset = args.dataset.expanduser()
    if not dataset.is_dir():
        parser.error(f"数据集目录不存在：{dataset}")
    dataset = dataset.resolve()

    categories: list[str] = []
    seen: set[Path] = set()
    for raw in args.categories:
        name = raw.strip()
        if not name:
            parser.error("分类目录名不能为空")
        destination = resolve_category_dir(dataset, name)
        if destination == dataset:
            parser.error(f"分类目录不能是数据集根目录：{destination}")
        if destination in seen:
            parser.error(f"分类目录重复：{destination}")
        if destination.exists() and not destination.is_dir():
            parser.error(f"分类目录与现有文件同名：{destination}")
        seen.add(destination)
        categories.append(name)

    args.dataset = dataset
    args.categories = categories
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Dataset RLHF")
    app.setOrganizationName("AxlTrainer")
    app.setStyle("Fusion")

    font = QFont()
    font.setFamilies(
        [
            "Inter",
            "SF Pro Display",
            "Segoe UI",
            "Noto Sans CJK SC",
            "Microsoft YaHei",
        ]
    )
    font.setPointSize(10)
    app.setFont(font)
    app.setStyleSheet(APP_STYLESHEET)
    app.setWindowIcon(make_app_icon())

    window = DatasetRLHFWindow(args.dataset, args.categories)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
