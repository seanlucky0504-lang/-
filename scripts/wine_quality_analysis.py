"""End-to-end data mining on the UCI Red Wine Quality dataset.

Implemented with only Python's standard library to avoid external dependencies.

新增：为了满足“使用相关 AI 工具辅助数据挖掘”的需求，加入一个可选的 DeepSeek
链式调用结构，便于向大模型输入任意探索性问题并返回分析建议。
"""
import argparse
import csv
import json
import math
import os
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import Request, urlopen


def _read_dotenv(var_name: str) -> str | None:
    """Lightweight `.env` reader to improve key discovery on Windows/shells.

    支持在项目根目录或脚本同级目录查找 `.env` 文件，并读取形如
    `VAR=value` 的键值。避免在 Windows CMD/PowerShell 下临时前置变量
    失效的问题。
    """

    for path in (Path(".env"), Path(__file__).resolve().parent / ".env"):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == var_name:
                return value.strip()
    return None


def _clean_env_value(value: str | None) -> str:
    """Strip quotes/whitespace to avoid `'"sk-***"'` being treated as empty."""

    return value.strip().strip("'\"") if value else ""


class DeepSeekChain:
    """Minimal chain wrapper for DeepSeek API calls.

    - 通过环境变量、`.env`（项目根/脚本目录）或 CLI 参数读取 `DEEPSEEK_API_KEY`。
    - 如果缺少密钥或网络不可达，会返回基于本地数据上下文的友好降级提示。
    """

    def __init__(self, api_key: str, api_url: str = "https://api.deepseek.com/chat/completions", model: str = "deepseek-chat"):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model

    @classmethod
    def from_env(
        cls, api_key: str | None = None, api_url: str | None = None
    ) -> "DeepSeekChain | None":
        """Support CLI/env/.env override for API key & URL."""

        resolved_key = _clean_env_value(api_key)
        if not resolved_key:
            resolved_key = _clean_env_value(os.getenv("DEEPSEEK_API_KEY"))
        if not resolved_key:
            resolved_key = _clean_env_value(_read_dotenv("DEEPSEEK_API_KEY"))
        if not resolved_key:
            return None

        resolved_url = _clean_env_value(api_url)
        if not resolved_url:
            resolved_url = _clean_env_value(os.getenv("DEEPSEEK_API_URL"))
        if not resolved_url:
            resolved_url = _clean_env_value(_read_dotenv("DEEPSEEK_API_URL"))
        resolved_url = resolved_url or "https://api.deepseek.com/chat/completions"
        return cls(api_key=resolved_key, api_url=resolved_url)

    def run(self, question: str, context: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是数据科学助手，基于提供的特征统计给出中文的探索性分析建议。",
                },
                {
                    "role": "user",
                    "content": f"问题：{question}\n数据上下文：\n{context}",
                },
            ],
            "temperature": 0.2,
        }
        req = Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("choices", [{}])[0].get("message", {}).get("content", "未返回内容")
        except Exception as exc:  # noqa: BLE001 - 明确提示网络/鉴权错误
            return f"调用 DeepSeek 失败（{exc}），请检查密钥或网络后重试。"

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
LOCAL_DATA = Path("data/winequality-red-sample.csv")
OUTPUT_DIR = Path("reports")
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data(url: str = DATA_URL, local_path: Path = LOCAL_DATA) -> List[Dict[str, float]]:
    """Load CSV from local snapshot; fall back to remote if available."""
    if local_path.exists():
        text = local_path.read_text(encoding="utf-8")
    else:
        with urlopen(url) as resp:
            text = resp.read().decode("utf-8")
    reader = csv.DictReader(text.splitlines(), delimiter=";")
    records: List[Dict[str, float]] = []
    for row in reader:
        numeric_row = {k.strip().replace(" ", "_").replace("/", "_"): float(v) for k, v in row.items()}
        records.append(numeric_row)
    return records


def preprocess_data(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Remove duplicates and add binary quality label."""
    seen = set()
    unique_rows = []
    for row in rows:
        key = tuple(sorted(row.items()))
        if key not in seen:
            seen.add(key)
            row = dict(row)
            row["quality_label"] = 1 if row["quality"] >= 7 else 0
            unique_rows.append(row)
    return unique_rows


def summarize_numeric(rows: List[Dict[str, float]], fields: List[str]) -> List[Dict[str, float]]:
    summary = []
    for field in fields:
        values = [r[field] for r in rows]
        summary.append(
            {
                "feature": field,
                "count": len(values),
                "mean": statistics.mean(values),
                "stdev": statistics.pstdev(values),
                "min": min(values),
                "max": max(values),
            }
        )
    return summary


def save_summary_csv(summary: List[Dict[str, float]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "count", "mean", "stdev", "min", "max"])
        writer.writeheader()
        writer.writerows(summary)


def save_bar_svg(counts: Counter, title: str, path: Path) -> None:
    """Save a simple SVG bar chart for categorical counts."""
    width_per_bar = 40
    max_height = 200
    keys = sorted(counts.keys())
    max_count = max(counts.values()) or 1
    svg_width = width_per_bar * len(keys) + 100
    svg_height = max_height + 100
    bars = []
    for i, k in enumerate(keys):
        count = counts[k]
        height = (count / max_count) * max_height
        x = 50 + i * width_per_bar
        y = svg_height - height - 40
        bars.append(f'<rect x="{x}" y="{y}" width="{width_per_bar - 10}" height="{height}" fill="#2c7fb8" />')
        bars.append(f'<text x="{x + 5}" y="{svg_height - 15}" font-size="12">{k}</text>')
        bars.append(f'<text x="{x + 5}" y="{y - 5}" font-size="12">{count}</text>')
    svg = f"""
<svg xmlns='http://www.w3.org/2000/svg' width='{svg_width}' height='{svg_height}'>
  <text x='{svg_width/2 - 60}' y='20' font-size='16'>{title}</text>
  {''.join(bars)}
</svg>
"""
    path.write_text(svg)


def save_group_boxplot_svg(
    grouped_values: Dict[str, List[float]],
    title: str,
    y_label: str,
    path: Path,
) -> None:
    """绘制分组箱线图（最小值/第一四分位/中位数/第三四分位/最大值）。"""

    categories = list(grouped_values.keys())
    width_per_group = 160
    svg_width = width_per_group * len(categories) + 80
    svg_height = 320
    chart_top, chart_bottom = 40, 260

    # 计算全局范围用于缩放
    all_values = [v for values in grouped_values.values() for v in values]
    if not all_values:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'><text x='10' y='20'>无数据</text></svg>")
        return
    global_min, global_max = min(all_values), max(all_values)
    scale = lambda v: chart_bottom - (v - global_min) / (global_max - global_min or 1) * (chart_bottom - chart_top)

    shapes = []
    for idx, (cat, values) in enumerate(grouped_values.items()):
        sorted_vals = sorted(values)
        q1 = statistics.quantiles(sorted_vals, n=4)[0]
        median = statistics.median(sorted_vals)
        q3 = statistics.quantiles(sorted_vals, n=4)[-1]
        min_v, max_v = sorted_vals[0], sorted_vals[-1]

        x_center = 60 + idx * width_per_group
        box_width = 60
        # Whiskers
        shapes.append(f"<line x1='{x_center}' y1='{scale(min_v)}' x2='{x_center}' y2='{scale(max_v)}' stroke='#555' />")
        # Box
        shapes.append(
            f"<rect x='{x_center - box_width/2}' y='{scale(q3)}' width='{box_width}' height='{abs(scale(q3) - scale(q1))}' fill='#cfe2f3' stroke='#08519c' />"
        )
        # Median
        shapes.append(f"<line x1='{x_center - box_width/2}' y1='{scale(median)}' x2='{x_center + box_width/2}' y2='{scale(median)}' stroke='#08306b' stroke-width='2' />")
        # Whisker caps
        for y in (scale(min_v), scale(max_v)):
            shapes.append(f"<line x1='{x_center - box_width/4}' y1='{y}' x2='{x_center + box_width/4}' y2='{y}' stroke='#555' />")
        # Label
        shapes.append(f"<text x='{x_center - 20}' y='{svg_height - 20}' font-size='12'>{cat}</text>")

    # Y-axis labels
    labels = []
    for i in range(5):
        v = global_min + i * (global_max - global_min) / 4
        y = scale(v)
        labels.append(f"<text x='10' y='{y + 4}' font-size='11'>{v:.1f}</text>")
        labels.append(f"<line x1='40' y1='{y}' x2='{svg_width - 20}' y2='{y}' stroke='#eee' />")

    svg = f"""
<svg xmlns='http://www.w3.org/2000/svg' width='{svg_width}' height='{svg_height}'>
  <text x='{svg_width/2 - 80}' y='20' font-size='16'>{title}</text>
  <text x='10' y='30' font-size='12'>{y_label}</text>
  {''.join(labels)}
  {''.join(shapes)}
</svg>
"""
    path.write_text(svg)


def save_violin_scatter_svg(
    values_by_quality: Dict[int, List[float]], title: str, x_label: str, y_label: str, path: Path
) -> None:
    """用简化小提琴（核密度条）+散点展示数值分布。"""

    svg_width, svg_height = 700, 360
    chart_left, chart_right, chart_top, chart_bottom = 70, 640, 40, 300
    all_values = [v for vals in values_by_quality.values() for v in vals]
    if not all_values:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'><text x='10' y='20'>无数据</text></svg>")
        return
    min_v, max_v = min(all_values), max(all_values)
    y_scale = lambda v: chart_bottom - (v - min_v) / (max_v - min_v or 1) * (chart_bottom - chart_top)

    groups = sorted(values_by_quality.keys())
    shapes = []
    for idx, g in enumerate(groups):
        vals = sorted(values_by_quality[g])
        x_center = chart_left + idx * ((chart_right - chart_left) / max(len(groups) - 1, 1))

        # Kernel-like vertical density: count per bin scaled左右
        bins = 15
        bin_size = (max_v - min_v) / bins or 1
        counts = [0] * bins
        for v in vals:
            b = min(bins - 1, int((v - min_v) / bin_size))
            counts[b] += 1
        max_count = max(counts) or 1
        for i, c in enumerate(counts):
            y = chart_bottom - i * (chart_bottom - chart_top) / bins
            half_width = (c / max_count) * 20
            shapes.append(
                f"<line x1='{x_center - half_width}' y1='{y}' x2='{x_center + half_width}' y2='{y}' stroke='#a6bddb' stroke-width='6' stroke-linecap='round' />"
            )

        # Scatter points with jitter
        for v in vals:
            jitter = (random.random() - 0.5) * 18
            x = x_center + jitter
            y = y_scale(v)
            shapes.append(
                f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3' fill='#2b8cbe' fill-opacity='0.65' />"
            )
        shapes.append(f"<text x='{x_center - 12}' y='{svg_height - 12}' font-size='12'>{g}</text>")

    svg = f"""
<svg xmlns='http://www.w3.org/2000/svg' width='{svg_width}' height='{svg_height}'>
  <text x='{svg_width/2 - 100}' y='20' font-size='16'>{title}</text>
  <text x='{svg_width/2 - 20}' y='{svg_height - 4}' font-size='12'>{x_label}</text>
  <text x='12' y='26' font-size='12'>{y_label}</text>
  {''.join(shapes)}
</svg>
"""
    path.write_text(svg)


def save_quality_donut(counts: Dict[str, int], title: str, path: Path) -> None:
    """生成简单环形图显示质量层级分布。"""

    total = sum(counts.values()) or 1
    svg_size = 360
    radius_outer, radius_inner = 140, 80
    cx = cy = svg_size / 2

    def polar(angle: float, r: float) -> Tuple[float, float]:
        return cx + r * math.cos(angle), cy + r * math.sin(angle)

    start_angle = -math.pi / 2
    paths = []
    palette = ["#2b8cbe", "#7bccc4", "#bae4bc", "#f7fcb9"]
    for idx, (label, count) in enumerate(counts.items()):
        frac = count / total
        end_angle = start_angle + 2 * math.pi * frac
        x1, y1 = polar(start_angle, radius_outer)
        x2, y2 = polar(end_angle, radius_outer)
        x1i, y1i = polar(start_angle, radius_inner)
        x2i, y2i = polar(end_angle, radius_inner)
        large_arc = 1 if frac > 0.5 else 0
        path_d = (
            f"M {x1:.1f},{y1:.1f} A {radius_outer},{radius_outer} 0 {large_arc} 1 {x2:.1f},{y2:.1f} "
            f"L {x2i:.1f},{y2i:.1f} A {radius_inner},{radius_inner} 0 {large_arc} 0 {x1i:.1f},{y1i:.1f} Z"
        )
        color = palette[idx % len(palette)]
        paths.append(f"<path d='{path_d}' fill='{color}' stroke='#fff' stroke-width='2' />")
        mid_angle = start_angle + (end_angle - start_angle) / 2
        lx, ly = polar(mid_angle, (radius_outer + radius_inner) / 2)
        paths.append(
            f"<text x='{lx:.1f}' y='{ly:.1f}' font-size='12' text-anchor='middle' fill='#08306b'>{label} {count}</text>"
        )
        start_angle = end_angle

    svg = f"""
<svg xmlns='http://www.w3.org/2000/svg' width='{svg_size}' height='{svg_size}'>
  <text x='{cx - 80}' y='24' font-size='16'>{title}</text>
  {''.join(paths)}
</svg>
"""
    path.write_text(svg)
def build_eda_context(rows: List[Dict[str, float]], feature_names: List[str], top_k: int = 3) -> str:
    """构造发送给大模型的上下文，包含关键统计量。"""
    summary = summarize_numeric(rows, feature_names + ["quality"])
    # 选取方差最大的若干特征，作为可能影响酒质的重点探查方向
    sorted_features = sorted(summary, key=lambda s: s["stdev"], reverse=True)[:top_k]
    lines = [
        "数据集：UCI 红酒质量 (binary label: quality>=7)",
        f"样本量：{len(rows)}",
        "方差最高的特征及均值/标准差：",
    ]
    for feat in sorted_features:
        lines.append(
            f"- {feat['feature']}: mean={feat['mean']:.2f}, stdev={feat['stdev']:.2f}, min={feat['min']:.2f}, max={feat['max']:.2f}"
        )
    # 给出质量分布以引导大模型考虑类别不平衡
    quality_counts = Counter(r["quality"] for r in rows)
    dist = ", ".join(f"{k}:{v}" for k, v in sorted(quality_counts.items()))
    lines.append(f"质量分布（原始 0-10 分）：{dist}")
    return "\n".join(lines)


def pearson(x: List[float], y: List[float]) -> float:
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y))
    return num / den if den else 0.0


def compute_correlation_matrix(rows: List[Dict[str, float]], fields: List[str]) -> List[List[float]]:
    return [[pearson([r[f1] for r in rows], [r[f2] for r in rows]) for f2 in fields] for f1 in fields]


def save_correlation_svg(fields: List[str], matrix: List[List[float]], path: Path) -> None:
    cell_size = 25
    padding = 80
    svg_width = padding + cell_size * len(fields)
    svg_height = padding + cell_size * len(fields)
    rects = []
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            color_intensity = int((val + 1) / 2 * 255)
            color = f"rgb({255 - color_intensity},{255 - color_intensity},{255})"
            x = padding + j * cell_size
            y = padding + i * cell_size
            rects.append(f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="{color}" />')
            rects.append(f'<text x="{x + 5}" y="{y + 17}" font-size="10">{val:.2f}</text>')
    labels = []
    for idx, field in enumerate(fields):
        x = padding + idx * cell_size + 2
        labels.append(f'<text x="{x}" y="{padding - 10}" font-size="10" transform="rotate(-45,{x},{padding - 10})">{field}</text>')
        labels.append(f'<text x="{padding - 70}" y="{padding + idx * cell_size + 15}" font-size="10">{field}</text>')
    svg = f"""
<svg xmlns='http://www.w3.org/2000/svg' width='{svg_width}' height='{svg_height}'>
  <text x='{svg_width/2 - 80}' y='20' font-size='16'>Feature Correlation</text>
  {''.join(rects)}
  {''.join(labels)}
</svg>
"""
    path.write_text(svg)


def train_test_split(rows: List[Dict[str, float]], test_ratio: float = 0.2, seed: int = 42) -> Tuple[List, List]:
    random.seed(seed)
    shuffled = rows.copy()
    random.shuffle(shuffled)
    split = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split], shuffled[split:]


def standardize(rows: List[Dict[str, float]], features: List[str]) -> Tuple[List[List[float]], List[float], List[float]]:
    means = [statistics.mean([r[f] for r in rows]) for f in features]
    stdevs = [statistics.pstdev([r[f] for r in rows]) or 1.0 for f in features]
    standardized = []
    for r in rows:
        standardized.append([(r[f] - m) / s for f, m, s in zip(features, means, stdevs)])
    return standardized, means, stdevs


def apply_standardization(rows: List[Dict[str, float]], features: List[str], means: List[float], stdevs: List[float]) -> List[List[float]]:
    return [[(r[f] - m) / s for f, m, s in zip(features, means, stdevs)] for r in rows]


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def train_logistic_regression(X: List[List[float]], y: List[int], lr: float = 0.05, epochs: int = 400) -> List[float]:
    weights = [0.0] * (len(X[0]) + 1)  # bias + weights
    for _ in range(epochs):
        for features, target in zip(X, y):
            z = weights[0] + sum(w * f for w, f in zip(weights[1:], features))
            pred = sigmoid(z)
            error = pred - target
            weights[0] -= lr * error
            for i in range(len(features)):
                weights[i + 1] -= lr * error * features[i]
    return weights


def predict_proba(weights: List[float], X: List[List[float]]) -> List[float]:
    probs = []
    for features in X:
        z = weights[0] + sum(w * f for w, f in zip(weights[1:], features))
        probs.append(sigmoid(z))
    return probs


def f1_score(y_true: List[int], y_pred: List[int]) -> float:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0


def roc_auc_score(y_true: List[int], scores: List[float]) -> float:
    thresholds = sorted(set(scores), reverse=True)
    tprs = []
    fprs = []
    pos = sum(y_true)
    neg = len(y_true) - pos
    for t in thresholds:
        tp = sum(1 for yt, s in zip(y_true, scores) if yt == 1 and s >= t)
        fp = sum(1 for yt, s in zip(y_true, scores) if yt == 0 and s >= t)
        tpr = tp / pos if pos else 0
        fpr = fp / neg if neg else 0
        tprs.append(tpr)
        fprs.append(fpr)
    # add (0,0) and (1,1)
    points = sorted(list(zip(fprs, tprs)))
    points.insert(0, (0.0, 0.0))
    points.append((1.0, 1.0))
    auc = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        auc += (x2 - x1) * (y1 + y2) / 2
    return auc


def evaluate_model(weights: List[float], X: List[List[float]], y_true: List[int]) -> Dict[str, float]:
    probs = predict_proba(weights, X)
    preds = [1 if p >= 0.5 else 0 for p in probs]
    return {
        "f1": f1_score(y_true, preds),
        "roc_auc": roc_auc_score(y_true, probs),
    }


def save_confusion_svg(y_true: List[int], y_pred: List[int], title: str, path: Path) -> None:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    cells = [[tn, fp], [fn, tp]]
    cell_size = 60
    svg_width = 200
    svg_height = 200
    rects = []
    for i in range(2):
        for j in range(2):
            x = 60 + j * cell_size
            y = 60 + i * cell_size
            rects.append(f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="#d9f0a3" stroke="#1b7837" />')
            rects.append(f'<text x="{x + 20}" y="{y + 35}" font-size="14">{cells[i][j]}</text>')
    labels = [
        '<text x="30" y="90" font-size="12">Actual 0</text>',
        '<text x="30" y="150" font-size="12">Actual 1</text>',
        '<text x="80" y="50" font-size="12">Pred 0</text>',
        '<text x="140" y="50" font-size="12">Pred 1</text>',
    ]
    svg = f"""
<svg xmlns='http://www.w3.org/2000/svg' width='{svg_width}' height='{svg_height}'>
  <text x='20' y='20' font-size='16'>{title}</text>
  {''.join(rects)}
  {''.join(labels)}
</svg>
"""
    path.write_text(svg)


def save_interactive_dashboard(
    rows: List[Dict[str, float]],
    feature_names: List[str],
    correlation_matrix: List[List[float]],
    correlation_fields: List[str],
    path: Path,
) -> None:
    data_payload = [
        {**{f: r[f] for f in feature_names}, "quality": r["quality"], "quality_label": r["quality_label"]}
        for r in rows
    ]
    features_json = json.dumps(feature_names, ensure_ascii=False)
    corr_fields_json = json.dumps(correlation_fields, ensure_ascii=False)
    data_json = json.dumps(data_payload, ensure_ascii=False)
    corr_json = json.dumps(correlation_matrix)

    html = f"""<!doctype html>
<html lang='zh'>
<head>
  <meta charset='utf-8'/>
  <title>红酒质量交互式分析板</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; background: #f7f7f7; }}
    h1 {{ margin-bottom: 4px; }}
    .card {{ background: #fff; padding: 16px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 16px; }}
    .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    select {{ padding: 6px 8px; border-radius: 8px; border: 1px solid #ccc; }}
    svg {{ background: #fff; border: 1px solid #e1e1e1; border-radius: 8px; }}
    .badge {{ display: inline-block; padding: 4px 8px; border-radius: 8px; background: #eef2ff; color: #3949ab; margin-right: 8px; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>红酒质量交互式分析板</h1>
  <div class='card'>
    <div class='badge'>数据集：UCI 红酒质量（红葡萄酒）</div>
    <div class='badge'>样本量：{len(rows)}</div>
    <div class='badge'>可选特征：{len(feature_names)}</div>
    <p>左侧选择任一理化特征查看分布直方图，右侧可切换特征与质量分数的散点关系，底部可查看相关系数热力图。</p>
  </div>

  <div class='charts'>
    <div class='card'>
      <h3>分布直方图</h3>
      <label>选择特征：
        <select id='histFeature'></select>
      </label>
      <svg id='histogram' width='640' height='360'></svg>
    </div>
    <div class='card'>
      <h3>质量散点图</h3>
      <label>横轴特征：
        <select id='scatterFeature'></select>
      </label>
      <svg id='scatter' width='640' height='360'></svg>
    </div>
  </div>

  <div class='card'>
    <h3>相关系数热力图</h3>
    <svg id='corr' width='800' height='500'></svg>
  </div>

  <script>
    const features = {features_json};
    const data = {data_json};
    const corr = {corr_json};
    const corrFields = {corr_fields_json};

    function fillOptions() {{
      const histSel = document.getElementById('histFeature');
      const scatterSel = document.getElementById('scatterFeature');
      features.forEach(f => {{
        const opt1 = document.createElement('option'); opt1.value = f; opt1.textContent = f; histSel.appendChild(opt1);
        const opt2 = document.createElement('option'); opt2.value = f; opt2.textContent = f; scatterSel.appendChild(opt2);
      }});
      histSel.value = features[0];
      scatterSel.value = features[0];
    }}

    function drawHistogram(feature) {{
      const svg = document.getElementById('histogram');
      svg.innerHTML = '';
      const width = svg.clientWidth, height = svg.clientHeight;
      const margin = {{top: 20, right: 20, bottom: 40, left: 40}};
      const values = data.map(d => d[feature]);
      const min = Math.min(...values);
      const max = Math.max(...values);
      const bins = 12;
      const binSize = (max - min) / bins || 1;
      const counts = Array.from({{length: bins}}, () => 0);
      values.forEach(v => {{
        const idx = Math.min(bins - 1, Math.floor((v - min) / binSize));
        counts[idx] += 1;
      }});
      const maxCount = Math.max(...counts) || 1;
      counts.forEach((c, i) => {{
        const barWidth = (width - margin.left - margin.right) / bins;
        const x = margin.left + i * barWidth;
        const barHeight = (c / maxCount) * (height - margin.top - margin.bottom);
        const y = height - margin.bottom - barHeight;
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', x);
        rect.setAttribute('y', y);
        rect.setAttribute('width', barWidth - 4);
        rect.setAttribute('height', barHeight);
        rect.setAttribute('fill', '#4f83cc');
        svg.appendChild(rect);
      }});
      // Axes labels
      const xlabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      xlabel.textContent = feature;
      xlabel.setAttribute('x', width / 2 - 20);
      xlabel.setAttribute('y', height - 8);
      xlabel.setAttribute('font-size', '12');
      svg.appendChild(xlabel);
      const ylabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      ylabel.textContent = '计数';
      ylabel.setAttribute('x', 10);
      ylabel.setAttribute('y', 16);
      ylabel.setAttribute('font-size', '12');
      svg.appendChild(ylabel);
    }}

    function drawScatter(feature) {{
      const svg = document.getElementById('scatter');
      svg.innerHTML = '';
      const width = svg.clientWidth, height = svg.clientHeight;
      const margin = {{top: 20, right: 20, bottom: 40, left: 50}};
      const values = data.map(d => d[feature]);
      const qualities = data.map(d => d.quality);
      const minX = Math.min(...values), maxX = Math.max(...values);
      const minY = Math.min(...qualities), maxY = Math.max(...qualities);
      const scaleX = v => margin.left + ((v - minX) / (maxX - minX || 1)) * (width - margin.left - margin.right);
      const scaleY = v => height - margin.bottom - ((v - minY) / (maxY - minY || 1)) * (height - margin.top - margin.bottom);
      data.forEach(d => {{
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', scaleX(d[feature]));
        circle.setAttribute('cy', scaleY(d.quality));
        circle.setAttribute('r', 3);
        circle.setAttribute('fill', d.quality_label ? '#2e7d32' : '#c62828');
        circle.setAttribute('opacity', '0.7');
        svg.appendChild(circle);
      }});
      // axes
      const axisX = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      axisX.setAttribute('x1', margin.left);
      axisX.setAttribute('y1', height - margin.bottom);
      axisX.setAttribute('x2', width - margin.right);
      axisX.setAttribute('y2', height - margin.bottom);
      axisX.setAttribute('stroke', '#555');
      svg.appendChild(axisX);
      const axisY = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      axisY.setAttribute('x1', margin.left);
      axisY.setAttribute('y1', margin.top);
      axisY.setAttribute('x2', margin.left);
      axisY.setAttribute('y2', height - margin.bottom);
      axisY.setAttribute('stroke', '#555');
      svg.appendChild(axisY);
      const xlabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      xlabel.textContent = feature;
      xlabel.setAttribute('x', width / 2 - 20);
      xlabel.setAttribute('y', height - 8);
      xlabel.setAttribute('font-size', '12');
      svg.appendChild(xlabel);
      const ylabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      ylabel.textContent = 'quality';
      ylabel.setAttribute('x', 8);
      ylabel.setAttribute('y', 16);
      ylabel.setAttribute('font-size', '12');
      svg.appendChild(ylabel);
    }}

    function drawCorrelation() {{
      const svg = document.getElementById('corr');
      svg.innerHTML = '';
      const cell = 28;
      const padding = 120;
      corrFields.forEach((f, i) => {{
        const textX = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        textX.textContent = f;
        textX.setAttribute('x', padding + i * cell + 2);
        textX.setAttribute('y', 40);
        textX.setAttribute('font-size', '10');
        textX.setAttribute('transform', 'rotate(-45,' + (padding + i * cell + 2) + ', 40)');
        svg.appendChild(textX);
        const textY = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        textY.textContent = f;
        textY.setAttribute('x', 20);
        textY.setAttribute('y', padding + i * cell + 15);
        textY.setAttribute('font-size', '10');
        svg.appendChild(textY);
      }});
      corr.forEach((row, i) => {{
        row.forEach((val, j) => {{
          const intensity = Math.floor((val + 1) / 2 * 255);
          const color = 'rgb(' + (255 - intensity) + ',' + (255 - intensity) + ',255)';
          const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
          rect.setAttribute('x', padding + j * cell);
          rect.setAttribute('y', padding + i * cell);
          rect.setAttribute('width', cell - 2);
          rect.setAttribute('height', cell - 2);
          rect.setAttribute('fill', color);
          svg.appendChild(rect);
          const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
          txt.textContent = val.toFixed(2);
          txt.setAttribute('x', padding + j * cell + 2);
          txt.setAttribute('y', padding + i * cell + 14);
          txt.setAttribute('font-size', '9');
          svg.appendChild(txt);
        }});
      }});
    }}

    function init() {{
      fillOptions();
      drawHistogram(features[0]);
      drawScatter(features[0]);
      drawCorrelation();
      document.getElementById('histFeature').addEventListener('change', e => drawHistogram(e.target.value));
      document.getElementById('scatterFeature').addEventListener('change', e => drawScatter(e.target.value));
    }}

    init();
  </script>
</body>
</html>
"""

    path.write_text(html, encoding="utf-8")


def main(
    question: str | None = None,
    deepseek_api_key: str | None = None,
    deepseek_api_url: str | None = None,
):
    raw_rows = load_data()
    rows = preprocess_data(raw_rows)
    feature_names = [k for k in rows[0].keys() if k not in {"quality", "quality_label"}]

    summary = summarize_numeric(rows, feature_names + ["quality"])
    save_summary_csv(summary, OUTPUT_DIR / "summary_stats.csv")

    quality_counts = Counter(r["quality"] for r in rows)
    save_bar_svg(quality_counts, "Quality Distribution", FIG_DIR / "quality_distribution.svg")

    # 质量分层环形图（低/中/高）
    banded = {"低质量(<=4)": 0, "中等(5-6)": 0, "高质量(>=7)": 0}
    for q in quality_counts:
        if q <= 4:
            banded["低质量(<=4)"] += quality_counts[q]
        elif q <= 6:
            banded["中等(5-6)"] += quality_counts[q]
        else:
            banded["高质量(>=7)"] += quality_counts[q]
    save_quality_donut(banded, "质量层级占比", FIG_DIR / "quality_donut.svg")

    fields_for_corr = feature_names + ["quality"]
    correlation_matrix = compute_correlation_matrix(rows, fields_for_corr)
    save_correlation_svg(fields_for_corr, correlation_matrix, FIG_DIR / "correlation_heatmap.svg")

    # 硫化物分布箱线图（按高/普通质量分组）
    grouped_sulfur = {
        "普通酒(quality<7)": [r["total_sulfur_dioxide"] for r in rows if r["quality_label"] == 0],
        "高质量酒(>=7)": [r["total_sulfur_dioxide"] for r in rows if r["quality_label"] == 1],
    }
    save_group_boxplot_svg(
        grouped_sulfur,
        title="总二氧化硫分布（按质量分组）",
        y_label="total_sulfur_dioxide",
        path=FIG_DIR / "total_sulfur_box.svg",
    )

    grouped_free = {
        "普通酒(quality<7)": [r["free_sulfur_dioxide"] for r in rows if r["quality_label"] == 0],
        "高质量酒(>=7)": [r["free_sulfur_dioxide"] for r in rows if r["quality_label"] == 1],
    }
    save_group_boxplot_svg(
        grouped_free,
        title="游离二氧化硫分布（按质量分组）",
        y_label="free_sulfur_dioxide",
        path=FIG_DIR / "free_sulfur_box.svg",
    )

    # 固定酸度 vs 质量分数：小提琴 + 散点
    quality_buckets = {}
    for r in rows:
        quality_buckets.setdefault(int(r["quality"]), []).append(r["fixed_acidity"])
    save_violin_scatter_svg(
        quality_buckets,
        title="不同质量分数下的固定酸度分布",
        x_label="quality",
        y_label="fixed_acidity",
        path=FIG_DIR / "fixed_acidity_violin.svg",
    )

    # 硫化物比例分析（free/total）按质量分组
    ratio_groups = {"普通酒(quality<7)": [], "高质量酒(>=7)": []}
    for r in rows:
        ratio = r["free_sulfur_dioxide"] / r["total_sulfur_dioxide"] if r["total_sulfur_dioxide"] else 0
        bucket = "高质量酒(>=7)" if r["quality_label"] == 1 else "普通酒(quality<7)"
        ratio_groups[bucket].append(ratio)
    save_group_boxplot_svg(
        ratio_groups,
        title="硫化物比例 (free/total) 分布",
        y_label="比例",
        path=FIG_DIR / "sulfur_ratio_box.svg",
    )

    train_rows, test_rows = train_test_split(rows, test_ratio=0.2, seed=42)
    X_train, means, stdevs = standardize(train_rows, feature_names)
    y_train = [r["quality_label"] for r in train_rows]
    X_test = apply_standardization(test_rows, feature_names, means, stdevs)
    y_test = [r["quality_label"] for r in test_rows]

    weights = train_logistic_regression(X_train, y_train, lr=0.05, epochs=500)
    train_metrics = evaluate_model(weights, X_train, y_train)
    test_metrics = evaluate_model(weights, X_test, y_test)

    test_probs = predict_proba(weights, X_test)
    test_preds = [1 if p >= 0.5 else 0 for p in test_probs]
    save_confusion_svg(y_test, test_preds, "Logistic Regression Confusion", FIG_DIR / "log_reg_confusion.svg")

    metrics = {
        "model": "custom_logistic_regression",
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "evaluation_standard": "F1-score and ROC-AUC on held-out 20% test set",
    }

    with open(OUTPUT_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_interactive_dashboard(
        rows, feature_names, correlation_matrix, fields_for_corr, OUTPUT_DIR / "interactive_dashboard.html"
    )

    print(json.dumps(metrics, indent=2))

    if question:
        context = build_eda_context(rows, feature_names)
        chain = DeepSeekChain.from_env(api_key=deepseek_api_key, api_url=deepseek_api_url)
        if chain:
            answer = chain.run(question, context)
        else:
            answer = (
                "未检测到有效的 DEEPSEEK_API_KEY，请按以下方式之一传入：\n"
                "- Bash: export DEEPSEEK_API_KEY=sk-xxx && python scripts/wine_quality_analysis.py --ask '...';\n"
                "- PowerShell: $Env:DEEPSEEK_API_KEY=\"sk-xxx\"; python scripts/wine_quality_analysis.py --ask \"...\";\n"
                "- Windows CMD: set DEEPSEEK_API_KEY=sk-xxx && python scripts/wine_quality_analysis.py --ask \"...\";\n"
                "- `.env` 文件：在项目根目录创建 .env，写入 DEEPSEEK_API_KEY=sk-xxx，必要时再运行脚本。\n\n"
                "以下为基于本地统计的离线建议：\n"
                "- 关注方差较大的理化指标（如酒精含量、总二氧化硫）与质量的关系。\n"
                "- 结合相关系数热力图，挑选正负相关最强的特征做特征工程。\n"
                "- 可以尝试阈值移动或采样方法平衡标签，提高 F1。"
            )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "deepseek_chain_output.txt").write_text(answer, encoding="utf-8")
        print("\n[DeepSeek Chain 输出]\n" + answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wine quality data mining with optional DeepSeek chain")
    parser.add_argument(
        "--ask",
        help=(
            "向 DeepSeek 提交的探索性分析问题，示例：'哪些特征最值得做特征交互？'。"
            "需要提供 DEEPSEEK_API_KEY 环境变量。"
        ),
    )
    parser.add_argument(
        "--deepseek-api-key",
        help="可选，直接通过命令行传入 DeepSeek API Key（优先于环境变量）。",
    )
    parser.add_argument(
        "--deepseek-api-url",
        help="可选，自定义 DeepSeek 网关地址，便于走代理或私有网关。",
    )
    args = parser.parse_args()
    main(question=args.ask, deepseek_api_key=args.deepseek_api_key, deepseek_api_url=args.deepseek_api_url)
