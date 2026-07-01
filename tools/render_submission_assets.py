import base64
import io
import json
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebook3663bc9634.ipynb"
OUT_ROOT = ROOT / "submission_assets"
SCREENSHOT_DIR = OUT_ROOT / "screenshots"
CHART_DIR = OUT_ROOT / "generated_charts"

FONT_DIR = Path("C:/Windows/Fonts")
FONT_REGULAR = FONT_DIR / "arial.ttf"
FONT_BOLD = FONT_DIR / "arialbd.ttf"
FONT_MONO = FONT_DIR / "consola.ttf"


SCREENSHOTS = [
    (6, "part1_cluster_monitoring.png", "Cell 3: View Cluster Nodes"),
    (7, "part1_cluster_metrics_summary.png", "Cell 4: Cluster Metrics Summary"),
    (9, "part2_workload_submission.png", "Cell 5: Submit Multiple Workloads"),
    (10, "part2_billing_summary.png", "Cell 6: Record Billing For Workloads"),
    (12, "part3_spot_pricing.png", "Cell 7: Check Spot Pricing"),
    (13, "part3_spot_request.png", "Cell 8: Request Spot Instances"),
    (14, "part3_spot_preemption.png", "Cell 9: Simulate Spot Preemption"),
    (16, "part4_autoscaler_policy.png", "Cell 10: Autoscaler Policy"),
    (17, "part4_autoscaler_evaluation.png", "Cell 11: Autoscaler Evaluation"),
    (19, "part5_cost_snapshots.png", "Cell 12: Take Cost Snapshots"),
    (20, "part5_waste_report.png", "Cell 13: Waste Report"),
    (21, "part5_recommendations.png", "Cell 14: Optimization Recommendations"),
    (22, "part5_dashboard.png", "Cell 15: Full Dashboard View"),
    (24, "part6_cost_breakdown_viz.png", "Cell 16: Cost Breakdown Visualization"),
    (25, "part6_timeseries_viz.png", "Cell 17: Time-Series Cost Tracking"),
    (27, "part7_full_workflow.png", "Cell 18: Full FinOps Workflow"),
    (29, "part8_gpu_detection.png", "Cell 19: Detect Real GPU"),
    (30, "part8_gpu_metrics_diagnostic.png", "Cell 20: GPU Metrics Collection"),
    (32, "part8_fp32_summary.png", "Cell 22: Train FP32 Summary"),
    (33, "part8_amp_summary.png", "Cell 23: Train AMP Summary"),
    (34, "part8_fp32_vs_amp_comparison.png", "Cell 24: FP32 vs AMP Comparison"),
    (35, "part8_real_gpu_cost_report.png", "Cell 25: Real GPU Cost Report"),
]

CHART_EXPORTS = {
    24: ["finops_cost_breakdown.png"],
    25: ["finops_timeseries.png"],
    34: ["real_gpu_comparison.png"],
    36: ["real_gpu_telemetry.png", "cost_per_epoch.png"],
}


def load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def parse_student_info(nb: dict) -> tuple[str, str]:
    source = "".join(nb["cells"][2].get("source", []))
    name = re.search(r'STUDENT_NAME\s*=\s*"([^"]+)"', source)
    student_id = re.search(r'STUDENT_ID\s*=\s*"([^"]+)"', source)
    return (
        name.group(1) if name else "Student Name",
        student_id.group(1) if student_id else "Student ID",
    )


def output_parts(cell: dict) -> list[tuple[str, object]]:
    parts: list[tuple[str, object]] = []
    for output in cell.get("outputs", []):
        output_type = output.get("output_type")
        if output_type == "stream":
            parts.append(("text", "".join(output.get("text", []))))
        elif output_type == "error":
            traceback = "\n".join(output.get("traceback", []))
            parts.append(("text", traceback or f"{output.get('ename')}: {output.get('evalue')}"))
        elif output_type in {"display_data", "execute_result"}:
            data = output.get("data", {})
            if "text/plain" in data:
                text = data["text/plain"]
                parts.append(("text", "".join(text) if isinstance(text, list) else str(text)))
            if "image/png" in data:
                raw = data["image/png"]
                raw = "".join(raw) if isinstance(raw, list) else raw
                image = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGBA")
                parts.append(("image", image))
    return parts


def wrapped_lines(text: str, width_chars: int = 132) -> list[str]:
    lines: list[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not line:
            lines.append("")
            continue
        chunks = textwrap.wrap(
            line,
            width=width_chars,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=False,
        )
        lines.extend(chunks or [""])
    return lines


def render_cell(cell: dict, student_name: str, student_id: str, title: str, target: Path) -> None:
    width = 1400
    margin = 48
    header_h = 132
    title_h = 58
    line_h = 24
    gap = 18

    font_header = load_font(FONT_BOLD, 34)
    font_sub = load_font(FONT_REGULAR, 25)
    font_title = load_font(FONT_BOLD, 25)
    font_mono = load_font(FONT_MONO, 19)

    parts = output_parts(cell)
    blocks: list[tuple[str, object, int]] = []
    content_h = 0

    for kind, value in parts:
        if kind == "text":
            lines = wrapped_lines(str(value))
            h = max(line_h, len(lines) * line_h) + gap
            blocks.append(("text", lines, h))
            content_h += h
        elif kind == "image":
            image = value
            max_w = width - margin * 2
            if image.width > max_w:
                ratio = max_w / image.width
                image = image.resize((int(image.width * ratio), int(image.height * ratio)), Image.LANCZOS)
            h = image.height + gap
            blocks.append(("image", image, h))
            content_h += h

    if not blocks:
        blocks.append(("text", ["No output captured in notebook."], line_h + gap))
        content_h += line_h + gap

    height = header_h + title_h + content_h + margin * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for x in range(width):
        t = x / max(width - 1, 1)
        r = int(102 * (1 - t) + 118 * t)
        g = int(126 * (1 - t) + 75 * t)
        b = int(234 * (1 - t) + 162 * t)
        draw.line([(x, 0), (x, header_h)], fill=(r, g, b))

    draw.text((margin, 26), "GPU FinOps Lab - Student Information", fill="white", font=font_header)
    draw.text((margin, 80), f"Ho va ten: {student_name}  |  MSSV: {student_id}", fill="white", font=font_sub)

    y = header_h
    draw.rectangle((0, y, width, y + title_h), fill=(245, 247, 251))
    draw.text((margin, y + 15), title, fill=(25, 33, 51), font=font_title)
    y += title_h + 24

    for kind, value, _ in blocks:
        if kind == "text":
            for line in value:
                draw.text((margin, y), line, fill=(30, 30, 30), font=font_mono)
                y += line_h
            y += gap
        elif kind == "image":
            image = value.convert("RGB")
            canvas.paste(image, (margin, y))
            y += image.height + gap

    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)


def export_charts(nb: dict) -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for cell_num, filenames in CHART_EXPORTS.items():
        images = [value for kind, value in output_parts(nb["cells"][cell_num - 1]) if kind == "image"]
        for image, filename in zip(images, filenames):
            image.convert("RGB").save(CHART_DIR / filename)


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    student_name, student_id = parse_student_info(nb)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    created = []
    for cell_num, filename, title in SCREENSHOTS:
        target = SCREENSHOT_DIR / filename
        render_cell(nb["cells"][cell_num - 1], student_name, student_id, title, target)
        created.append(target.relative_to(ROOT))

    export_charts(nb)
    chart_files = sorted(p.relative_to(ROOT) for p in CHART_DIR.glob("*.png"))

    print("Created screenshots:")
    for path in created:
        print(f"  {path}")
    print("Created chart files:")
    for path in chart_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
