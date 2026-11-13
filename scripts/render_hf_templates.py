"""Render hf bundle files that share one template."""

from pathlib import Path

HF = Path(__file__).absolute().parents[1] / "hf"
TEMPLATES = {
    "c_radio_35_vision.py": {
        "C-RADIOv4-H-3.5-Bolt/vision.py": "C-RADIOv4-H-3.5-Bolt",
        "C-RADIOv4-H-3.5-Super/vision.py": "C-RADIOv4-H-3.5-Super",
    },
}
MARKER = "{model}"


def render() -> dict[Path, str]:
    rendered = {}
    for template, targets in TEMPLATES.items():
        text = (HF / "templates" / template).read_text()
        for target, model in targets.items():
            rendered[HF / target] = text.replace(MARKER, model)
    return rendered


def main() -> None:
    for path, text in render().items():
        path.write_text(text)
        print(path.relative_to(HF.parent))


if __name__ == "__main__":
    main()
