"""Print-ready AprilTag (tag36h11) PDF for the M7 head-camera check. One A4 page per tag, true scale.

    python -m aaltonomous.interface.apriltag_pdf     # -> aaltonomous/scene/assets/apriltags_m7.pdf

Size = edge of the BLACK square (what detectors report as the tag size). Each page has a 100 mm scale bar:
measure it after printing ("Actual size" / 100 %, no "fit to page"); if it is not 100 mm, the tag is wrong too.
Page 1: id 0, 100 mm -- counter tag (head camera ~0.5-0.7 m away: ~90-120 px wide, sub-cm pose).
Page 2: id 1, 50 mm  -- palm tag (fits the Dex3 palm; arm-mounted variant of the check).
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parents[1] / "scene/assets/apriltags_m7.pdf"
A4 = (210.0, 297.0)  # mm
TAGS = [(0, 100.0, "counter tag"), (1, 50.0, "palm tag")]


def page(pdf, tag_id, size_mm, use):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    img = cv2.aruco.generateImageMarker(d, tag_id, 8)  # 8x8 cells = black border + data
    fig = plt.figure(figsize=(A4[0] / 25.4, A4[1] / 25.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, A4[0]); ax.set_ylim(0, A4[1]); ax.axis("off")
    x0, y0 = (A4[0] - size_mm) / 2, 150.0 - size_mm / 2
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
              extent=(x0, x0 + size_mm, y0, y0 + size_mm))
    q = size_mm / 8 * 2  # cut line two cells out: keeps the white quiet zone the detector needs
    ax.add_patch(Rectangle((x0 - q, y0 - q), size_mm + 2 * q, size_mm + 2 * q, fill=False, ec="0.7", lw=0.5, ls="--"))
    ax.text(A4[0] / 2, 275, f"AprilTag tag36h11  id {tag_id}  -  {size_mm:.0f} mm black square  ({use})",
            ha="center", fontsize=12)
    ax.text(A4[0] / 2, 267, "Print at 100 % / Actual size. Check the bar below = 100 mm. Cut on the dashed line.",
            ha="center", fontsize=9)
    ax.plot([55, 155], [40, 40], "k", lw=1.5)
    for x in (55, 155):
        ax.plot([x, x], [37, 43], "k", lw=1.5)
    ax.text(105, 32, "100 mm", ha="center", fontsize=10)
    ax.text(A4[0] / 2, 18, "UNI3 / Aaltonomous Humanoid M7 - keep flat (glue to card), matte paper if possible",
            ha="center", fontsize=8, color="0.4")
    pdf.savefig(fig); plt.close(fig)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        for t in TAGS:
            page(pdf, *t)
    print(OUT)


if __name__ == "__main__":
    main()
