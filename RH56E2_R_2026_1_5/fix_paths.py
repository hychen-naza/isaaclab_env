"""Rewrite RH56E2_R_2026_1_5.urdf so all `package://` mesh refs become absolute
filesystem paths Isaac Lab's URDF importer can resolve directly.

Output: RH56E2_R_2026_1_5_abs.urdf alongside the original.
"""
from pathlib import Path

HERE = Path(__file__).parent.resolve()
SRC  = HERE / "urdf" / "RH56E2_R_2026_1_5.urdf"
DST  = HERE / "urdf" / "RH56E2_R_2026_1_5_abs.urdf"

text = SRC.read_text()
text = text.replace("package://RH56E2_R_2026_1_5/", str(HERE) + "/")
DST.write_text(text)
print(f"wrote {DST}")
print(f"{text.count(str(HERE))} mesh refs rewritten")
