"""
Center both object meshes and regenerate their URDFs.
Run this once before running scene.py.
"""
import sys
from pathlib import Path
import trimesh as tm

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE / "obj_utils"))
from obj2urdf import process_obj_onelink, CoacdArgs, URDFArgs


def reorient_to_z_up(src: Path, dst: Path) -> None:
    """Rotate mesh so that Meshy's -Y up becomes Z-up (Isaac Sim convention).

    Applies R_x(-90°):
      R_x(-90°) = [[1,  0,  0],
                   [0,  0,  1],
                   [0, -1,  0]]
    Effect: old -Y → new Z (up),  old Z → new Y,  old X unchanged.
    """
    import numpy as np
    mesh = tm.load(str(src), force="mesh")
    R = np.array([[1,  0,  0],
                  [0,  0,  1],
                  [0, -1,  0]], dtype=np.float64)
    mesh.vertices = mesh.vertices @ R.T
    mesh.export(str(dst))
    print(f"  reoriented {src.name} → {dst.name}  new extents (xyz): {mesh.extents.round(4)}")


def center_mesh(src: Path, dst: Path) -> None:
    mesh = tm.load(str(src), force="mesh")
    # Translate so centroid is at origin, then shift Z so bottom is at z=0
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_translation([0, 0, -mesh.bounds[0, 2]])  # bottom sits at z=0
    mesh.export(str(dst))
    extents = mesh.extents
    print(f"  saved {dst.name}  extents (xyz): {extents.round(4)}")


def main():
    pairs = [
        ("tasks/grasp_and_place/assets/grasp/frame_000000_object_mesh_grasp.obj",
         "tasks/grasp_and_place/assets/grasp/frame_000000_object_mesh_grasp_centered.obj"),
        ("tasks/grasp_and_place/assets/target/frame_000000_object_mesh_target.obj",
         "tasks/grasp_and_place/assets/target/frame_000000_object_mesh_target_centered.obj"),
    ]

    coacd_args = CoacdArgs()
    urdf_args  = URDFArgs()

    for src_rel, dst_rel in pairs:
        src = (BASE / src_rel).resolve()
        dst = (BASE / dst_rel).resolve()
        label = src.parent.name

        print(f"\n[{label}] centering mesh ...")
        center_mesh(src, dst)

        print(f"[{label}] running CoACD & generating URDF ...")
        process_obj_onelink(dst, coacd_args, urdf_args)
        # Fix bug in obj2urdf.py: it appends an empty duplicate <link name="link_original"/>
        urdf_path = dst.parent / "coacd_decomposed_object_one_link.urdf"
        text = urdf_path.read_text()
        text = text.replace("  <link name=\"link_original\"/>\n</robot>", "</robot>")
        urdf_path.write_text(text)
        print(f"[{label}] URDF written to {urdf_path}")


if __name__ == "__main__":
    main()
