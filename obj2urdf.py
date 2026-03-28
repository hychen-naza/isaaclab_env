import os
import numpy as np
from trimesh.version import __version__ as trimesh_version
import trimesh as tm
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from pathlib import Path
# from onedex import YCB_MODELS_DIR_PATH_STR
import pdb 

YCB_MODELS_DIR_PATH_STR = "/home/hongyi/DRO-Grasp/data/data_urdf/object/video_umi/" #"/home/hongyi/DRO-Grasp/data/data_urdf/object/video/"

@dataclass(frozen=True)
class CoacdArgs:
    """Arguments to pass to CoACD.
    Defaults and descriptions are copied from: https://github.com/SarahWeiii/CoACD
    """

    preprocess_resolution: int = 50
    """resolution for manifold preprocess (20~100), default = 50"""
    threshold: float = 0.05
    """concavity threshold for terminating the decomposition (0.01~1), default = 0.05"""
    max_convex_hull: int = -1
    """max # convex hulls in the result, -1 for no maximum limitation"""
    mcts_iterations: int = 100
    """number of search iterations in MCTS (60~2000), default = 100"""
    mcts_max_depth: int = 3
    """max search depth in MCTS (2~7), default = 3"""
    mcts_nodes: int = 20
    """max number of child nodes in MCTS (10~40), default = 20"""
    resolution: int = 2000
    """sampling resolution for Hausdorff distance calculation (1e3~1e4), default = 2000"""
    pca: bool = False
    """enable PCA pre-processing, default = false"""
    seed: int = 0
    """random seed used for sampling, default = 0"""


@dataclass(frozen=True)
class YCBArgs:
    obj_dir: str = YCB_MODELS_DIR_PATH_STR
    """path to a directory containing obj files. All obj files in the directory will be converted"""
    coacd_args: CoacdArgs = field(default_factory=CoacdArgs)
    """arguments to pass to CoACD"""
    num_processes: int = 1
    """number of processes to use for multiprocessing"""


@dataclass(frozen=True)
class URDFArgs:
    scale: float = 1.0
    """scale factor for the mesh"""
    density: float = 1000
    """density of the mesh"""


def process_obj(obj_file_path: Path, args: CoacdArgs, urdf_args: URDFArgs) -> None:
    """
    - args:
        - obj_file_path: Path

    ---
    under the same directory as the obj_file_path, create a new directory named "coacd_meshes"
    and save the decomposed parts as separate OBJ files.


    """
    import coacd
    import lxml.etree as et

    mesh = tm.load(obj_file_path.resolve(), force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)  # type: ignore
    parts = coacd.run_coacd(
        mesh=mesh,
        **asdict(args),
    )
    convex_pieces = []
    for vs, fs in parts:
        convex_pieces.append(tm.Trimesh(vs, fs))

    # Save the decomposed parts as separate OBJ files.
    sub_meshes_dir_name = "coacd_meshes"
    sub_meshes_dir_path = (obj_file_path.parent / sub_meshes_dir_name).resolve()
    sub_meshes_dir_path.mkdir(exist_ok=True)
    sub_meshes_dir_path_str = sub_meshes_dir_path.as_posix()
    scale = urdf_args.scale
    density = urdf_args.density

    urdf_name = "coacd_decomposed_object"
    root = et.Element("robot", name="root")

    # create a only visual link using the original mesh
    piece_name = "original"
    link_name = "link_{}".format(piece_name)
    link = et.SubElement(root, "link", name=link_name)
    visual = et.SubElement(link, "visual")
    et.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
    geometry = et.SubElement(visual, "geometry")
    et.SubElement(
        geometry,
        "mesh",
        filename=os.path.basename(str(obj_file_path.resolve())),
        scale="{:.4E} {:.4E} {:.4E}".format(scale, scale, scale),
    )

    prev_link_name = link_name

    for i, piece in enumerate(convex_pieces):

        # Save each nearly convex mesh out to a file
        piece_name = f"convex_piece_{i}"
        piece_filename = f"{piece_name}.obj"
        piece_filepath = os.path.join(sub_meshes_dir_path_str, piece_filename)
        piece.export(piece_filepath)

        piece.density = density

        link_name = "link_{}".format(piece_name)
        geom_name = "{}".format(piece_name)
        I = [["{:.2E}".format(y) for y in x] for x in piece.moment_inertia]  # NOQA

        # Write the link out to the XML Tree
        link = et.SubElement(root, "link", name=link_name)

        # Inertial information
        inertial = et.SubElement(link, "inertial")
        et.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        et.SubElement(
            inertial,
            "inertia",
            ixx=I[0][0],
            ixy=I[0][1],
            ixz=I[0][2],
            iyy=I[1][1],
            iyz=I[1][2],
            izz=I[2][2],
        )
        # Visual Information
        # visual = et.SubElement(link, 'visual')
        # et.SubElement(visual, 'origin', xyz="0 0 0", rpy="0 0 0")
        # geometry = et.SubElement(visual, 'geometry')
        # et.SubElement(geometry, 'mesh', filename=geom_name,
        #               scale="{:.4E} {:.4E} {:.4E}".format(scale,
        #                                                   scale,
        #                                                   scale))
        # material = et.SubElement(visual, 'material', name='')
        # et.SubElement(material,
        #               'color',
        #               rgba="{:.2E} {:.2E} {:.2E} 1".format(color[0],
        #                                                    color[1],
        #                                                    color[2]))

        # Collision Information
        collision = et.SubElement(link, "collision")
        et.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
        geometry = et.SubElement(collision, "geometry")
        et.SubElement(
            geometry,
            "mesh",
            filename=f"{sub_meshes_dir_name}/{piece_filename}",
            scale="{:.4E} {:.4E} {:.4E}".format(scale, scale, scale),
        )

        # Create rigid joint to previous link
        if prev_link_name is not None:
            joint_name = "{}_joint".format(link_name)
            joint = et.SubElement(root, "joint", name=joint_name, type="fixed")
            et.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
            et.SubElement(joint, "parent", link=prev_link_name)
            et.SubElement(joint, "child", link=link_name)

        prev_link_name = link_name

    # Write URDF file
    tree = et.ElementTree(root)
    urdf_filename = "{}.urdf".format(urdf_name)
    tree.write(
        os.path.join(obj_file_path.resolve().parent.as_posix(), urdf_filename),
        pretty_print=True,
    )

    # Write Gazebo config file
    root = et.Element("model")
    model = et.SubElement(root, "name")
    model.text = urdf_name
    version = et.SubElement(root, "version")
    version.text = "1.0"
    sdf = et.SubElement(root, "sdf", version="1.4")
    sdf.text = "{}.urdf".format(urdf_name)

    author = et.SubElement(root, "author")
    et.SubElement(author, "name").text = "trimesh {}".format(trimesh_version)
    et.SubElement(author, "email").text = "blank@blank.blank"

    description = et.SubElement(root, "description")
    description.text = urdf_name

    tree = et.ElementTree(root)
    tree.write(os.path.join(sub_meshes_dir_path_str, "model.config"))

    return np.sum(convex_pieces)


def process_obj_onelink(
    obj_file_path: Path, args: CoacdArgs, urdf_args: URDFArgs
) -> None:
    """
    - args:
        - obj_file_path: Path

    ---
    since isaacgym now support treating submeshes in the mesh as the convex decomposition of the mesh,
    we can create a URDF file that has only one link, which is the original mesh, and use the submeshes
    as the convex decomposition of the mesh.

    """
    import coacd
    import lxml.etree as et
    import pdb 
    mesh_tm = tm.load(obj_file_path.resolve(), force="mesh")

    # # Get the bounding box extents
    # bounding_box = mesh_tm.bounds
    # min_coords, max_coords = bounding_box[0], bounding_box[1]
    # max_extent = max(max_coords - min_coords)

    # # Compute scale factor to fit in [-0.05, 0.05]
    # scale_factor = 0.1 / max_extent

    # # Center the object
    # centroid = (min_coords + max_coords) / 2
    # mesh_tm.apply_translation(-centroid)

    # # Scale the object
    # mesh_tm.apply_scale(scale_factor)


    mesh = coacd.Mesh(mesh_tm.vertices, mesh_tm.faces)  # type: ignore
    parts = coacd.run_coacd(
        mesh=mesh,
        **asdict(args),
    )
    convex_pieces = []
    for vs, fs in parts:
        convex_pieces.append(tm.Trimesh(vs, fs))

    one_mesh_path = obj_file_path.parent.resolve() / "coacd_allinone.obj"
    one_mesh = tm.Scene(convex_pieces)
    one_mesh.export(one_mesh_path)

    scale = urdf_args.scale
    density = urdf_args.density

    urdf_name = "coacd_decomposed_object_one_link"
    root = et.Element("robot", name="root")

    # create a only visual link using the original mesh
    piece_name = "original"
    link_name = "link_{}".format(piece_name)
    link = et.SubElement(root, "link", name=link_name)
    # Visual Information
    visual = et.SubElement(link, "visual")
    et.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
    geometry = et.SubElement(visual, "geometry")
    et.SubElement(
        geometry,
        "mesh",
        filename=os.path.basename(str(obj_file_path.resolve())),
        scale="{:.4E} {:.4E} {:.4E}".format(scale, scale, scale),
    )
    # Inertial information
    mesh_tm.density = density
    I = [["{:.2E}".format(y) for y in x] for x in mesh_tm.moment_inertia]  # NOQA
    inertial = et.SubElement(link, "inertial")
    et.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    et.SubElement(
        inertial,
        "inertia",
        ixx=I[0][0],
        ixy=I[0][1],
        ixz=I[0][2],
        iyy=I[1][1],
        iyz=I[1][2],
        izz=I[2][2],
    )
    # Collision Information
    collision = et.SubElement(link, "collision")
    et.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
    geometry = et.SubElement(collision, "geometry")
    et.SubElement(
        geometry,
        "mesh",
        filename=f"coacd_allinone.obj",
        scale="{:.4E} {:.4E} {:.4E}".format(scale, scale, scale),
    )
    # Write URDF file
    tree = et.ElementTree(root)
    urdf_filename = "{}.urdf".format(urdf_name)

    # print(f"\n\n\n !!!!!!! final path {os.path.join(obj_file_path.resolve().parent.as_posix(), urdf_filename)}")
    tree.write(
        os.path.join(obj_file_path.resolve().parent.as_posix(), urdf_filename),
        pretty_print=True,
    )

    return np.sum(convex_pieces)


def main() -> None:
    import tyro
    args = tyro.cli(YCBArgs)
    urdf_args = URDFArgs()

    # obj_id_and_names = ['gt'] #os.listdir(args.obj_dir) #['scissors'] #

    # #pdb.set_trace()
    # obj_files = [
    #     (Path(args.obj_dir) / obj_id_and_name / "obj_mesh.obj").resolve()
    #     for obj_id_and_name in obj_id_and_names
    # ]

    objs = ['dinning_bowl'] #['whip', 'teapot', 'axe', 'remoter', 'fire_torch', 'whisk', 'hand_soap', 'chinese_calligraphy_pen', 'whip'] #['remoter','chinese_calligraphy_pen']
    #['axe', 'fire_torch', 'hammer', 'whisk', 'hand_soap', 'longspoon', 'spatula', 'chinese_calligraphy_pen', 'longsword', 'lantern', 'tea_cup', 'tea_pot'] 
    obj_files = [
        (Path(args.obj_dir) / obj_id_and_name / "obj_mesh.obj").resolve()
        for obj_id_and_name in objs
    ]

    for obj_file in obj_files:
        print(f"obj_file {obj_file}")
        process_obj_onelink(obj_file, args.coacd_args, urdf_args)
    # pool_params = [(obj_file, args.coacd_args, urdf_args) for obj_file in obj_files]

    # with Pool(args.num_processes) as pool:
    #     # pool.starmap(process_obj, pool_params)
    #     pool.starmap(process_obj_onelink, pool_params)


if __name__ == "__main__":
    main()
