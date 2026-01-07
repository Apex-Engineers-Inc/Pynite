# %%
# `__future__` import required to use bar operators for optional type annotations
from __future__ import annotations  # Allows more recent type hints features
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy import array, zeros, matmul, subtract
from numpy.linalg import solve
import scipy as sp
from scipy.sparse.linalg import spsolve

from Pynite.Node3D import Node3D
from Pynite.Material import Material
from Pynite.Section import Section, SteelSection
from Pynite.PhysMember import PhysMember
from Pynite.Spring3D import Spring3D
from Pynite.Quad3D import Quad3D
from Pynite.Plate3D import Plate3D
from Pynite.LoadCombo import LoadCombo
from Pynite.Mesh import Mesh, RectangleMesh, AnnulusMesh, FrustrumMesh, CylinderMesh
from Pynite.ShearWall import ShearWall
from Pynite import Analysis
from Pynite.ParallelUtils import is_free_threaded, get_optimal_worker_count

if TYPE_CHECKING:
    from typing import Dict, List, Tuple, Union, Any
    from numpy import float64
    from numpy.typing import NDArray
    from Pynite.Member3D import Member3D as Member3DType


# %%
class FEModel3D():
    """A 3D finite element model object. This object has methods and dictionaries to create, store,
       and retrieve results from a finite element model.
    """

    def __init__(self) -> None:
        """Creates a new 3D finite element model.
        """

        # Initialize the model's various dictionaries. The dictionaries will be prepopulated with
        # the data types they store, and then those types will be removed. This will give us the
        # ability to get type-based hints when using the dictionaries.

        self.nodes: Dict[str, Node3D] = {}             # A dictionary of the model's nodes
        self.materials: Dict[str, Material] = {}       # A dictionary of the model's materials
        self.sections: Dict[str, Section] = {}         # A dictonary of the model's cross-sections
        self.springs: Dict[str, Spring3D] = {}         # A dictionary of the model's springs
        self.members: Dict[str, PhysMember] = {}       # A dictionary of the model's physical members
        self.quads: Dict[str, Quad3D] = {}             # A dictionary of the model's quadiralterals
        self.plates: Dict[str, Plate3D] = {}           # A dictionary of the model's rectangular plates
        self.meshes: Dict[str, Mesh] = {}              # A dictionary of the model's meshes
        self.shear_walls: Dict[str, ShearWall] = {}    # A dictionary of the model's shear walls
        self.load_combos: Dict[str, LoadCombo] = {}    # A dictionary of the model's load combinations
        self._D: Dict[str, NDArray[float64]] = {}      # A dictionary of the model's nodal displacements by load combination

        self.solution: str | None = None  # Indicates the solution type for the latest run of the model

    @staticmethod
    def _build_dof_vector(*nodes: Node3D) -> np.ndarray:
        """Returns the flattened list of global DOF indices for the supplied nodes.

        Example for a 2-node member:
            [i_node*6 + (0..5), j_node*6 + (0..5)] -> 12 indices total.

        Once this vector is created we can operate on entire element sub-matrices via
        numpy broadcasting, instead of repeating the ``node.ID*6 + local_dof`` math in
        Python loops.
        """
        # Preallocate the DOF array (nodes * 6 DOFs each) as 64-bit ints.
        dofs = np.empty(len(nodes) * 6, dtype=np.int64)

        # Build a template 0..5 array to shift per node.
        local = np.arange(6, dtype=np.int64)

        # Iterate through each supplied node with its ordinal index.
        for i, node in enumerate(nodes):
            # Compute the slice start for this node's 6 DOFs.
            start = i * 6
            # Fill the slice with the node's base DOF plus the 0..5 offsets.
            dofs[start:start + 6] = node.ID * 6 + local

        return dofs

    @staticmethod
    def _append_sparse_block(dofs: np.ndarray, block: np.ndarray,
                             row_parts: list, col_parts: list,
                             data_parts: list) -> None:
        """Converts an element sub-matrix into row/col/data arrays for COO assembly.

        Compared to nested loops, this function handles the conversion in
        three numpy statements using repeat/tile operations.
        """
        # Ensure we are working with a float ndarray copy of the element block.
        block = np.asarray(block, dtype=float)

        # Cache the number of DOFs
        size = dofs.size

        # Flatten the element block into a 1-D vector
        flat = block.reshape(-1)

        # Skip work entirely if this block contains only zeros
        nonzero_mask = flat != 0.0
        if not np.any(nonzero_mask):
            return

        # Build the repeated row indices for the full set of row/column combinations
        rows = np.repeat(dofs, size)

        # Build the tiled column indices
        cols = np.tile(dofs, size)

        # Append only the nonzero entries
        row_parts.append(rows[nonzero_mask])
        col_parts.append(cols[nonzero_mask])
        data_parts.append(flat[nonzero_mask])

    @staticmethod
    def _add_dense_block(global_matrix: np.ndarray, dofs: np.ndarray, block: np.ndarray) -> None:
        """Adds an element block to the dense global matrix using vectorized indexing.

        ``np.ix_(dofs, dofs)`` builds every row/column combination of those DOFs, letting the
        12x12 or 24x24 block be summed in a single vectorized add.
        """
        # Convert the block to a float ndarray so dtype math aligns with the global matrix.
        block = np.asarray(block, dtype=float)

        # Use numpy advanced indexing to add the entire block in one statement.
        global_matrix[np.ix_(dofs, dofs)] += block

    @property
    def load_cases(self) -> List[str]:
        """Returns a list of all the load cases in the model (in alphabetical order).
        """

        # Create an empty list of load cases
        cases: List[str] = []

        # Step through each node
        for node in self.nodes.values():
            # Step through each nodal load
            for load in node.NodeLoads:
                # Get the load case for each nodal laod
                cases.append(load[2])

        # Step through each member
        for member in self.members.values():
            # Step through each member point load
            for load in member.PtLoads:
                # Get the load case for each member point load
                cases.append(load[3])
            # Step through each member distributed load
            for load in member.DistLoads:
                # Get the load case for each member distributed load
                cases.append(load[5])

        # Step through each plate/quad
        for plate in list(self.plates.values()) + list(self.quads.values()):
            # Step through each surface load
            for load in plate.pressures:
                # Get the load case for each plate/quad pressure
                cases.append(load[1])

        # Remove duplicates and return the list (sorted ascending)
        return sorted(list(dict.fromkeys(cases)))

    def add_node(self, name: str, X: float, Y: float, Z: float) -> str:
        """Adds a new node to the model.

        :param name: A unique user-defined name for the node. If set to None or "" a name will be
                     automatically assigned.
        :type name: str
        :param X: The node's global X-coordinate.
        :type X: number
        :param Y: The node's global Y-coordinate.
        :type Y: number
        :param Z: The node's global Z-coordinate.
        :type Z: number
        :raises NameError: Occurs when the specified name already exists in the model.
        :return: The name of the node added to the model.
        :rtype: str
        """

        # Name the node or check it doesn't already exist
        if name:
            if name in self.nodes:
                raise NameError(f"Node name '{name}' already exists")
        else:
            # As a guess, start with the length of the dictionary
            name = "N" + str(len(self.nodes))
            count = 1
            while name in self.nodes: 
                name = "N" + str(len(self.nodes) + count)
                count += 1

        # Create a new node
        new_node = Node3D(name, X, Y, Z)

        # Add the new node to the model
        self.nodes[name] = new_node

        # Flag the model as unsolved
        self.solution = None

        # Return the node name
        return name

    def add_material(self, name: str, E: float, G: float, nu: float, rho: float, fy: float | None = None) -> str:
        """Adds a new material to the model.

        :param name: A unique user-defined name for the material.
        :type name: str
        :param E: The modulus of elasticity of the material.
        :type E: number
        :param G: The shear modulus of elasticity of the material.
        :type G: number
        :param nu: Poisson's ratio of the material.
        :type nu: number
        :param rho: The density of the material
        :type rho: number
        :raises NameError: Occurs when the specified name already exists in the model.
        """

        # Name the material or check it doesn't already exist
        if name:
            if name in self.materials:
                raise NameError(f"Material name '{name}' already exists")
        else:
            # As a guess, start with the length of the dictionary
            name = "M" + str(len(self.materials))
            count = 1
            while name in self.materials:
                name = "M" + str(len(self.materials) + count)
                count += 1

        # Create a new material
        new_material = Material(self, name, E, G, nu, rho, fy)

        # Add the new material to the model
        self.materials[name] = new_material

        # Flag the model as unsolved
        self.solution = None

        # Return the materal name
        return name

    def add_section(self, name: str, A: float, Iy: float, Iz: float, J: float) -> str:
        """Adds a cross-section to the model.

        :param name: A unique name for the cross-section.
        :type name: string
        :param name: Name of the section
        :type name: str
        :param A: Cross-sectional area of the section
        :type A: float
        :param Iy: The second moment of area the section about the Y (minor) axis
        :type Iy: float
        :param Iz: The second moment of area the section about the Z (major) axis
        :type Iz: float
        :param J: The torsion constant of the section
        :type J: float
        """

        # Name the section or check it doesn't already exist
        if name:
            if name in self.sections:
                raise NameError(f"Section name '{name}' already exists")
        else:
            # As a guess, start with the length of the dictionary
            name = "SC" + str(len(self.sections))
            count = 1
            while name in self.sections: 
                name = "SC" + str(len(self.sections) + count)
                count += 1

        # Add the new section to the model
        self.sections[name] = Section(self, name, A, Iy, Iz, J)

        # Return the section name
        return name

    def add_steel_section(self, name: str, A: float, Iy: float, Iz: float, J: float, Zy: float, Zz: float, material_name: str) -> str:
        """Adds a cross-section to the model.

        :param name: A unique name for the cross-section.
        :type name: string
        :param name: Name of the section
        :type name: str
        :param A: Cross-sectional area of the section
        :type A: float
        :param Iy: The second moment of area the section about the Y (minor) axis
        :type Iy: float
        :param Iz: The second moment of area the section about the Z (major) axis
        :type Iz: float
        :param J: The torsion constant of the section
        :type J: float
        :param Zy: The section modulus about the Y (minor) axis
        :type Zy: float
        :param Zz: The section modulus about the Z (major) axis
        :type Zz: float
        :param material_name: The name of the steel material
        :type material_name: str
        """

        # Name the section or check it doesn't already exist
        if name:
            if name in self.sections:
                raise NameError(f"Section name '{name}' already exists")
        else:
            # As a guess, start with the length of the dictionary
            name = "SC" + str(len(self.sections))
            count = 1
            while name in self.sections: 
                name = "SC" + str(len(self.sections) + count)
                count += 1

        # Add the new section to the model
        self.sections[name] = SteelSection(self, name, A, Iy, Iz, J, Zy, Zz, material_name)

        # Return the section name
        return name

    def add_spring(self, name: str, i_node: str, j_node: str, ks: float, tension_only: bool = False, comp_only: bool = False) -> str:
        """Adds a new spring to the model.

        :param name: A unique user-defined name for the member. If None or "", a name will be
                    automatically assigned
        :type name: str
        :param i_node: The name of the i-node (start node).
        :type i_node: str
        :param j_node: The name of the j-node (end node).
        :type j_node: str
        :param ks: The spring constant (force/displacement).
        :type ks: number
        :param tension_only: Indicates if the member is tension-only, defaults to False
        :type tension_only: bool, optional
        :param comp_only: Indicates if the member is compression-only, defaults to False
        :type comp_only: bool, optional
        :raises NameError: Occurs when the specified name already exists in the model.
        :return: The name of the spring that was added to the model.
        :rtype: str
        """

        # Name the spring or check it doesn't already exist
        if name:
            if name in self.springs:
                raise NameError(f"Spring name '{name}' already exists")
        else:
            # As a guess, start with the length of the dictionary
            name = "S" + str(len(self.springs))
            count = 1
            while name in self.springs: 
                name = "S" + str(len(self.springs) + count)
                count += 1

        # Lookup node names and safely handle exceptions
        try:
            pn_nodes = [self.nodes[node_name] for node_name in (i_node, j_node)]
        except KeyError as e:
            raise NameError(f"Node '{e.args[0]}' does not exist in the model")

        # Create a new spring
        new_spring = Spring3D(name, pn_nodes[0], pn_nodes[1],
                              ks, self.load_combos, tension_only=tension_only,
                              comp_only=comp_only)

        # Add the new spring to the model
        self.springs[name] = new_spring

        # Flag the model as unsolved
        self.solution = None

        # Return the spring name
        return name

    def add_member(self, name: str, i_node: str, j_node: str, material_name: str, section_name: str, rotation: float = 0.0, tension_only: bool = False, comp_only: bool = False) -> str:
        """Adds a new physical member to the model.

        :param name: A unique user-defined name for the member. If ``None`` or ``""``, a name will be automatically assigned
        :type name: str
        :param i_node: The name of the i-node (start node).
        :type i_node: str
        :param j_node: The name of the j-node (end node).
        :type j_node: str
        :param material_name: The name of the material of the member.
        :type material_name: str
        :param section_name: The name of the cross section to use for section properties.
        :type section_name: string
        :param rotation: The angle of rotation (degrees) of the member cross-section about its longitudinal (local x) axis. Default is 0.
        :type rotation: float, optional
        :param tension_only: Indicates if the member is tension-only, defaults to False
        :type tension_only: bool, optional
        :param comp_only: Indicates if the member is compression-only, defaults to False
        :type comp_only: bool, optional
        :raises NameError: Occurs if the specified name already exists.
        :return: The name of the member added to the model.
        :rtype: str
        """

        # Name the member or check it doesn't already exist
        if name:
            if name in self.members:
                raise NameError(f"Member name '{name}' already exists")
        else:
            # As a guess, start with the length of the dictionary
            name = "M" + str(len(self.members))
            count = 1
            while name in self.members: 
                name = "M" + str(len(self.members)+count)
                count += 1

        # Lookup node names and safely handle exceptions
        try:
            pn_nodes = [self.nodes[node_name] for node_name in (i_node, j_node)]
        except KeyError as e:
            raise NameError(f"Node '{e.args[0]}' does not exist in the model")

        # Create a new member
        new_member = PhysMember(self, name, pn_nodes[0], pn_nodes[1], material_name, section_name, rotation=rotation, tension_only=tension_only, comp_only=comp_only)

        # Add the new member to the model
        self.members[name] = new_member

        # Flag the model as unsolved
        self.solution = None

        # Return the member name
        return name

    def add_plate(self, name: str, i_node: str, j_node: str, m_node: str, n_node: str, t: float, material_name: str, kx_mod: float = 1.0, ky_mod: float = 1.0) -> str:
        """Adds a new rectangular plate to the model. The plate formulation for in-plane (membrane)
        stiffness is based on an isoparametric formulation. For bending, it is based on a 12-term
        polynomial formulation. This element must be rectangular, and must not be used where a
        thick plate formulation is needed. For a more versatile plate element that can handle
        distortion and thick plate conditions, consider using the `add_quad` method instead.

        :param name: A unique user-defined name for the plate. If None or "", a name will be
                     automatically assigned.
        :type name: str
        :param i_node: The name of the i-node.
        :type i_node: str
        :param j_node: The name of the j-node.
        :type j_node: str
        :param m_node: The name of the m-node.
        :type m_node: str
        :param n_node: The name of the n-node.
        :type n_node: str
        :param t: The thickness of the element.
        :type t: number
        :param material_name: The name of the material for the element.
        :type material_name: str
        :param kx_mod: Stiffness modification factor for in-plane stiffness in the element's local
                       x-direction, defaults to 1 (no modification).
        :type kx_mod: number, optional
        :param ky_mod: Stiffness modification factor for in-plane stiffness in the element's local
                       y-direction, defaults to 1 (no modification).
        :type ky_mod: number, optional
        :raises NameError: Occurs when the specified name already exists in the model.
        :return: The name of the element added to the model.
        :rtype: str
        """

        # Name the plate or check it doesn't already exist
        if name:
            if name in self.plates:
                raise NameError(f"Plate name '{name}' already exists")
        else:
            # As a guess, start with the length of the dictionary
            name = "P" + str(len(self.plates))
            count = 1
            while name in self.plates:
                name = "P" + str(len(self.plates)+count)
                count += 1

        # Lookup node names and safely handle exceptions
        try:
            pn_nodes = [self.nodes[node_name] for node_name in (i_node, j_node, m_node, n_node)]
        except KeyError as e:
            raise NameError(f"Node '{e.args[0]}' does not exist in the model")

        # Create a new plate
        new_plate = Plate3D(name, pn_nodes[0], pn_nodes[1], pn_nodes[2], pn_nodes[3],
                            t, material_name, self, kx_mod, ky_mod)

        # Add the new plate to the model
        self.plates[name] = new_plate

        # Flag the model as unsolved
        self.solution = None

        # Return the plate name
        return name

    def add_quad(self, name: str, i_node: str, j_node: str, m_node: str, n_node: str,
                 t: float, material_name: str, kx_mod: float = 1.0, ky_mod: float = 1.0) -> str:
        """Adds a new quadrilateral to the model. The quad formulation for in-plane (membrane)
        stiffness is based on an isoparametric formulation. For bending, it is based on an MITC4
        formulation. This element handles distortion relatively well, and is appropriate for thick
        and thin plates. One limitation with this element is that it does a poor job of reporting
        corner stresses. Corner forces, however are very accurate. Center stresses are very
        accurate as well. For cases where corner stress results are important, consider using the
        `add_plate` method instead.

        :param name: A unique user-defined name for the quadrilateral. If None or "", a name will
                     be automatically assigned.
        :type name: str
        :param i_node: The name of the i-node.
        :type i_node: str
        :param j_node: The name of the j-node.
        :type j_node: str
        :param m_node: The name of the m-node.
        :type m_node: str
        :param n_node: The name of the n-node.
        :type n_node: str
        :param t: The thickness of the element.
        :type t: number
        :param material_name: The name of the material for the element.
        :type material_name: str
        :param kx_mod: Stiffness modification factor for in-plane stiffness in the element's local
            x-direction, defaults to 1 (no modification).
        :type kx_mod: number, optional
        :param ky_mod: Stiffness modification factor for in-plane stiffness in the element's local
            y-direction, defaults to 1 (no modification).
        :type ky_mod: number, optional
        :raises NameError: Occurs when the specified name already exists in the model.
        :return: The name of the element added to the model.
        :rtype: str
        """

        # Name the quad or check it doesn't already exist
        if name:
            if name in self.quads:
                raise NameError(f"Quad name '{name}' already exists")
        else:
            # As a guess, start with the length of the dictionary
            name = "Q" + str(len(self.quads))
            count = 1
            while name in self.quads:
                name = "Q" + str(len(self.quads) + count)
                count += 1

        # Lookup node names and safely handle exceptions
        try:
            pn_nodes = [self.nodes[node_name] for node_name in (i_node, j_node, m_node, n_node)]
        except KeyError as e:
            raise NameError(f"Node '{e.args[0]}' does not exist in the model")

        # Create a new member
        new_quad = Quad3D(name, pn_nodes[0], pn_nodes[1], pn_nodes[2], pn_nodes[3],
                          t, material_name, self, kx_mod, ky_mod)

        # Add the new member to the model
        self.quads[name] = new_quad

        # Flag the model as unsolved
        self.solution = None

        # Return the quad name
        return name

    def add_rectangle_mesh(self, name: str, mesh_size: float, width: float, height: float, thickness: float, material_name: str, kx_mod: float = 1.0, ky_mod: float = 1.0, origin: list | tuple = (0, 0, 0), plane: str = 'XY', x_control: list | None = None, y_control: list | None = None, start_node: str | None = None, start_element: str | None = None, element_type: str = 'Quad') -> str:
        """Adds a rectangular mesh of elements to the model.

        :param name: A unique name for the mesh.
        :type name: str
        :param mesh_size: The desired mesh size.
        :type mesh_size: number
        :param width: The overall width of the rectangular mesh measured along its local x-axis.
        :type width: number
        :param height: The overall height of the rectangular mesh measured along its local y-axis.
        :type height: number
        :param thickness: The thickness of each element in the mesh.
        :type thickness: number
        :param material_name: The name of the material for elements in the mesh.
        :type material_name: str
        :param kx_mod: Stiffness modification factor for in-plane stiffness in the element's local x-direction. Defaults to 1.0 (no modification).
        :type kx_mod: float, optional
        :param ky_mod: Stiffness modification factor for in-plane stiffness in the element's local y-direction. Defaults to 1.0 (no modification).
        :type ky_mod: float, optional
        :param origin: The origin of the regtangular mesh's local coordinate system. Defaults to [0, 0, 0]
        :type origin: list, optional
        :param plane: The plane the mesh will be parallel to. Options are 'XY', 'YZ', and 'XZ'. Defaults to 'XY'.
        :type plane: str, optional
        :param x_control: A list of control points along the mesh's local x-axis to work into the mesh. Defaults to `None`.
        :type x_control: list, optional
        :param y_control: A list of control points along the mesh's local y-axis to work into the mesh. Defaults to None.
        :type y_control: list, optional
        :param start_node: The name of the first node in the mesh. If set to `None` the program will use the next available node name. Default is `None`.
        :type start_node: str, optional
        :param start_element: The name of the first element in the mesh. If set to `None` the program will use the next available element name. Default is `None`.
        :type start_element: str, optional
        :param element_type: They type of element to make the mesh out of. Either 'Quad' or 'Rect'. Defaults to 'Quad'.
        :type element_type: str, optional
        :raises NameError: Occurs when the specified name already exists in the model.
        :return: The name of the mesh added to the model.
        :rtype: str
        """
        
        # Check if a mesh name has been provided
        if name:
            # Check that the mesh name isn't already being used
            if name in self.meshes: raise NameError(f"Mesh name '{name}' already exists")
        # Rename the mesh if necessary
        else:
            name = self.unique_name(self.meshes, 'MSH')
        
        # Identify the starting node and element
        if start_node is None:
            start_node = self.unique_name(self.nodes, 'N')
        if element_type == 'Rect' and start_element is None:
            start_element = self.unique_name(self.plates, 'R')
        elif element_type == 'Quad' and start_element is None:
            start_element = self.unique_name(self.quads, 'Q')
        
        # Create the mesh
        new_mesh = RectangleMesh(mesh_size, width, height, thickness, material_name, self, kx_mod,
                                 ky_mod, origin, plane, x_control, y_control, start_node,
                                 start_element, element_type=element_type)

        # Add the new mesh to the `Meshes` dictionary
        self.meshes[name] = new_mesh

        # Flag the model as unsolved
        self.solution = None

        # Return the mesh's name
        return name

    def add_annulus_mesh(self, name: str, mesh_size: float, outer_radius: float, inner_radius: float, thickness: float, material_name: str, kx_mod: float = 1.0, ky_mod: float = 1.0, origin: list | tuple = (0, 0, 0), axis: str = 'Y', start_node: str | None = None, start_element: str | None = None) -> str:
        """Adds a mesh of quadrilaterals forming an annulus (a donut).

        :param name: A unique name for the mesh.
        :type name: str
        :param mesh_size: The target mesh size.
        :type mesh_size: float
        :param outer_radius: The radius to the outside of the annulus.
        :type outer_radius: float
        :param inner_radius: The radius to the inside of the annulus.
        :type inner_radius: float
        :param thickness: Element thickness.
        :type thickness: float
        :param material_name: The name of the element material.
        :type material_name: str
        :param kx_mod: Stiffness modification factor for radial stiffness in the element's local
                       x-direction. Default is 1.0 (no modification).
        :type kx_mod: float, optional
        :param ky_mod: Stiffness modification factor for meridional stiffness in the element's
                       local y-direction. Default is 1.0 (no modification).
        :type ky_mod: float, optional
        :param origin: The origin of the mesh. The default is [0, 0, 0].
        :type origin: list, optional
        :param axis: The global axis about which the mesh will be generated. The default is 'Y'.
        :type axis: str, optional
        :param start_node: The name of the first node in the mesh. If set to `None` the program
                           will use the next available node name. Default is `None`.
        :type start_node: str, optional
        :param start_element: The name of the first element in the mesh. If set to `None` the
                              program will use the next available element name. Default is `None`.
        :type start_element: str, optional
        :raises NameError: Occurs if the specified name already exists in the model.
        :return: The name of the mesh added to the model.
        :rtype: str
        """

        # Check if a mesh name has been provided
        if name:
            # Check that the mesh name doesn't already exist
            if name in self.meshes: raise NameError(f"Mesh name '{name}' already exists")
        # Give the mesh a new name if necessary
        else:
            name = self.unique_name(self.meshes, 'MSH')

        # Identify the starting node and element
        if start_node is None:
            start_node = self.unique_name(self.nodes, 'N')
        if start_element is None:
            start_element = self.unique_name(self.quads, 'Q')

        # Create a new mesh
        new_mesh = AnnulusMesh(mesh_size, outer_radius, inner_radius, thickness, material_name, self,
                               kx_mod, ky_mod, origin, axis, start_node, start_element)

        # Add the new mesh to the `Meshes` dictionary
        self.meshes[name] = new_mesh

        # Flag the model as unsolved
        self.solution = None

        # Return the mesh's name
        return name

    def add_frustrum_mesh(self, name: str, mesh_size: float, large_radius: float, small_radius: float, height: float, thickness: float, material_name: str, kx_mod: float = 1.0, ky_mod: float = 1.0, origin: list | tuple = (0, 0, 0), axis: str = 'Y', start_node: str | None = None, start_element: str | None = None) -> str:
        """Adds a mesh of quadrilaterals forming a frustrum (a cone intersected by a horizontal plane).

        :param name: A unique name for the mesh.
        :type name: str
        :param mesh_size: The target mesh size
        :type mesh_size: number
        :param large_radius: The larger of the two end radii.
        :type large_radius: number
        :param small_radius: The smaller of the two end radii.
        :type small_radius: number
        :param height: The height of the frustrum.
        :type height: number
        :param thickness: The thickness of the elements.
        :type thickness: number
        :param material_name: The name of the element material.
        :type material_name: str
        :param kx_mod: Stiffness modification factor for radial stiffness in each element's local x-direction, defaults to 1 (no modification).
        :type kx_mod: number, optional
        :param ky_mod: Stiffness modification factor for meridional stiffness in each element's local y-direction, defaults to 1 (no modification).
        :type ky_mod: number, optional
        :param origin: The origin of the mesh, defaults to [0, 0, 0].
        :type origin: list, optional
        :param axis: The global axis about which the mesh will be generated, defaults to 'Y'.
        :type axis: str, optional
        :param start_node: The name of the first node in the mesh. If set to None the program will use the next available node name, defaults to None.
        :type start_node: str, optional
        :param start_element: The name of the first element in the mesh. If set to `None` the
                              program will use the next available element name, defaults to None
        :type start_element: str, optional
        :raises NameError: Occurs if the specified name already exists.
        :return: The name of the mesh added to the model.
        :rtype: str
        """

        # Check if a name has been provided
        if name:
            # Check that the mesh name doesn't already exist
            if name in self.meshes: raise NameError(f"Mesh name '{name}' already exists")
        # Give the mesh a new name if necessary
        else:
            name = self.unique_name(self.meshes, 'MSH')

        # Identify the starting node and element
        if start_node is None:
            start_node = self.unique_name(self.nodes, 'N')
        if start_element is None:
            start_element = self.unique_name(self.quads, 'Q')

        # Create a new mesh
        new_mesh = FrustrumMesh(mesh_size, large_radius, small_radius, height, thickness, material_name,
                                self, kx_mod, ky_mod, origin, axis, start_node, start_element)

        # Add the new mesh to the `Meshes` dictionary
        self.meshes[name] = new_mesh

        # Flag the model as unsolved
        self.solution = None

        # Return the mesh's name
        return name

    def add_cylinder_mesh(self, name:str, mesh_size:float, radius:float, height:float,
                          thickness:float, material_name:str, kx_mod:float = 1,
                          ky_mod:float = 1, origin:list | tuple = (0, 0, 0),
                          axis:str = 'Y', num_elements:int | None = None,
                          start_node: str | None = None, start_element:str | None = None,
                          element_type:str = 'Quad') -> str:
        """Adds a mesh of elements forming a cylinder.

        :param name: A unique name for the mesh.
        :type name: str
        :param mesh_size: The target mesh size.
        :type mesh_size: float
        :param radius: The radius of the cylinder.
        :type radius: float
        :param height: The height of the cylinder.
        :type height: float
        :param thickness: Element thickness.
        :type thickness: float
        :param material_name: The name of the element material.
        :type material_name: str
        :param kx_mod: Stiffness modification factor for hoop stiffness in each element's local
                       x-direction. Defaults to 1.0 (no modification).
        :type kx_mod: int, optional
        :param ky_mod: Stiffness modification factor for meridional stiffness in each element's
                       local y-direction. Defaults to 1.0 (no modification).
        :type ky_mod: int, optional
        :param origin: The origin [X, Y, Z] of the mesh. Defaults to [0, 0, 0].
        :type origin: list, optional
        :param axis: The global axis about which the mesh will be generated. Defaults to 'Y'.
        :type axis: str, optional
        :param num_elements: The number of elements to use to form each course of elements. This
                             is typically only used if you are trying to match the nodes to another
                             mesh's nodes. If set to `None` the program will automatically
                             calculate the number of elements to use based on the mesh size.
                             Defaults to None.
        :type num_elements: int, optional
        :param start_node: The name of the first node in the mesh. If set to `None` the program
                           will use the next available node name. Defaults to `None`.
        :type start_node: str, optional
        :param start_element: The name of the first element in the mesh. If set to `None` the
                              program will use the next available element name. Defaults to `None`.
        :type start_element: str, optional
        :param element_type: The type of element to make the mesh out of. Either 'Quad' or 'Rect'.
                             Defaults to 'Quad'.
        :type element_type: str, optional
        :raises NameError: Occurs when the specified mesh name is already being used in the model.
        :return: The name of the mesh added to the model
        :rtype: str
        """            
        
        # Check if a name has been provided
        if name:
            # Check that the mesh name doesn't already exist
            if name in self.meshes: raise NameError(f"Mesh name '{name}' already exists")
        # Give the mesh a new name if necessary
        else:
            name = self.unique_name(self.meshes, 'MSH')

        # Identify the starting node and element
        if start_node is None:
            start_node = self.unique_name(self.nodes, 'N')
        if element_type == 'Rect' and start_element is None:
            start_element = self.unique_name(self.plates, 'R')
        elif element_type == 'Quad' and start_element is None:
            start_element = self.unique_name(self.quads, 'Q')
        
        # Create a new mesh
        new_mesh = CylinderMesh(mesh_size, radius, height, thickness, material_name, self,
                               kx_mod, ky_mod, origin, axis, start_node, start_element,
                               num_elements, element_type)

        # Add the new mesh to the `Meshes` dictionary
        self.meshes[name] = new_mesh

        # Flag the model as unsolved
        self.solution = None
        
        # Return the mesh's name
        return name

    def add_shear_wall(self, name: str, mesh_size: float, length: float, height: float, thickness: float, material_name: str, ky_mod: float = 0.35, plane: Literal['XY', 'YZ'] = 'XY', origin: List[float] = [0, 0, 0]):

        # Create a new shear wall
        new_shear_wall = ShearWall(self, name, mesh_size, length, height, thickness, material_name, ky_mod, origin, plane)

        # Add the wall to the model
        self.shear_walls[name] = new_shear_wall

    def merge_duplicate_nodes(self, tolerance:float = 0.001) -> list[tuple[str, str]]:
        """Removes duplicate nodes from the model and returns a list of tuples showing which nodes were merged.

        :param tolerance: The maximum distance between two nodes in order to consider them duplicates. Defaults to 0.001.
        :type tolerance: float, optional
        :return: A list of tuples where each tuple contains (deleted_node_name, merged_into_node_name).
        """

        # Initialize a dictionary marking where each node is used
        node_lookup = {node_name: [] for node_name in self.nodes.keys()}
        element_dicts = ('springs', 'members', 'plates', 'quads')
        node_types = ('i_node', 'j_node', 'm_node', 'n_node')

        # Step through each dictionary of elements in the model (springs, members, plates, quads)
        for element_dict in element_dicts:

            # Step through each element in the dictionary
            for element in getattr(self, element_dict).values():

                # Step through each possible node type in the element (i-node, j-node, m-node, n-node)
                for node_type in node_types:

                    # Get the current element's node having the current type
                    # Return `None` if the element doesn't have this node type
                    node = getattr(element, node_type, None)

                    # Determine if the node exists on the element
                    if node is not None:
                        # Add the element to the list of elements attached to the node
                        node_lookup[node.name].append((element, node_type))

        # Make a list of the names of each node in the model
        node_names = list(self.nodes.keys())

        # Make a list of merge mappings (deleted_node, merged_into_node)
        merge_list = []

        # Step through each node in the copy of the `Nodes` dictionary
        for i, node_1_name in enumerate(node_names):

            # Skip iteration if `node_1` has already been removed
            if node_lookup[node_1_name] is None:
                continue

            # There is no need to check `node_1` against itself
            for node_2_name in node_names[i + 1:]:

                # Skip iteration if node_2 has already been removed
                if node_lookup[node_2_name] is None:
                    continue

                # Calculate the distance between nodes
                if self.nodes[node_1_name].distance(self.nodes[node_2_name]) > tolerance:
                    continue

                # Replace references to `node_2` in each element with references to `node_1`
                for element, node_type in node_lookup[node_2_name]:
                    setattr(element, node_type, self.nodes[node_1_name])

                # Flag `node_2` as no longer used
                node_lookup[node_2_name] = None

                # Merge any boundary conditions
                support_cond = ('support_DX', 'support_DY', 'support_DZ', 'support_RX', 'support_RY', 'support_RZ')
                for dof in support_cond:
                    if getattr(self.nodes[node_2_name], dof) == True:
                        setattr(self.nodes[node_1_name], dof, True)
                
                # Merge any spring supports
                spring_cond = ('spring_DX', 'spring_DY', 'spring_DZ', 'spring_RX', 'spring_RY', 'spring_RZ')
                for dof in spring_cond:
                    value = getattr(self.nodes[node_2_name], dof)
                    if value != [None, None, None]:
                        setattr(self.nodes[node_1_name], dof, value)
                
                # Fix the mesh labels
                for mesh in self.meshes.values():

                    # Fix the nodes in the mesh
                    if node_2_name in mesh.nodes.keys():

                        # Attach the correct node to the mesh
                        mesh.nodes[node_2_name] = self.nodes[node_1_name]

                        # Fix the dictionary key
                        mesh.nodes[node_1_name] = mesh.nodes.pop(node_2_name)

                    # Fix the elements in the mesh
                    for element in mesh.elements.values():
                        if node_2_name == element.i_node.name: element.i_node = self.nodes[node_1_name]
                        if node_2_name == element.j_node.name: element.j_node = self.nodes[node_1_name]
                        if node_2_name == element.m_node.name: element.m_node = self.nodes[node_1_name]
                        if node_2_name == element.n_node.name: element.n_node = self.nodes[node_1_name]
                    
                # Add the node mapping to the merge list (deleted_node, merged_into_node)
                merge_list.append((node_2_name, node_1_name))

        # Remove `node_2` from the model's `Nodes` dictionary
        for node_name, _ in merge_list:
            self.nodes.pop(node_name)

        # Flag the model as unsolved
        self.solution = None

        # Return the list of merge mappings (deleted_node, merged_into_node)
        return merge_list

    def delete_node(self, node_name:str):
        """Removes a node from the model. All nodal loads associated with the node and elements attached to the node will also be removed.

        :param node_name: The name of the node to be removed.
        :type node_name: str
        """
            
        # Remove the node. Nodal loads are stored within the node, so they
        # will be deleted automatically when the node is deleted.
        self.nodes.pop(node_name)
        
        # Find any elements attached to the node and remove them
        self.members = {name: member for name, member in self.members.items() if member.i_node.name != node_name and member.j_node.name != node_name}
        self.plates = {name: plate for name, plate in self.plates.items() if plate.i_node.name != node_name and plate.j_node.name != node_name and plate.m_node.name != node_name and plate.n_node.name != node_name}
        self.quads = {name: quad for name, quad in self.quads.items() if quad.i_node.name != node_name and quad.j_node.name != node_name and quad.m_node.name != node_name and quad.n_node.name != node_name}

        # Flag the model as unsolved
        self.solution = None

    def delete_spring(self, spring_name:str):
        """Removes a spring from the model.

        :param spring_name: The name of the spring to be removed.
        :type spring_name: str
        """
        
        # Remove the spring
        self.springs.pop(spring_name)

        # Flag the model as unsolved
        self.solution = None

    def delete_member(self, member_name:str):
        """Removes a member from the model. All member loads associated with the member will also
           be removed.

        :param member_name: The name of the member to be removed.
        :type member_name: str
        """
        
        # Remove the member. Member loads are stored within the member, so they
        # will be deleted automatically when the member is deleted.
        self.members.pop(member_name)

        # Flag the model as unsolved
        self.solution = None
        
    def def_support(self, node_name:str, support_DX:bool=False, support_DY:bool=False,
                    support_DZ:bool=False, support_RX:bool=False, support_RY:bool=False,
                    support_RZ:bool=False):
        """Defines the support conditions at a node. Nodes will default to fully unsupported
           unless specified otherwise.

        :param node_name: The name of the node where the support is being defined.
        :type node_name: str
        :param support_DX: Indicates whether the node is supported against translation in the
                           global X-direction. Defaults to False.
        :type support_DX: bool, optional
        :param support_DY: Indicates whether the node is supported against translation in the
                           global Y-direction. Defaults to False.
        :type support_DY: bool, optional
        :param support_DZ: Indicates whether the node is supported against translation in the
                           global Z-direction. Defaults to False.
        :type support_DZ: bool, optional
        :param support_RX: Indicates whether the node is supported against rotation about the
                           global X-axis. Defaults to False.
        :type support_RX: bool, optional
        :param support_RY: Indicates whether the node is supported against rotation about the
                           global Y-axis. Defaults to False.
        :type support_RY: bool, optional
        :param support_RZ: Indicates whether the node is supported against rotation about the
                           global Z-axis. Defaults to False.
        :type support_RZ: bool, optional
        """            
        
        # Get the node to be supported
        try:
            node = self.nodes[node_name]
        except KeyError:
            raise NameError(f"Node '{node_name}' does not exist in the model")
                   
        # Set the node's support conditions
        node.support_DX = support_DX
        node.support_DY = support_DY
        node.support_DZ = support_DZ
        node.support_RX = support_RX
        node.support_RY = support_RY
        node.support_RZ = support_RZ

        # Flag the model as unsolved
        self.solution = None

    def def_support_spring(self, node_name:str, dof:str, stiffness:float, direction:str | None = None):
        """Defines a spring support at a node.

        :param node_name: The name of the node to apply the spring support to.
        :type node_name: str
        :param dof: The degree of freedom to apply the spring support to.
        :type dof: str ('DX', 'DY', 'DZ', 'RX', 'RY', or 'RZ')
        :param stiffness: The translational or rotational stiffness of the spring support.
        :type stiffness: float
        :param direction: The direction in which the spring can act. '+' allows the spring to resist positive displacements. '-' allows the spring to resist negative displacements. None allows the spring to act in both directions. Default is None.
        :type direction: str or None ('+', '-', None), optional
        :raises ValueError: Occurs when an invalid support spring direction has been specified.
        :raises ValueError: Occurs when an invalid support spring degree of freedom has been specified.
        """        
        
        if dof in ('DX', 'DY', 'DZ', 'RX', 'RY', 'RZ'):
            if direction in ('+', '-', None):
                try:
                    if dof == 'DX':
                        self.nodes[node_name].spring_DX = [stiffness, direction, True]
                    elif dof == 'DY':
                        self.nodes[node_name].spring_DY = [stiffness, direction, True]
                    elif dof == 'DZ':
                        self.nodes[node_name].spring_DZ = [stiffness, direction, True]
                    elif dof == 'RX':
                        self.nodes[node_name].spring_RX = [stiffness, direction, True]
                    elif dof == 'RY':
                        self.nodes[node_name].spring_RY = [stiffness, direction, True]
                    elif dof == 'RZ':
                        self.nodes[node_name].spring_RZ = [stiffness, direction, True]
                except KeyError:
                    raise NameError(f"Node '{node_name}' does not exist in the model")
            else:
                raise ValueError('Invalid support spring direction. Specify \'+\', \'-\', or None.')
        else:
            raise ValueError('Invalid support spring degree of freedom. Specify \'DX\', \'DY\', \'DZ\', \'RX\', \'RY\', or \'RZ\'')
        
        # Flag the model as unsolved
        self.solution = None

    def def_node_disp(self, node_name:str, direction:str, magnitude:float): 
        """Defines a nodal displacement at a node.

        :param node_name: The name of the node where the nodal displacement is being applied.
        :type node_name: str
        :param direction: The global direction the nodal displacement is being applied in. Displacements are 'DX', 'DY', and 'DZ'. Rotations are 'RX', 'RY', and 'RZ'.
        :type direction: str
        :param magnitude: The magnitude of the displacement.
        :type magnitude: float
        :raises ValueError: _description_
        """
            
        # Validate the value of direction
        if direction not in ('DX', 'DY', 'DZ', 'RX', 'RY', 'RZ'):
            raise ValueError(f"direction must be 'DX', 'DY', 'DZ', 'RX', 'RY', or 'RZ'. {direction} was given.")
        
        # Get the node
        try:
            node = self.nodes[node_name]
        except KeyError:
            raise NameError(f"Node '{node_name}' does not exist in the model")

        if direction == 'DX':
            node.EnforcedDX = magnitude
        if direction == 'DY':
            node.EnforcedDY = magnitude
        if direction == 'DZ':
            node.EnforcedDZ = magnitude
        if direction == 'RX':
            node.EnforcedRX = magnitude
        if direction == 'RY':
            node.EnforcedRY = magnitude
        if direction == 'RZ':
            node.EnforcedRZ = magnitude
        
        # Flag the model as unsolved
        self.solution = None

    def def_releases(self, member_name:str, Dxi:bool=False, Dyi:bool=False, Dzi:bool=False,
                     Rxi:bool=False, Ryi:bool=False, Rzi:bool=False,
                     Dxj:bool=False, Dyj:bool=False, Dzj:bool=False,
                     Rxj:bool=False, Ryj:bool=False, Rzj:bool=False):
        """Defines member end realeses for a member. All member end releases will default to unreleased unless specified otherwise.

        :param member_name: The name of the member to have its releases modified.
        :type member_name: str
        :param Dxi: Indicates whether the member is released axially at its start. Defaults to False.
        :type Dxi: bool, optional
        :param Dyi: Indicates whether the member is released for shear in the local y-axis at its start. Defaults to False.
        :type Dyi: bool, optional
        :param Dzi: Indicates whether the member is released for shear in the local z-axis at its start. Defaults to False.
        :type Dzi: bool, optional
        :param Rxi: Indicates whether the member is released for torsion at its start. Defaults to False.
        :type Rxi: bool, optional
        :param Ryi: Indicates whether the member is released for moment about the local y-axis at its start. Defaults to False.
        :type Ryi: bool, optional
        :param Rzi: Indicates whether the member is released for moment about the local z-axis at its start. Defaults to False.
        :type Rzi: bool, optional
        :param Dxj: Indicates whether the member is released axially at its end. Defaults to False.
        :type Dxj: bool, optional
        :param Dyj: Indicates whether the member is released for shear in the local y-axis at its end. Defaults to False.
        :type Dyj: bool, optional
        :param Dzj: Indicates whether the member is released for shear in the local z-axis. Defaults to False.
        :type Dzj: bool, optional
        :param Rxj: Indicates whether the member is released for torsion at its end. Defaults to False.
        :type Rxj: bool, optional
        :param Ryj: Indicates whether the member is released for moment about the local y-axis at its end. Defaults to False.
        :type Ryj: bool, optional
        :param Rzj: Indicates whether the member is released for moment about the local z-axis at its end. Defaults to False.
        :type Rzj: bool, optional
        """
        
        # Apply the end releases to the member
        try:
            self.members[member_name].Releases = [Dxi, Dyi, Dzi, Rxi, Ryi, Rzi, Dxj, Dyj, Dzj, Rxj, Ryj, Rzj] 
        except KeyError:
            raise NameError(f"Member '{member_name}' does not exist in the model")

        # Flag the model as unsolved
        self.solution = None

    def add_load_combo(self, name:str, factors:dict, combo_tags:list | None = None):
        """Adds a load combination to the model.

        :param name: A unique name for the load combination (e.g. '1.2D+1.6L+0.5S' or 'Gravity Combo').
        :type name: str
        :param factors: A dictionary containing load cases and their corresponding factors (e.g. {'D':1.2, 'L':1.6, 'S':0.5}).
        :type factors: dict
        :param combo_tags: A list of tags used to categorize load combinations. Default is `None`. This can be useful for filtering results later on, or for limiting analysis to only those combinations with certain tags. This feature is provided for convenience. It is not necessary to use tags.
        :type combo_tags: list, optional
        """            

        # Create a new load combination object
        new_combo = LoadCombo(name, combo_tags, factors)

        # Add the load combination to the dictionary of load combinations
        self.load_combos[name] = new_combo

        # Flag the model as solved
        self.solution = None

    def add_node_load(self, node_name:str, direction:str, P:float, case:str = 'Case 1'):
        """Adds a nodal load to the model.

        :param node_name: The name of the node where the load is being applied.
        :type node_name: str
        :param direction: The global direction the load is being applied in. Forces are `'FX'`, `'FY'`, and `'FZ'`. Moments are `'MX'`, `'MY'`, and `'MZ'`.
        :type direction: str
        :param P: The numeric value (magnitude) of the load.
        :type P: float
        :param case: The name of the load case the load belongs to. Defaults to 'Case 1'.
        :type case: str, optional
        :raises ValueError: Occurs when an invalid load direction was specified.
        """
        
        # Validate the value of direction
        if direction not in ('FX', 'FY', 'FZ', 'MX', 'MY', 'MZ'):
            raise ValueError(f"direction must be 'FX', 'FY', 'FZ', 'MX', 'MY', or 'MZ'. {direction} was given.")
        
        # Add the node load to the model
        try:
            self.nodes[node_name].NodeLoads.append((direction, P, case))
        except KeyError:
            raise NameError(f"Node '{node_name}' does not exist in the model")

        # Flag the model as unsolved
        self.solution = None

    def add_member_pt_load(self, member_name:str, direction:str, P:float, x:float, case:str = 'Case 1'):
        """Adds a member point load to the model.

        :param member_name: The name of the member the load is being applied to.
        :type member_name: str
        :param direction: The direction in which the load is to be applied. Valid values are `'Fx'`,
                          `'Fy'`, `'Fz'`, `'Mx'`, `'My'`, `'Mz'`, `'FX'`, `'FY'`, `'FZ'`, `'MX'`, `'MY'`, or `'MZ'`.
                          Note that lower-case notation indicates use of the beam's local
                          coordinate system, while upper-case indicates use of the model's globl
                          coordinate system.
        :type direction: str
        :param P: The numeric value (magnitude) of the load.
        :type P: float
        :param x: The load's location along the member's local x-axis.
        :type x: float
        :param case: The load case to categorize the load under. Defaults to 'Case 1'.
        :type case: str, optional
        :raises ValueError: Occurs when an invalid load direction has been specified.
        """            

        # Validate the value of direction
        if direction not in ('Fx', 'Fy', 'Fz', 'FX', 'FY', 'FZ', 'Mx', 'My', 'Mz', 'MX', 'MY', 'MZ'):
            raise ValueError(f"direction must be 'Fx', 'Fy', 'Fz', 'FX', 'FY', FZ', 'Mx', 'My', 'Mz', 'MX', 'MY', or 'MZ'. {direction} was given.")
        
        # Add the point load to the member
        try:
            self.members[member_name].PtLoads.append((direction, P, x, case))
        except KeyError:
            raise NameError(f"Member '{member_name}' does not exist in the model")
                
        # Flag the model as unsolved
        self.solution = None

    def add_member_dist_load(self, member_name:str, direction:str, w1:float, w2:float,
                             x1:float | None = None, x2:float | None = None,
                             case:str = 'Case 1'):
        """Adds a member distributed load to the model.

        :param member_name: The name of the member the load is being appied to.
        :type member_name: str
        :param direction: The direction in which the load is to be applied. Valid values are `'Fx'`,
                          `'Fy'`, `'Fz'`, `'FX'`, `'FY'`, or `'FZ'`.
                          Note that lower-case notation indicates use of the beam's local
                          coordinate system, while upper-case indicates use of the model's globl
                          coordinate system.
        :type direction: str
        :param w1: The starting value (magnitude) of the load.
        :type w1: float
        :param w2: The ending value (magnitude) of the load.
        :type w2: float
        :param x1: The load's start location along the member's local x-axis. If this argument is
                   not specified, the start of the member will be used. Defaults to `None`
        :type x1: float, optional
        :param x2: The load's end location along the member's local x-axis. If this argument is not
                   specified, the end of the member will be used. Defaults to `None`.
        :type x2: float, optional
        :param case: _description_, defaults to 'Case 1'
        :type case: str, optional
        :raises ValueError: Occurs when an invalid load direction has been specified.
        """
       
        # Validate the value of direction
        if direction not in ('Fx', 'Fy', 'Fz', 'FX', 'FY', 'FZ'):
            raise ValueError(f"direction must be 'Fx', 'Fy', 'Fz', 'FX', 'FY', or 'FZ'. {direction} was given.")
        # Determine if a starting and ending points for the load have been specified.
        # If not, use the member start and end as defaults
        if x1 == None:
            start = 0
        else:
            start = x1
        
        if x2 == None:
            end = self.members[member_name].L()
        else:
            end = x2

        # Add the distributed load to the member
        try:
            self.members[member_name].DistLoads.append((direction, w1, w2, start, end, case))
        except KeyError:
            raise NameError(f"Member '{member_name}' does not exist in the model")
                
        # Flag the model as unsolved
        self.solution = None

    def add_member_self_weight(self, global_direction:str, factor:float, case:str = 'Case 1'):
        """Adds self weight to all members in the model. Note that this only works for members. Plate and Quad elements will be ignored by this command.

        :param global_direction: The global direction to apply the member load in: 'FX', 'FY', or 'FZ'.
        :type global_direction: string
        :param factor: A factor to apply to the member self-weight. Can be used to account for items like connections, or to switch the direction of the self-weight load.
        :type factor: float
        :param case: The load case to apply the self-weight to. Defaults to 'Case 1'
        :type case: str, optional
        :raises ValueError: IF a local direction ('Fx', 'Fy', or 'Fz') is used instead of a global direction.
        """

        # Validate that a global direction was provided, not a local direction
        if global_direction in ('Fx', 'Fy', 'Fz'):
            raise ValueError(
                f"Local direction '{global_direction}' is not allowed for self-weight.  \
                    Use global directions 'FX', 'FY', or 'FZ' instead."
            )

        # Validate the value of direction
        if global_direction not in ('FX', 'FY', 'FZ'):
            raise ValueError(f"Direction must be 'FX', 'FY', or 'FZ'. {global_direction} was given.")

        # Step through each member in the model
        for member in self.members.values():

            # Calculate the self weight of the member
            self_weight = factor*member.material.rho*member.section.A

            # Add the self-weight load to the member
            self.add_member_dist_load(member.name, global_direction, self_weight, self_weight, case=case)
        
        # No need to flag the model as unsolved. That has already been taken care of by our call to `add_member_dist_load`

    def add_plate_surface_pressure(self, plate_name:str, pressure:float, case:str = 'Case 1'):
        """Adds a surface pressure to the rectangular plate element.
        

        :param plate_name: The name for the rectangular plate to add the surface pressure to.
        :type plate_name: str
        :param pressure: The value (magnitude) for the surface pressure.
        :type pressure: float
        :param case: The load case to add the surface pressure to. Defaults to 'Case 1'.
        :type case: str, optional
        :raises Exception: Occurs when an invalid plate name has been specified.
        """   

        # Add the surface pressure to the rectangle
        try:
            self.plates[plate_name].pressures.append([pressure, case])
        except KeyError:
            raise NameError(f"Plate '{plate_name}' does not exist in the model")
        
        # Flag the model as unsolved
        self.solution = None

    def add_quad_surface_pressure(self, quad_name:str, pressure:float, case:str = 'Case 1'):
        """Adds a surface pressure to the quadrilateral element.

        :param quad_name: The name for the quad to add the surface pressure to.
        :type quad_name: str
        :param pressure: The value (magnitude) for the surface pressure.
        :type pressure: float
        :param case: The load case to add the surface pressure to. Defaults to 'Case 1'.
        :type case: str, optional
        :raises Exception: Occurs when an invalid quad name has been specified.
        """

        # Add the surface pressure to the quadrilateral
        try:
            self.quads[quad_name].pressures.append([pressure, case])
        except KeyError:
            raise NameError(f"Quad '{quad_name}' does not exist in the model")
        
        # Flag the model as unsolved
        self.solution = None

    def delete_loads(self):
        """Deletes all loads from the model along with any results based on the loads.
        """

        # Delete the member loads and the calculated internal forces
        for member in self.members.values():
            member.DistLoads = []
            member.PtLoads = []
            member.SegmentsZ = []
            member.SegmentsY = []
            member.SegmentsX = []
        
        # Delete the plate loads
        for plate in self.plates.values():
            plate.pressures = []
        
        # Delete the quadrilateral loads
        for quad in self.quads.values():
            quad.pressures = []
        
        # Delete the nodal loads, calculated displacements, and calculated reactions
        for node in self.nodes.values():

            node.NodeLoads = []

            node.DX = {}
            node.DY = {}
            node.DZ = {}
            node.RX = {}
            node.RY = {}
            node.RZ = {}

            node.RxnFX = {}
            node.RxnFY = {}
            node.RxnFZ = {}
            node.RxnMX = {}
            node.RxnMY = {}
            node.RxnMZ = {}

        # Flag the model as unsolved
        self.solution = None

    def K(self, combo_name='Combo 1', log=False, check_stability=True, sparse=True):
        """Returns the model's global stiffness matrix. The stiffness matrix will be returned in
           scipy's sparse lil format, which reduces memory usage and can be easily converted to
           other formats.

        :param combo_name: The load combination to get the stiffness matrix for. Defaults to 'Combo 1'.
        :type combo_name: str, optional
        :param log: Prints updates to the console if set to True. Defaults to False.
        :type log: bool, optional
        :param check_stability: Causes Pynite to check for instabilities if set to True. Defaults
                                to True. Set to False if you want the model to run faster.
        :type check_stability: bool, optional
        :param sparse: Returns a sparse matrix if set to True, and a dense matrix otherwise.
                       Defaults to True.
        :type sparse: bool, optional
        :return: The global stiffness matrix for the structure.
        :rtype: ndarray or coo_matrix
        """

        # Determine if a sparse matrix has been requested
        if sparse == True:
            # Instead of pushing one entry at a time, we keep batched row/col/data arrays
            # per element and concatenate once. This drastically cuts Python overhead.
            row_parts: list = []
            col_parts: list = []
            data_parts: list = []
        else:
            # Initialize a dense matrix of zeros
            K = np.zeros((len(self.nodes) * 6, len(self.nodes) * 6))

        # Add stiffness terms for each nodal spring in the model
        if log: print('- Adding nodal spring support stiffness terms to global stiffness matrix')
        for node in self.nodes.values():

            # Determine if the node has any spring supports
            if node.spring_DX[0] is not None:
                # Check for an active spring support
                if node.spring_DX[2] == True:
                    m = node.ID * 6
                    val = float(node.spring_DX[0])
                    if sparse == True:
                        row_parts.append(np.array([m], dtype=np.int64))
                        col_parts.append(np.array([m], dtype=np.int64))
                        data_parts.append(np.array([val], dtype=float))
                    else:
                        K[m, m] += val

            if node.spring_DY[0] is not None:
                # Check for an active spring support
                if node.spring_DY[2] == True:
                    m = node.ID * 6 + 1
                    val = float(node.spring_DY[0])
                    if sparse == True:
                        row_parts.append(np.array([m], dtype=np.int64))
                        col_parts.append(np.array([m], dtype=np.int64))
                        data_parts.append(np.array([val], dtype=float))
                    else:
                        K[m, m] += val

            if node.spring_DZ[0] is not None:
                # Check for an active spring support
                if node.spring_DZ[2] == True:
                    m = node.ID * 6 + 2
                    val = float(node.spring_DZ[0])
                    if sparse == True:
                        row_parts.append(np.array([m], dtype=np.int64))
                        col_parts.append(np.array([m], dtype=np.int64))
                        data_parts.append(np.array([val], dtype=float))
                    else:
                        K[m, m] += val

            if node.spring_RX[0] is not None:
                # Check for an active spring support
                if node.spring_RX[2] == True:
                    m = node.ID * 6 + 3
                    val = float(node.spring_RX[0])
                    if sparse == True:
                        row_parts.append(np.array([m], dtype=np.int64))
                        col_parts.append(np.array([m], dtype=np.int64))
                        data_parts.append(np.array([val], dtype=float))
                    else:
                        K[m, m] += val

            if node.spring_RY[0] is not None:
                # Check for an active spring support
                if node.spring_RY[2] == True:
                    m = node.ID * 6 + 4
                    val = float(node.spring_RY[0])
                    if sparse == True:
                        row_parts.append(np.array([m], dtype=np.int64))
                        col_parts.append(np.array([m], dtype=np.int64))
                        data_parts.append(np.array([val], dtype=float))
                    else:
                        K[m, m] += val

            if node.spring_RZ[0] is not None:
                # Check for an active spring support
                if node.spring_RZ[2] == True:
                    m = node.ID * 6 + 5
                    val = float(node.spring_RZ[0])
                    if sparse == True:
                        row_parts.append(np.array([m], dtype=np.int64))
                        col_parts.append(np.array([m], dtype=np.int64))
                        data_parts.append(np.array([val], dtype=float))
                    else:
                        K[m, m] += val

        # Add stiffness terms for each spring in the model
        if log: print('- Adding spring stiffness terms to global stiffness matrix')
        for spring in self.springs.values():

            if spring.active[combo_name] == True:

                # Build the DOF index vector once and add the whole 12x12 block in one shot
                dofs = self._build_dof_vector(spring.i_node, spring.j_node)
                spring_K = spring.K()

                if sparse == True:
                    self._append_sparse_block(dofs, spring_K, row_parts, col_parts, data_parts)
                else:
                    self._add_dense_block(K, dofs, spring_K)

        # Add stiffness terms for each physical member in the model
        if log: print('- Adding member stiffness terms to global stiffness matrix')
        for phys_member in self.members.values():

            # Check to see if the physical member is active for the given load combination
            if phys_member.active[combo_name] == True:

                # Step through each sub-member in the physical member and add terms
                for member in phys_member.sub_members.values():

                    # Build the member DOF vector once so we can add the entire 12x12 block
                    dofs = self._build_dof_vector(member.i_node, member.j_node)
                    member_K = member.K()

                    if sparse == True:
                        self._append_sparse_block(dofs, member_K, row_parts, col_parts, data_parts)
                    else:
                        self._add_dense_block(K, dofs, member_K)

        # Add stiffness terms for each quadrilateral in the model
        if log: print('- Adding quadrilateral stiffness terms to global stiffness matrix')
        for quad in self.quads.values():

            # Get the quadrilateral's global stiffness matrix
            quad_K = quad.K()
            # Four nodes -> 24 DOFs. The helper keeps those indices contiguous
            dofs = self._build_dof_vector(quad.i_node, quad.j_node, quad.m_node, quad.n_node)

            if sparse == True:
                self._append_sparse_block(dofs, quad_K, row_parts, col_parts, data_parts)
            else:
                self._add_dense_block(K, dofs, quad_K)
        
        # Add stiffness terms for each plate in the model
        if log: print('- Adding plate stiffness terms to global stiffness matrix')
        for plate in self.plates.values():

            # Get the plate's global stiffness matrix
            plate_K = plate.K()
            # Same concept as the quad above, but for the rectangular plate element
            dofs = self._build_dof_vector(plate.i_node, plate.j_node, plate.m_node, plate.n_node)

            if sparse == True:
                self._append_sparse_block(dofs, plate_K, row_parts, col_parts, data_parts)
            else:
                self._add_dense_block(K, dofs, plate_K)

        if sparse:
            # Concatenate the per-element contributions into the vectors scipy expects
            if row_parts:
                row = np.concatenate(row_parts)
                col = np.concatenate(col_parts)
                data = np.concatenate(data_parts)
            else:
                # Provide empty vectors when no elements contributed (edge case)
                row = np.array([], dtype=np.int64)
                col = np.array([], dtype=np.int64)
                data = np.array([], dtype=float)

            # Build the sparse COO matrix from the assembled vectors
            K = sp.sparse.coo_matrix((data, (row, col)), shape=(len(self.nodes) * 6, len(self.nodes) * 6))

        # Check that there are no nodal instabilities
        if check_stability:
            if log: print('- Checking nodal stability')
            if sparse: Analysis._check_stability(self, K.tocsr())
            else: Analysis._check_stability(self, K)

        # Return the global stiffness matrix
        return K

    def Kg(self, combo_name='Combo 1', log=False, sparse=True, first_step=True):
        """Returns the model's global geometric stiffness matrix. Geometric stiffness of plates is not considered.

        :param combo_name: The name of the load combination to derive the matrix for. Defaults to 'Combo 1'.
        :type combo_name: str, optional
        :param log: Prints updates to the console if set to `True`. Defaults to `False`.
        :type log: bool, optional
        :param sparse: Returns a sparse matrix if set to `True`, and a dense matrix otherwise. Defaults to `True`.
        :type sparse: bool, optional
        :param first_step: Used to indicate if the analysis is occuring at the first load step. Used in nonlinear analysis where the load is broken into multiple steps. Default is `True`.
        :type first_step: book, optional
        :return: The global geometric stiffness matrix for the structure.
        :rtype: ndarray or coo_matrix
        """

        if sparse == True:
            # Initialize a zero matrix to hold all the stiffness terms. The matrix will be stored as a scipy sparse `lil_matrix`. This matrix format has several advantages. It uses less memory if the matrix is sparse, supports slicing, and can be converted to other formats (sparse or dense) later on for mathematical operations.
            from scipy.sparse import lil_matrix
            Kg = lil_matrix((len(self.nodes)*6, len(self.nodes)*6))
        else:
            Kg = zeros(len(self.nodes)*6, len(self.nodes)*6)

        # Add stiffness terms for each physical member in the model
        if log:
            print('- Adding member geometric stiffness terms to global geometric stiffness matrix')
        for phys_member in self.members.values():

            # Check to see if the physical member is active for the given load combination
            if phys_member.active[combo_name] == True:

                # Step through each sub-member in the physical member and add terms
                for member in phys_member.sub_members.values():

                    # Calculate the axial force in the member
                    E = member.material.E
                    A = member.section.A
                    L = member.L()

                    # Calculate the axial force acting on the member
                    if first_step:
                        # For the first load step take P = 0
                        P = 0
                    else:
                        # Calculate the member axial force due to axial strain
                        d = member.d(combo_name)
                        P = E*A/L*(d[6, 0] - d[0, 0])

                    # Get the member's global stiffness matrix
                    # Storing it as a local variable eliminates the need to rebuild it every time a term is needed
                    member_Kg = member.Kg(P)

                    # Step through each term in the member's stiffness matrix
                    # 'a' & 'b' below are row/column indices in the member's stiffness matrix
                    # 'm' & 'n' are corresponding row/column indices in the global stiffness matrix
                    for a in range(12):

                        # Determine if index 'a' is related to the i-node or j-node
                        if a < 6:
                            # Find the corresponding index 'm' in the global stiffness matrix
                            m = member.i_node.ID*6 + a
                        else:
                            # Find the corresponding index 'm' in the global stiffness matrix
                            m = member.j_node.ID*6 + (a-6)

                        for b in range(12):

                            # Determine if index 'b' is related to the i-node or j-node
                            if b < 6:
                                # Find the corresponding index 'n' in the global stiffness matrix
                                n = member.i_node.ID*6 + b
                            else:
                                # Find the corresponding index 'n' in the global stiffness matrix
                                n = member.j_node.ID*6 + (b-6)

                            # Now that 'm' and 'n' are known, place the term in the global stiffness matrix
                            Kg[m, n] += member_Kg[(a, b)]

        # Return the global geometric stiffness matrix
        return Kg

    def Km(self, combo_name='Combo 1', push_combo='Push', step_num=1, log=False, sparse=True):
        """Calculates the structure's global plastic reduction matrix, which is used for nonlinear inelastic analysis.

        :param combo_name: The name of the load combination to get the plastic reduction matrix for. Defaults to 'Combo 1'.
        :type combo_name: str, optional
        :param push_combo: The name of the load combination that contains the pushover load definition. Defaults to 'Push'.
        :type push_combo: str, optional
        :param step_num: The load step used to generate the plastic reduction matrix. Defaults to 1.
        :type step_num: int, optional
        :param log: Determines whether this method writes output to the console as it runs. Defaults to False.
        :type log: bool, optional
        :param sparse: Indicates whether the sparse solver should be used. Defaults to True.
        :type sparse: bool, optional
        :return: The gloabl plastic reduction matrix.
        :rtype: array
        """

        # Determine if a sparse matrix has been requested
        if sparse == True:
            # The plastic reduction matrix will be stored as a scipy `coo_matrix`. Scipy's documentation states that this type of matrix is ideal for efficient construction of finite element matrices. When converted to another format, the `coo_matrix` sums values at the same (i, j) index. We'll build the matrix from three lists.
            row = []
            col = []
            data = []
        else:
            # Initialize a dense matrix of zeros
            Km = zeros((len(self.nodes)*6, len(self.nodes)*6))

        # Add stiffness terms for each physical member in the model
        for phys_member in self.members.values():

            # Check to see if the physical member is active for the given load combination
            if phys_member.active[combo_name] == True:

                # Step through each sub-member in the physical member and add terms
                for member in phys_member.sub_members.values():

                    # Get the member's global plastic reduction matrix
                    # Storing it as a local variable eliminates the need to rebuild it every time a term is needed
                    member_Km = member.Km(combo_name)

                    # Step through each term in the member's plastic reduction matrix
                    # 'a' & 'b' below are row/column indices in the member's matrix
                    # 'm' & 'n' are corresponding row/column indices in the structure's global matrix
                    for a in range(12):

                        # Determine if index 'a' is related to the i-node or j-node
                        if a < 6:
                            # Find the corresponding index 'm' in the global plastic reduction matrix
                            m = member.i_node.ID*6 + a
                        else:
                            # Find the corresponding index 'm' in the global plastic reduction matrix
                            m = member.j_node.ID*6 + (a-6)

                        for b in range(12):

                            # Determine if index 'b' is related to the i-node or j-node
                            if b < 6:
                                # Find the corresponding index 'n' in the global plastic reduction matrix
                                n = member.i_node.ID*6 + b
                            else:
                                # Find the corresponding index 'n' in the global plastic reduction matrix
                                n = member.j_node.ID*6 + (b-6)

                            # Now that 'm' and 'n' are known, place the term in the global plastic reduction matrix
                            if sparse == True:
                                row.append(m)
                                col.append(n)
                                data.append(member_Km[a, b])
                            else:
                                Km[m, n] += member_Km[a, b]

        if sparse:
            # The plastic reduction matrix will be stored as a scipy `coo_matrix`. Scipy's documentation states that this type of matrix is ideal for efficient construction of finite element matrices. When converted to another format, the `coo_matrix` sums values at the same (i, j) index.
            from scipy.sparse import coo_matrix
            row = array(row)
            col = array(col)
            data = array(data)
            Km = coo_matrix((data, (row, col)), shape=(len(self.nodes)*6, len(self.nodes)*6))

        # Check that there are no nodal instabilities
        # if check_stability:
        #     if log: print('- Checking nodal stability')
        #     if sparse: Analysis._check_stability(self, Km.tocsr())
        #     else: Analysis._check_stability(self, Km)

        # Return the global plastic reduction matrix
        return Km

    def FER(self, combo_name='Combo 1') -> NDArray[float64]:
        """Assembles and returns the global fixed end reaction vector for any given load combo.

        :param combo_name: The name of the load combination to get the fixed end reaction vector
                           for. Defaults to 'Combo 1'.
        :type combo_name: str, optional
        :return: The fixed end reaction vector
        :rtype: NDArray[float64]
        """

        # Initialize a zero vector to hold all the terms
        FER = np.zeros((len(self.nodes) * 6, 1))

        # Step through each physical member in the model
        for phys_member in self.members.values():

            # Step through each sub-member and add terms
            for member in phys_member.sub_members.values():

                # Grab the member's fixed-end reactions and add the entire 12x1 block
                # directly at the matching DOF locations
                member_FER = np.asarray(member.FER(combo_name), dtype=float).reshape(-1)
                dofs = self._build_dof_vector(member.i_node, member.j_node)
                FER[dofs, 0] += member_FER

        # Add terms for each rectangular plate in the model
        for plate in self.plates.values():

            # Add the 24x1 plate reactions with the same DOF helper
            plate_FER = np.asarray(plate.FER(combo_name), dtype=float).reshape(-1)
            dofs = self._build_dof_vector(plate.i_node, plate.j_node, plate.m_node, plate.n_node)
            FER[dofs, 0] += plate_FER

        # Add terms for each quadrilateral in the model
        for quad in self.quads.values():

            # Add the 24x1 quad reactions via the DOF helper
            quad_FER = np.asarray(quad.FER(combo_name), dtype=float).reshape(-1)
            dofs = self._build_dof_vector(quad.i_node, quad.j_node, quad.m_node, quad.n_node)
            FER[dofs, 0] += quad_FER

        # Return the global fixed end reaction vector
        return FER
    
    def P(self, combo_name='Combo 1') -> NDArray[float64]:
        """Assembles and returns the global nodal force vector.

        :param combo_name: The name of the load combination to get the force vector for. Defaults
                           to 'Combo 1'.
        :type combo_name: str, optional
        :return: The global nodal force vector.
        :rtype: NDArray[float64]
        """

        # Initialize a zero vector to hold all the terms
        P = np.zeros((len(self.nodes) * 6, 1))

        # Get the load combination for the given 'combo_name'
        combo = self.load_combos[combo_name]

        # Map load direction strings to their DOF offsets once
        dof_lookup = {'FX': 0, 'FY': 1, 'FZ': 2, 'MX': 3, 'MY': 4, 'MZ': 5}

        # Add terms for each node in the model
        for node in self.nodes.values():

            # Accumulate this node's six DOF loads locally before writing to the global vector
            local = np.zeros(6, dtype=float)

            for load in node.NodeLoads:
                direction, magnitude, case = load[0], load[1], load[2]

                # Look up the combo factor once per load
                factor = combo.factors.get(case)
                if factor is None:
                    continue

                # Normalize the direction string and map it to the correct DOF slot
                idx = dof_lookup.get(direction.upper() if isinstance(direction, str) else direction)
                if idx is None:
                    continue  # Ignore load types outside the standard 6 DOFs

                # Add the scaled load into the local 6-entry accumulator
                local[idx] += factor * magnitude

            # Once all loads for this node are tallied, drop the 6x1 block into the global vector
            if np.any(local):
                dofs = self._build_dof_vector(node)
                P[dofs, 0] += local

        # Return the global nodal force vector
        return P

    def D(self, combo_name='Combo 1') -> NDArray[float64]:
        """Returns the global displacement vector for the model.

        :param combo_name: The name of the load combination to get the results for. Defaults to
                           'Combo 1'.
        :type combo_name: str, optional
        :return: The global displacement vector for the model
        :rtype: NDArray[float64]
        """

        # Return the global displacement vector
        return self._D[combo_name]

    # def _deprecated_analyze_old(self, log=False, check_stability=True, check_statics=False, max_iter=30, sparse=True, combo_tags=None, spring_tolerance=0, member_tolerance=0):
    #     """Performs first-order static analysis. Iterations are performed if tension-only members or compression-only members are present.

    #     :param log: Prints the analysis log to the console if set to True. Default is False.
    #     :type log: bool, optional
    #     :param check_stability: When set to `True`, checks for nodal instabilities. This slows down analysis a little. Default is `True`.
    #     :type check_stability: bool, optional
    #     :param check_statics: When set to `True`, causes a statics check to be performed
    #     :type check_statics: bool, optional
    #     :param max_iter: The maximum number of iterations to try to get convergence for tension/compression-only analysis. Defaults to 30.
    #     :type max_iter: int, optional
    #     :param sparse: Indicates whether the sparse matrix solver should be used. A matrix can be considered sparse or dense depening on how many zero terms there are. Structural stiffness matrices often contain many zero terms. The sparse solver can offer faster solutions for such matrices. Using the sparse solver on dense matrices may lead to slower solution times.
    #     :type sparse: bool, optional
    #     :raises Exception: _description_
    #     :raises Exception: _description_
    #     """

    #     if log:
    #         print('+-----------+')
    #         print('| Analyzing |')
    #         print('+-----------+')

    #     # Import `scipy` features if the sparse solver is being used
    #     if sparse == True:
    #         from scipy.sparse.linalg import spsolve

    #     # Prepare the model for analysis
    #     Analysis._prepare_model(self)

    #     # Get the auxiliary list used to determine how the matrices will be partitioned
    #     D1_indices, D2_indices, D2 = Analysis._partition_D(self)

    #     # Identify which load combinations have the tags the user has given
    #     combo_list = Analysis._identify_combos(self, combo_tags)

    #     # Step through each load combination
    #     for combo in combo_list:

    #         if log:
    #             print('')
    #             print('- Analyzing load combination ' + combo.name)

    #         # Keep track of the number of iterations
    #         iter_count = 1
    #         convergence = False
    #         divergence = False

    #         # Iterate until convergence or divergence occurs
    #         while convergence == False and divergence == False:

    #             # Check for tension/compression-only divergence
    #             if iter_count > max_iter:
    #                 divergence = True
    #                 raise Exception('Model diverged during tension/compression-only analysis')

    #             # Get the partitioned global stiffness matrix K11, K12, K21, K22
    #             if sparse == True:
    #                 K11, K12, K21, K22 = Analysis._partition(self, self.K(combo.name, log, check_stability, sparse).tolil(), D1_indices, D2_indices)
    #             else:
    #                 K11, K12, K21, K22 = Analysis._partition(self, self.K(combo.name, log, check_stability, sparse), D1_indices, D2_indices)

    #             # Get the partitioned global fixed end reaction vector
    #             FER1, FER2 = Analysis._partition(self, self.FER(combo.name), D1_indices, D2_indices)

    #             # Get the partitioned global nodal force vector
    #             P1, P2 = Analysis._partition(self, self.P(combo.name), D1_indices, D2_indices)

    #             # Calculate the global displacement vector
    #             if log:
    #                 print('- Calculating global displacement vector')
    #             if K11.shape == (0, 0):
    #                 # All displacements are known, so D1 is an empty vector
    #                 D1 = []
    #             else:
    #                 try:
    #                     # Calculate the unknown displacements D1
    #                     if sparse == True:
    #                         # The partitioned stiffness matrix is in `lil` format, which is great for memory, but slow for mathematical operations. The stiffness matrix will be converted to `csr` format for mathematical operations. The `@` operator performs matrix multiplication on sparse matrices.
    #                         D1 = spsolve(K11.tocsr(), subtract(subtract(P1, FER1), K12.tocsr() @ D2))
    #                         D1 = D1.reshape(len(D1), 1)
    #                     else:
    #                         D1 = solve(K11, subtract(subtract(P1, FER1), matmul(K12, D2)))
    #                 except:
    #                     # Return out of the method if 'K' is singular and provide an error message
    #                     raise Exception('The stiffness matrix is singular, which implies rigid body motion. The structure is unstable. Aborting analysis.')

    #             # Store the calculated displacements to the model and the nodes in the model
    #             Analysis._store_displacements(self, D1, D2, D1_indices, D2_indices, combo)

    #             # Check for tension/compression-only convergence
    #             convergence = Analysis._check_TC_convergence(self, combo.name, log=log, spring_tolerance=spring_tolerance, member_tolerance=member_tolerance)

    #             if convergence == False:

    #                 if log:
    #                     print('- Tension/compression-only analysis did not converge. Adjusting stiffness matrix and reanalyzing.')
    #             else:
    #                 if log:
    #                     print('- Tension/compression-only analysis converged after ' + str(iter_count) + ' iteration(s)')

    #             # Keep track of the number of tension/compression only iterations
    #             iter_count += 1

    #     # Calculate reactions
    #     Analysis._calc_reactions(self, log, combo_tags)

    #     if log:
    #         print('')     
    #         print('- Analysis complete')
    #         print('')

    #     # Check statics if requested
    #     if check_statics == True:
    #         Analysis._check_statics(self, combo_tags)

    #     # Flag the model as solved
    #     self.solution = 'Linear TC'

    def _handle_solve_error(self, error: Exception, combo_name: str) -> None:
        """Handle errors during load combination solving with diagnostics.

        :param error: The exception that was raised
        :param combo_name: Name of the load combination that failed
        :raises Analysis.AnalysisError: Always raises after diagnostics
        """
        from Pynite.Diagnostics import ModelDiagnostics

        print('')
        print('=' * 60)
        print(f'ANALYSIS FAILED - Error in load combination {combo_name}')
        print('=' * 60)
        print('')
        print(f'Error: {str(error)}')
        print('')

        # Run diagnostics for singular matrix errors
        if 'singular' in str(error).lower():
            print('Running diagnostics to identify root cause...')
            print('')
            diagnostics = ModelDiagnostics(self)
            report = diagnostics.run_full_diagnosis()
            diagnostic_text = report.format(verbose=True)
            print(diagnostic_text)

            raise Analysis.AnalysisError(
                'The stiffness matrix is singular (structure is unstable)',
                diagnostic_text
            ) from error
        else:
            raise error

    @staticmethod
    def _solve_combo_linear_worker(
        model: 'FEModel3D',
        combo: LoadCombo,
        K11,  # scipy sparse matrix or ndarray
        K11_factored,  # SuperLU object or LU factorization tuple
        K12,  # scipy sparse matrix or ndarray
        K12_csr,  # scipy csr_matrix or None
        D2,  # ndarray
        D1_indices: list[int],
        D2_indices: list[int],
        sparse: bool
    ):
        """Worker function to solve a single load combination.

        This function can be called directly (sequential) or via ThreadPoolExecutor
        (parallel on free-threaded Python 3.14t).

        :param model: The finite element model
        :param combo: The load combination to solve
        :param K11: The partitioned stiffness matrix
        :param K11_factored: The factored K11 matrix (SuperLU for sparse, LU tuple for dense)
        :param K12: The K12 partition matrix
        :param K12_csr: The K12 partition in CSR format (for sparse solve)
        :param D2: Known displacements vector
        :param D1_indices: Indices for unknown displacements
        :param D2_indices: Indices for known displacements
        :param sparse: Whether to use sparse solver
        :return: Tuple of (combo, displacement vector)
        """
        # Get the partitioned global fixed end reaction vector
        FER1, FER2 = Analysis._partition(model, model.FER(combo.name), D1_indices, D2_indices)

        # Get the partitioned global nodal force vector
        P1, P2 = Analysis._partition(model, model.P(combo.name), D1_indices, D2_indices)

        # Calculate the global displacement vector
        if K11.shape == (0, 0):
            # All displacements are known, so disp1 is an empty vector
            disp1 = []
        else:
            # Calculate the unknown displacements disp1
            if sparse:
                # Use the factored sparse matrix (SuperLU object)
                # The solve method of SuperLU performs back-substitution
                disp1 = K11_factored.solve(subtract(subtract(P1, FER1), K12_csr @ D2))
                disp1 = disp1.reshape(len(disp1), 1)
            else:
                # Use the factored dense matrix (LU factorization tuple)
                from scipy.linalg import lu_solve
                disp1 = lu_solve(K11_factored, subtract(subtract(P1, FER1), matmul(K12, D2)))

            # Check for NaN or Inf values which indicate a singular matrix
            if np.any(np.isnan(disp1)) or np.any(np.isinf(disp1)):
                raise ValueError(f"Solution for combo '{combo.name}' contains NaN or Inf values - matrix is singular")

        return (combo, disp1)

    def analyze_linear(self, log: bool = False, check_stability: bool = True, check_statics: bool = False, sparse: bool = True,
                      combo_tags = None, parallel: bool = True, max_workers: int | None = None):
        """Performs first-order static analysis. This analysis procedure is much faster since it only assembles the global stiffness matrix once, rather than once for each load combination. It is not appropriate when non-linear behavior such as tension/compression only analysis or P-Delta analysis are required.

        :param log: Prints the analysis log to the console if set to True. Default is False.
        :type log: bool, optional
        :param check_stability: When set to True, checks the stiffness matrix for any unstable degrees of freedom and reports them back to the console. This does add to the solution time. Defaults to True.
        :type check_stability: bool, optional
        :param check_statics: When set to True, causes a statics check to be performed. Defaults to False.
        :type check_statics: bool, optional
        :param sparse: Indicates whether the sparse matrix solver should be used. A matrix can be considered sparse or dense depening on how many zero terms there are. Structural stiffness matrices often contain many zero terms. The sparse solver can offer faster solutions for such matrices. Using the sparse solver on dense matrices may lead to slower solution times. Be sure ``scipy`` is installed to use the sparse solver. Default is True.
        :type sparse: bool, optional
        :param combo_tags: Optional list of load combination tags to filter which combinations are analyzed. If None, all combinations are analyzed.
        :type combo_tags: list, optional
        :param parallel: Whether to use parallel processing when running on free-threaded Python (Python 3.14t). On standard Python with GIL, this parameter is ignored and sequential processing is used. Default is True.
        :type parallel: bool, optional
        :param max_workers: Maximum number of worker threads to use for parallel processing. If None, uses the number of CPU cores. Only used when parallel=True and running on free-threaded Python. Default is None.
        :type max_workers: int, optional
        :raises Exception: Occurs when a singular stiffness matrix is found. This indicates an unstable structure has been modeled.
        """

        if log:
            print('+-------------------+')
            print('| Analyzing: Linear |')
            print('+-------------------+')
        
        # Import `scipy` features if the sparse solver is being used
        if sparse == True:
            from scipy.sparse.linalg import splu

        # Prepare the model for analysis
        Analysis._prepare_model(self)

        # Get the auxiliary list used to determine how the matrices will be partitioned
        D1_indices, D2_indices, D2 = Analysis._partition_D(self)

        # Get the partitioned global stiffness matrix K11, K12, K21, K22
        # Note that for linear analysis the stiffness matrix can be obtained for any load combination, as it's the same for all of them
        combo_name = list(self.load_combos.keys())[0]
        if sparse == True:
            K11, K12, K21, K22 = Analysis._partition(self, self.K(combo_name, log, check_stability, sparse).tolil(), D1_indices, D2_indices)
        else:
            K11, K12, K21, K22 = Analysis._partition(self, self.K(combo_name, log, check_stability, sparse), D1_indices, D2_indices)

        # Identify which load combinations have the tags the user has given
        combo_list = Analysis._identify_combos(self, combo_tags)

        # Factorize K11 once for all load combinations (major optimization)
        # LU factorization is O(n³), back-substitution is O(n²)
        # By factorizing once, we avoid repeating the expensive O(n³) operation
        K11_factored = None
        K12_csr = None
        if K11.shape != (0, 0):
            try:
                if sparse == True:
                    # Sparse LU factorization - factorize once, solve many times
                    K11_factored = splu(K11.tocsc())
                    K12_csr = K12.tocsr()
                else:
                    # Dense LU factorization
                    from scipy.linalg import lu_factor
                    K11_factored = lu_factor(K11)
            except Exception as e:
                # Diagnose the root cause of the singular matrix
                from Pynite.Diagnostics import ModelDiagnostics
                print('')
                print('=' * 60)
                print('ANALYSIS FAILED - Singular Stiffness Matrix')
                print('=' * 60)
                print('')
                print('The stiffness matrix could not be factored, which means the')
                print('structure has one or more rigid body modes (it can move freely).')
                print('')
                print('Running diagnostics to identify root cause...')
                print('')

                diagnostics = ModelDiagnostics(self)
                report = diagnostics.run_full_diagnosis()
                diagnostic_text = report.format(verbose=True)
                print(diagnostic_text)

                raise Analysis.AnalysisError(
                    'The stiffness matrix is singular (structure is unstable)',
                    diagnostic_text
                ) from e

        # Determine if we should use parallel processing
        # Only use threads if:
        # 1. parallel=True (user enabled it)
        # 2. Running on free-threaded Python (no GIL)
        # 3. Have enough combos to justify overhead (>= 4)
        use_parallel = parallel and is_free_threaded() and len(combo_list) >= 4

        if use_parallel:
            # Parallel execution using ThreadPoolExecutor (free-threaded Python only)
            from concurrent.futures import ThreadPoolExecutor, as_completed

            num_workers = get_optimal_worker_count(len(combo_list), max_workers)

            if log:
                print(f'- Using parallel processing with {num_workers} workers (free-threaded Python detected)')
                print('')

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_combo = {
                    executor.submit(
                        FEModel3D._solve_combo_linear_worker,
                        self, combo, K11, K11_factored, K12, K12_csr, D2,
                        D1_indices, D2_indices, sparse
                    ): combo
                    for combo in combo_list
                }

                for future in as_completed(future_to_combo):
                    combo = future_to_combo[future]
                    try:
                        result_combo, D1 = future.result()
                        if log:
                            print(f'- Completed load combination {result_combo.name}')
                        Analysis._store_displacements(self, D1, D2, D1_indices, D2_indices, result_combo)
                    except Exception as e:
                        self._handle_solve_error(e, combo.name)
        else:
            # Sequential execution - call worker directly without thread pool
            if log:
                if not parallel:
                    print('- Using sequential processing (parallel=False)')
                elif not is_free_threaded():
                    print('- Using sequential processing (free-threaded Python not detected)')
                else:
                    print('- Using sequential processing (too few load combinations)')
                print('')

            for combo in combo_list:
                if log:
                    print(f'- Analyzing load combination {combo.name}')

                try:
                    # Call the same worker function used by parallel execution
                    result_combo, D1 = FEModel3D._solve_combo_linear_worker(
                        self, combo, K11, K11_factored, K12, K12_csr, D2,
                        D1_indices, D2_indices, sparse
                    )
                    Analysis._store_displacements(self, D1, D2, D1_indices, D2_indices, result_combo)
                except Exception as e:
                    self._handle_solve_error(e, combo.name)

        # Calculate reactions
        Analysis._calc_reactions(self, log, combo_tags)

        if log:
            print('')
            print('- Analysis complete')
            print('')

        # Check statics if requested
        if check_statics == True:
            Analysis._check_statics(self, combo_tags)

        # Flag the model as solved
        self.solution = 'Linear'

    def analyze(self, log=False, check_stability=True, check_statics=False, max_iter=30, sparse=True, combo_tags=None, spring_tolerance=0, member_tolerance=0, num_steps=1):
        """
        Performs a first-order elastic analysis of the model.

        Allows the use of sparse solvers for improved performance on large models. Handles tension/compression-only behavior for nodal springs and elements. Loads can be applied in steps for better convergence of complex tension/compression-only models.

        Parameters
        ----------
        log : bool, optional
            If True, prints progress messages during analysis (default: False).
        check_stability : bool, optional
            If True, checks model stability at each analysis step (default: True).
        check_statics : bool, optional
            If True, performs a statics check after analysis (default: False).
        max_iter : int, optional
            Maximum number of tension/compression-only iterations allowed per load step before assuming divergence (default: 30).
        sparse : bool, optional
            If True, uses sparse matrix solvers for improved efficiency on large models (default: True).
        combo_tags : list[str] or None, optional
            List of tags used to select which load combinations to analyze. If None, all combinations are analyzed (default: None).
        spring_tolerance : float, optional
            Tolerance used to determine convergence for springs in tension/compression-only analysis (default: 0).
        member_tolerance : float, optional
            Tolerance used to determine convergence for members in tension/compression-only analysis (default: 0).
        num_steps : int, optional
            Number of load increments for applying load combinations. Use more steps for better convergence in highly nonlinear cases (default: 1).

        Raises
        ------
        Exception
            If the stiffness matrix is singular (indicating instability), or if the model fails to converge within the maximum allowed iterations.

        Notes
        -----
        - Flags the model as solved upon successful completion.
        - Stores calculated displacements and reactions in the model.
        - If statics checking is enabled, runs a global equilibrium check on the results.
        """

        if log:
            print('+-----------+')
            print('| Analyzing |')
            print('+-----------+')

        # Import `scipy` features if the sparse solver is being used
        if sparse == True:
            from scipy.sparse.linalg import spsolve

        # Prepare the model for analysis
        Analysis._prepare_model(self)

        # Identify which load combinations have the tags the user has given
        combo_list = Analysis._identify_combos(self, combo_tags)

        # Get the auxiliary list used to determine how the matrices will be partitioned
        D1_indices, D2_indices, D2 = Analysis._partition_D(self)

        # Calculate the incremental enforced displacement vector
        Delta_D2 = D2/num_steps

        # Step through each load combination
        for combo in combo_list:

            if log:
                print('')
                print('- Analyzing load combination ' + combo.name)

            # Get the partitioned total global fixed end reaction vector
            FER1, FER2 = Analysis._partition(self, self.FER(combo.name), D1_indices, D2_indices)

            # Calculate the incremental global fixed end reaction vector
            Delta_FER1 = FER1/num_steps

            # Get the partitioned total global nodal force vector
            P1, P2 = Analysis._partition(self, self.P(combo.name), D1_indices, D2_indices)

            # Calculate the incremental global nodal force vector
            Delta_P1 = P1/num_steps

            # Apply the load incrementally
            load_step = 1
            while load_step <= num_steps:

                # Keep track of the number of iterations in this load step
                iter_count = 1
                convergence = False
                divergence = False

                # Iterate until convergence or divergence occurs
                while convergence == False and divergence == False:

                    # Check for tension/compression-only divergence
                    if iter_count > max_iter:
                        divergence = True
                        from Pynite.Diagnostics import ModelDiagnostics
                        print('')
                        print('=' * 60)
                        print('ANALYSIS FAILED - Tension/Compression-Only Divergence')
                        print('=' * 60)
                        print('')
                        print(f'The model failed to converge after {max_iter} iterations.')
                        print('')
                        print('This typically happens when:')
                        print('  1. Too many tension-only or compression-only elements')
                        print('  2. The structure becomes unstable as elements deactivate')
                        print('  3. Loads cause elements to repeatedly activate/deactivate')
                        print('')
                        print('Suggestions:')
                        print('  - Increase max_iter if convergence is nearly achieved')
                        print('  - Reduce num_steps for better load stepping')
                        print('  - Check if T/C-only element arrangement is physically sensible')
                        print('  - Consider using regular elements for some members')
                        print('')

                        diagnostics = ModelDiagnostics(self)
                        report = diagnostics.run_full_diagnosis()
                        diagnostic_text = report.format(verbose=True)
                        print(diagnostic_text)

                        raise Analysis.AnalysisError(
                            'Model diverged during tension/compression-only analysis',
                            diagnostic_text
                        )

                    # Report which load step we are on
                    if log:
                        print(f'- Analyzing load step #{str(load_step)}')

                    # Get the partitioned global stiffness matrix K11, K12, K21, K22
                    if sparse == True:
                        K11, K12, K21, K22 = Analysis._partition(self, self.K(combo.name, log, check_stability, sparse).tolil(), D1_indices, D2_indices)
                    else:
                        K11, K12, K21, K22 = Analysis._partition(self, self.K(combo.name, log, check_stability, sparse), D1_indices, D2_indices)

                    if K11.shape == (0, 0):
                        # All displacements are known, so Delta_D1 is an empty vector
                        Delta_D1 = []
                    else:
                        try:
                            # Calculate the unknown displacements Delta_D1
                            if sparse == True:
                                # The partitioned stiffness matrix is in `lil` format, which is great for memory, but slow for mathematical operations. The stiffness matrix will be converted to `csr` format for mathematical operations. The `@` operator performs matrix multiplication on sparse matrices.
                                Delta_D1 = spsolve(K11.tocsr(), subtract(subtract(Delta_P1, Delta_FER1), K12.tocsr() @ Delta_D2))
                                Delta_D1 = Delta_D1.reshape(len(Delta_D1), 1)
                            else:
                                Delta_D1 = solve(K11, subtract(subtract(Delta_P1, Delta_FER1), matmul(K12, Delta_D2)))

                            # Check for NaN or Inf values which indicate a singular matrix
                            # (scipy spsolve may not raise an exception for singular matrices)
                            import numpy as np
                            if np.any(np.isnan(Delta_D1)) or np.any(np.isinf(Delta_D1)):
                                raise ValueError("Solution contains NaN or Inf values - matrix is singular")

                        except Exception as e:
                            # Return out of the method if 'K' is singular and provide an error message
                            # Run diagnostics to explain why the matrix is singular
                            from Pynite.Diagnostics import ModelDiagnostics
                            print('')
                            print('=' * 60)
                            print('ANALYSIS FAILED - Singular Stiffness Matrix')
                            print('=' * 60)
                            print('')
                            print('The stiffness matrix could not be inverted, which means the')
                            print('structure has one or more rigid body modes (it can move freely).')
                            print('')
                            print('Running diagnostics to identify root cause...')
                            print('')

                            diagnostics = ModelDiagnostics(self)
                            report = diagnostics.run_full_diagnosis()
                            diagnostic_text = report.format(verbose=True)
                            print(diagnostic_text)

                            raise Analysis.AnalysisError(
                                'The stiffness matrix is singular (structure is unstable)',
                                diagnostic_text
                            ) from e

                    # Store or sum the calculated displacements to the model and the nodes in the model
                    if load_step == 1:
                        Analysis._store_displacements(self, Delta_D1, Delta_D2, D1_indices, D2_indices, combo)
                    else:
                        Analysis._sum_displacements(self, Delta_D1, Delta_D2, D1_indices, D2_indices, combo)

                    # Check for tension/compression-only convergence at this load step
                    convergence = Analysis._check_TC_convergence(self, combo.name, log=log, spring_tolerance=spring_tolerance, member_tolerance=member_tolerance)

                    if convergence == False:

                        if log:
                            print(f'- Undoing load step #{load_step} due to failed convergence.')

                        # Undo the latest analysis step to prepare for re-analysis of the load step
                        Analysis._sum_displacements(self, -Delta_D1, -Delta_D2, D1_indices, D2_indices, combo)

                    else:
                        # Move on to the next load step
                        load_step += 1

                    # Keep track of the number of tension/compression only iterations
                    iter_count += 1

        # Calculate reactions
        Analysis._calc_reactions(self, log, combo_tags)

        if log:
            print('')
            print('- Analysis complete')
            print('')

        # Check statics if requested
        if check_statics == True:
            Analysis._check_statics(self, combo_tags)

        # Flag the model as solved
        self.solution = 'Nonlinear TC'

    def analyze_PDelta(self, log=False, check_stability=True, max_iter=30, sparse=True, combo_tags=None):
        """Performs second order (P-Delta) analysis. This type of analysis is appropriate for most models using beams, columns and braces. Second order analysis is usually required by material specific codes. The analysis is iterative and takes longer to solve. Models with slender members and/or members with combined bending and axial loads will generally have more significant P-Delta effects. P-Delta effects in plates/quads are not considered.

        :param log: Prints updates to the console if set to True. Default is False.
        :type log: bool, optional
        :param check_stability: When set to True, checks the stiffness matrix for any unstable degrees of freedom and reports them back to the console. This does add to the solution time. Defaults to True.
        :type check_stability: bool, optional
        :param max_iter: The maximum number of iterations permitted. If this value is exceeded the program will report divergence. Defaults to 30.
        :type max_iter: int, optional
        :param sparse: Indicates whether the sparse matrix solver should be used. A matrix can be considered sparse or dense depening on how many zero terms there are. Structural stiffness matrices often contain many zero terms. The sparse solver can offer faster solutions for such matrices. Using the sparse solver on dense matrices may lead to slower solution times. Be sure ``scipy`` is installed to use the sparse solver. Default is True.
        :type sparse: bool, optional
        :raises ValueError: Occurs when there is a singularity in the stiffness matrix, which indicates an unstable structure.
        :raises Exception: Occurs when a model fails to converge.
        """

        if log:
            print('+--------------------+')
            print('| Analyzing: P-Delta |')
            print('+--------------------+')

        # Import `scipy` features if the sparse solver is being used
        if sparse == True:
            from scipy.sparse.linalg import spsolve

        # Prepare the model for analysis
        Analysis._prepare_model(self)

        # Get the auxiliary list used to determine how the matrices will be partitioned
        D1_indices, D2_indices, D2 = Analysis._partition_D(self)

        # Identify which load combinations have the tags the user has given
        combo_list = Analysis._identify_combos(self, combo_tags)

        # Step through each load combination
        for combo in combo_list:

            # Get the partitioned global fixed end reaction vector
            FER1, FER2 = Analysis._partition(self, self.FER(combo.name), D1_indices, D2_indices)

            # Get the partitioned global nodal force vector
            P1, P2 = Analysis._partition(self, self.P(combo.name), D1_indices, D2_indices)

            # Run the P-Delta analysis for this load combination
            Analysis._PDelta(self, combo.name, P1, FER1, D1_indices, D2_indices, D2, log, sparse, check_stability, max_iter)

        # Calculate reactions
        Analysis._calc_reactions(self, log, combo_tags)

        if log:
            print('')
            print('- Analysis complete')
            print('')

        # Flag the model as solved
        self.solution = 'P-Delta'

    def _not_ready_yet_analyze_pushover(self, log=False, check_stability=True, push_combo='Push', max_iter=30, tol=0.01, sparse=True, combo_tags=None):

        if log:
            print('+---------------------+')
            print('| Analyzing: Pushover |')
            print('+---------------------+')

        # Prepare the model for analysis
        Analysis._prepare_model(self)

        # Get the auxiliary list used to determine how the matrices will be partitioned
        D1_indices, D2_indices, D2 = Analysis._partition_D(self)

        # Identify and tag the primary load combinations the pushover load will be added to
        for combo in self.load_combos.values():

            # No need to tag the pushover combo
            if combo.name != push_combo:

                # Add 'primary' to the combo's tags if it's not already there
                if combo.combo_tags is None:
                    combo.combo_tags = ['primary']
                elif 'primary' not in combo.combo_tags:
                    combo.combo_tags.append('primary')

        # Identify which load combinations have the tags the user has given
        # TODO: Remove the pushover combo istelf from `combo_list`
        combo_list = Analysis._identify_combos(self, combo_tags)
        combo_list = [combo for combo in combo_list if combo.name != push_combo]

        # Step through each load combination
        for combo in combo_list:

            # Skip the pushover combo
            if combo.name == push_combo:
                continue

            if log:
                print('')
                print('- Analyzing load combination ' + combo.name)

            # Reset nonlinear material member end forces to zero
            for phys_member in self.members.values():
                for sub_member in phys_member.sub_members.values():
                    sub_member._fxi, sub_member._myi, sub_member._mzi = 0, 0, 0
                    sub_member._fxj, sub_member._myj, sub_member._mzj = 0, 0, 0

            # Get the partitioned global fixed end reaction vector for the load combination
            FER1, FER2 = Analysis._partition(self, self.FER(combo.name), D1_indices, D2_indices)

            # Get the partitioned global nodal force vector for the load combination
            P1, P2 = Analysis._partition(self, self.P(combo.name), D1_indices, D2_indices)

            # Run an elastic P-Delta analysis for the load combination (w/o pushover loads)
            # This will be used to preload the member with non-pushover loads prior to pushover anlaysis
            Analysis._PDelta(self, combo.name, P1, FER1, D1_indices, D2_indices, D2, False, sparse, check_stability, 30)

            # The previous step flagged the solution as a P-Delta solution, but we need to indicate that this is actually a Pushover solution so that the calls to Member3D.f() are excecuted considering nonlinear behavior
            self.solution = 'Pushover'

            # Get the partitioned global fixed end reaction vector for a pushover load increment
            FER1_push, FER2_push = Analysis._partition(self, self.FER(push_combo), D1_indices, D2_indices)

            # Get the partitioned global nodal force vector for a pushover load increment
            P1_push, P2_push = Analysis._partition(self, self.P(push_combo), D1_indices, D2_indices)

            # Get the pushover load step and initialize the load factor
            load_step = list(self.load_combos[push_combo].factors.values())[0]  # TODO: This line can probably live outside the loop
            load_factor = load_step
            step_num = 1

            # Apply the pushover load in steps, summing deformations as we go, until the full pushover load has been analyzed
            while round(load_factor, 8) <= 1.0:

                # Inform the user which pushover load step we're on
                if log:
                    print('- Beginning pushover load step #' + str(step_num))
                    print(f'- Load_factor = {load_factor}')

                # Run the next pushover load step
                Analysis._pushover_step(self, combo.name, push_combo, step_num, P1_push, FER1_push, D1_indices, D2_indices, D2, log, sparse, check_stability)

                # Update nonlinear material member end forces for each member
                for phys_member in self.members.values():

                    for member in phys_member.sub_members.values():

                        # Calculate the local member end force vector (once)
                        f = member.f(combo.name, push_combo, step_num)

                        # Store the end forces in the member
                        member._fxi = f[0, 0]
                        member._myi = f[4, 0]
                        member._mzi = f[5, 0]
                        member._fxj = f[6, 0]
                        member._myj = f[10, 0]
                        member._mzj = f[11, 0]

                # Move on to the next load step
                step_num += 1
                load_factor += load_step

        # Calculate reactions for every primary load combination
        Analysis._calc_reactions(self, log, combo_tags=['primary'])

        if log:
            print('')
            print('- Analysis complete')
            print('')

        # Flag the model as solved
        self.solution = 'Pushover'

    def get_all_member_forces(
        self,
        combo_names: List[str] | None = None,
        member_names: List[str] | None = None,
        n_points: int = 20,
        include_shear: bool = True,
        include_moment: bool = True,
        include_axial: bool = True,
        include_torque: bool = True
    ) -> Dict[str, Dict[str, NDArray[float64]]]:
        """
        Extract internal forces for multiple members using batched matrix operations.

        This method groups members by section properties and batches matrix
        multiplications across members, providing significant speedup for models
        with many similar members (e.g., wall studs).

        Parameters
        ----------
        combo_names : List[str] | None, optional
            List of load combination names to extract. If None, uses all combos.
        member_names : List[str] | None, optional
            List of member names to extract. If None, uses all members.
        n_points : int, optional
            Number of points along each member (default: 20).
        include_shear : bool, optional
            Include shear forces in output (default: True).
        include_moment : bool, optional
            Include bending moments in output (default: True).
        include_axial : bool, optional
            Include axial forces in output (default: True).
        include_torque : bool, optional
            Include torsion in output (default: True).

        Returns
        -------
        Dict[str, Dict[str, NDArray[float64]]]
            Nested dictionary: {member_name: {'x': array, 'Fx': array, ...}}
            Each inner dict has keys:
            - 'x': position array of shape (n_points,)
            - 'Fx': axial force, shape (n_combos, n_points)
            - 'Fy': shear y, shape (n_combos, n_points)
            - 'Fz': shear z, shape (n_combos, n_points)
            - 'Mx': torque, shape (n_combos, n_points)
            - 'My': moment y, shape (n_combos, n_points)
            - 'Mz': moment z, shape (n_combos, n_points)
            - 'dx': axial deflection, shape (n_combos, n_points)
            - 'dy': deflection y, shape (n_combos, n_points)
            - 'dz': deflection z, shape (n_combos, n_points)

        Examples
        --------
        >>> model.analyze()
        >>> forces = model.get_all_member_forces(['1.2D+1.6L'], n_points=20)
        >>> stud_moment = forces['Stud_1']['Mz']  # shape: (1, 20)

        Notes
        -----
        For wall-type structures with many identical studs, this method batches
        matrix operations across members sharing the same section properties,
        providing 2-4x speedup over sequential per-member extraction.
        """
        from collections import defaultdict
        from numpy import empty, zeros, linspace, einsum

        # Default to all combos and all members
        if combo_names is None:
            combo_names = list(self.load_combos.keys())
        if member_names is None:
            member_names = list(self.members.keys())

        n_combos = len(combo_names)
        results: Dict[str, Dict[str, NDArray[float64]]] = {}

        # Check if we can use fast path (linear analysis, simple members)
        use_fast_path = (self.solution != 'P-Delta' and self.solution != 'Pushover')

        # Group members by (material, section, length, i_node, j_node direction)
        # Members in same group share k matrix and can batch T @ D
        section_groups: Dict[Tuple[Any, ...], List[Any]] = defaultdict(list)

        for name in member_names:
            member = self.members[name]

            # Check if member can use fast path
            if not use_fast_path or not member._can_use_fast_path():
                # Fall back to per-member extraction for complex members
                results[name] = member.get_all_forces_array(combo_names, n_points)
                continue

            # Group key: section properties that determine k matrix
            # Round length to handle floating point comparison
            L_rounded = round(member.L(), 6)
            mat_name = member.material.name
            sec_name = member.section.name

            # Include direction vector to group members with same orientation
            # This allows batching the T matrix multiplication
            dx = member.j_node.X - member.i_node.X
            dy = member.j_node.Y - member.i_node.Y
            dz = member.j_node.Z - member.i_node.Z
            # Normalize and round to handle floating point
            L = member.L()
            if L > 0:
                dir_key = (round(dx/L, 6), round(dy/L, 6), round(dz/L, 6), round(member.rotation, 6))
            else:
                dir_key = (0, 0, 0, 0)

            # Include end releases - they affect the condensed stiffness matrix
            # Releases is a list of bools for each DOF
            releases_key = tuple(member.Releases)

            # Include tension/compression flags - they affect active state per combo
            tc_key = (member.tension_only, member.comp_only)

            key = (mat_name, sec_name, L_rounded, dir_key, releases_key, tc_key)
            section_groups[key].append(member)

        # Pre-build combo factor matrix for vectorized load accumulation
        # This eliminates nested Python loops when computing load totals
        # First, collect all unique load case names used in any combo
        all_cases = set()
        for combo_name in combo_names:
            all_cases.update(self.load_combos[combo_name].factors.keys())
        case_list = sorted(all_cases)  # Consistent ordering
        case_to_idx = {case: i for i, case in enumerate(case_list)}
        n_cases = len(case_list)

        # Build combo_matrix: shape (n_cases, n_combos)
        # combo_matrix[case_idx, combo_idx] = factor for that load case in that combo
        combo_matrix = zeros((n_cases, n_combos))
        for c_idx, combo_name in enumerate(combo_names):
            factors = self.load_combos[combo_name].factors
            for case, factor in factors.items():
                combo_matrix[case_to_idx[case], c_idx] = factor

        # Cache for x-coordinates per (L, n_points) to avoid repeated linspace calls
        x_cache: Dict[tuple, tuple] = {}

        # Process each group with batched operations
        for group_key, group_members in section_groups.items():
            n_members = len(group_members)

            if n_members == 1:
                # Single member - use standard extraction
                member = group_members[0]
                results[member.name] = member.get_all_forces_array(combo_names, n_points)
                continue

            # Get shared matrices from first member (all members in group have same k, T)
            ref_member = group_members[0]
            k = ref_member.k()  # 12x12 stiffness matrix
            T = ref_member.T()  # 12x12 transformation matrix
            L = ref_member.L()

            # Get or create cached x-coordinates for this (L, n_points)
            x_key = (round(L, 6), n_points)
            if x_key not in x_cache:
                x = linspace(0, L, n_points)
                x2 = x * x
                x3 = x2 * x
                x_cache[x_key] = (x, x2, x3)
            x, x2, x3 = x_cache[x_key]
            inv_2L = 1.0 / (2.0 * L)
            inv_6L = 1.0 / (6.0 * L)

            # Build batched global displacement array: (12, n_members, n_combos)
            D_batch = empty((12, n_members, n_combos))

            for m_idx, member in enumerate(group_members):
                i_node = member.i_node
                j_node = member.j_node
                is_active = member.active

                for c_idx, combo_name in enumerate(combo_names):
                    active = is_active.get(combo_name, True)
                    D_batch[0, m_idx, c_idx] = i_node.DX[combo_name] if active else 0.0
                    D_batch[1, m_idx, c_idx] = i_node.DY[combo_name]
                    D_batch[2, m_idx, c_idx] = i_node.DZ[combo_name]
                    D_batch[3, m_idx, c_idx] = i_node.RX[combo_name]
                    D_batch[4, m_idx, c_idx] = i_node.RY[combo_name]
                    D_batch[5, m_idx, c_idx] = i_node.RZ[combo_name]
                    D_batch[6, m_idx, c_idx] = j_node.DX[combo_name] if active else 0.0
                    D_batch[7, m_idx, c_idx] = j_node.DY[combo_name]
                    D_batch[8, m_idx, c_idx] = j_node.DZ[combo_name]
                    D_batch[9, m_idx, c_idx] = j_node.RX[combo_name]
                    D_batch[10, m_idx, c_idx] = j_node.RY[combo_name]
                    D_batch[11, m_idx, c_idx] = j_node.RZ[combo_name]

            # Batched transformation: d = T @ D for all members and combos
            # T: (12, 12), D_batch: (12, n_members, n_combos)
            # Result d_batch: (12, n_members, n_combos)
            d_batch = einsum('ij,jmc->imc', T, D_batch)

            # Batched force computation: f = k @ d for all members and combos
            # k: (12, 12), d_batch: (12, n_members, n_combos)
            # Result f_batch: (12, n_members, n_combos)
            f_batch = einsum('ij,jmc->imc', k, d_batch)

            # Add fer values directly to f_batch (avoids allocating separate fer_batch array)
            # member.fer() is already cached, so this is efficient
            for m_idx, member in enumerate(group_members):
                for c_idx, combo_name in enumerate(combo_names):
                    f_batch[:, m_idx, c_idx] += member.fer(combo_name).flatten()

            # Pre-compute T_rot for global direction loads (shared across group)
            T_rot = T[:3, :3]

            for m_idx, member in enumerate(group_members):
                # Extract end forces for this member: shape (12, n_combos)
                f_member = f_batch[:, m_idx, :]

                # End forces: P1 (axial), V1y/V1z (shear), M1y/M1z (moment), T1 (torque)
                P1_all = f_member[0, :]  # shape (n_combos,)
                V1y_all = f_member[1, :]  # Shear in local y direction
                V1z_all = f_member[2, :]  # Shear in local z direction
                T1_all = f_member[3, :]   # Torque
                M1y_all = f_member[4, :]  # Moment about local y axis
                M1z_all = f_member[5, :]  # Moment about local z axis

                # Check for distributed loads
                dist_loads = member.DistLoads
                has_dist_loads = len(dist_loads) > 0

                # Initialize result dict with x-coordinates
                # Use x.copy() to prevent aliasing bugs if downstream code mutates the array
                member_results: Dict[str, NDArray[float64]] = {'x': x.copy()}

                # Pre-allocate force arrays
                shear_y: NDArray[float64] = empty((n_combos, n_points))
                shear_z: NDArray[float64] = empty((n_combos, n_points))
                moment_z: NDArray[float64] = empty((n_combos, n_points))
                moment_y: NDArray[float64] = empty((n_combos, n_points))
                axial_arr: NDArray[float64] = empty((n_combos, n_points))
                torque_arr: NDArray[float64] = empty((n_combos, n_points))

                if has_dist_loads:
                    # Build load vectors for vectorized combo factor multiplication
                    # Each vector has shape (n_cases,) and we multiply by combo_matrix
                    # w = load in y direction, wz = load in z direction, p = axial load
                    w1_vec = zeros(n_cases)
                    w2_vec = zeros(n_cases)
                    wz1_vec = zeros(n_cases)
                    wz2_vec = zeros(n_cases)
                    p1_vec = zeros(n_cases)
                    p2_vec = zeros(n_cases)

                    for dist_load in dist_loads:
                        case = dist_load[5]
                        direction = dist_load[0]
                        w1_raw = dist_load[1]
                        w2_raw = dist_load[2]

                        if case not in case_to_idx:
                            continue  # Load case not used in any requested combo

                        case_idx = case_to_idx[case]
                        if direction == 'Fy':
                            w1_vec[case_idx] += w1_raw
                            w2_vec[case_idx] += w2_raw
                        elif direction == 'Fz':
                            wz1_vec[case_idx] += w1_raw
                            wz2_vec[case_idx] += w2_raw
                        elif direction == 'Fx':
                            p1_vec[case_idx] += w1_raw
                            p2_vec[case_idx] += w2_raw
                        elif direction in ('FX', 'FY', 'FZ'):
                            from numpy import array
                            FX = 1 if direction == 'FX' else 0
                            FY = 1 if direction == 'FY' else 0
                            FZ = 1 if direction == 'FZ' else 0
                            f1_local = T_rot @ array([FX * w1_raw, FY * w1_raw, FZ * w1_raw])
                            f2_local = T_rot @ array([FX * w2_raw, FY * w2_raw, FZ * w2_raw])
                            p1_vec[case_idx] += f1_local[0]
                            p2_vec[case_idx] += f2_local[0]
                            w1_vec[case_idx] += f1_local[1]
                            w2_vec[case_idx] += f2_local[1]
                            wz1_vec[case_idx] += f1_local[2]
                            wz2_vec[case_idx] += f2_local[2]

                    # Vectorized combo factor multiplication: totals = load_vec @ combo_matrix
                    # Result shape: (n_combos,) - one total per combo
                    w1_totals = w1_vec @ combo_matrix
                    w2_totals = w2_vec @ combo_matrix
                    wz1_totals = wz1_vec @ combo_matrix
                    wz2_totals = wz2_vec @ combo_matrix
                    p1_totals = p1_vec @ combo_matrix
                    p2_totals = p2_vec @ combo_matrix

                    dw = w2_totals - w1_totals
                    dwz = wz2_totals - wz1_totals
                    dp = p2_totals - p1_totals

                    shear_y = V1y_all[:, None] + w1_totals[:, None] * x + dw[:, None] * x2 * inv_2L
                    shear_z = V1z_all[:, None] + wz1_totals[:, None] * x + dwz[:, None] * x2 * inv_2L
                    moment_z = M1z_all[:, None] - V1y_all[:, None] * x - w1_totals[:, None] * x2 * 0.5 - dw[:, None] * x3 * inv_6L
                    moment_y = M1y_all[:, None] + V1z_all[:, None] * x + wz1_totals[:, None] * x2 * 0.5 + dwz[:, None] * x3 * inv_6L
                    axial_arr = P1_all[:, None] + p1_totals[:, None] * x + dp[:, None] * inv_2L * x2
                else:
                    # No distributed loads
                    shear_y[:] = V1y_all[:, None]
                    shear_z[:] = V1z_all[:, None]
                    moment_z = M1z_all[:, None] - V1y_all[:, None] * x
                    moment_y = M1y_all[:, None] + V1z_all[:, None] * x
                    axial_arr[:] = P1_all[:, None]

                # Torque is constant
                torque_arr[:] = T1_all[:, None]

                # Zero out forces for inactive members (tension/compression-only)
                is_active = member.active
                for c_idx, combo_name in enumerate(combo_names):
                    if not is_active.get(combo_name, True):
                        shear_y[c_idx, :] = 0.0
                        shear_z[c_idx, :] = 0.0
                        moment_z[c_idx, :] = 0.0
                        moment_y[c_idx, :] = 0.0
                        axial_arr[c_idx, :] = 0.0
                        torque_arr[c_idx, :] = 0.0

                # Build result dict with all force types
                member_results['Fx'] = axial_arr
                member_results['Fy'] = shear_y
                member_results['Fz'] = shear_z
                member_results['Mx'] = torque_arr
                member_results['My'] = moment_y
                member_results['Mz'] = moment_z

                # For deflections in the batched path, fall back to per-member extraction
                # since deflection computation requires segment integration
                full_results = member.get_all_forces_array(combo_names, n_points)
                member_results['dx'] = full_results['dx']
                member_results['dy'] = full_results['dy']
                member_results['dz'] = full_results['dz']

                results[member.name] = member_results

        return results

    def get_all_member_forces_array(
        self,
        combo_names: List[str] | None = None,
        member_names: List[str] | None = None,
        n_points: int = 20
    ) -> Dict[str, Union[List[str], NDArray[float64]]]:
        """
        Extract internal forces for all members as stacked 3D arrays.

        Returns contiguous arrays suitable for vectorized downstream operations
        like finding max forces across all members.

        Parameters
        ----------
        combo_names : List[str] | None, optional
            List of load combination names. If None, uses all combos.
        member_names : List[str] | None, optional
            List of member names. If None, uses all members.
        n_points : int, optional
            Number of points along each member (default: 20).

        Returns
        -------
        Dict[str, Union[List[str], NDArray[float64]]]
            Dictionary with keys:
            - 'member_names': List[str] of member names (for indexing)
            - 'combo_names': List[str] of combo names (for indexing)
            - 'x': array of shape (n_members, n_points) - positions along each member
            - 'Fx': array of shape (n_members, n_combos, n_points) - axial force
            - 'Fy': array of shape (n_members, n_combos, n_points) - shear y
            - 'Fz': array of shape (n_members, n_combos, n_points) - shear z
            - 'Mx': array of shape (n_members, n_combos, n_points) - torque
            - 'My': array of shape (n_members, n_combos, n_points) - moment y
            - 'Mz': array of shape (n_members, n_combos, n_points) - moment z
            - 'dx': array of shape (n_members, n_combos, n_points) - axial deflection
            - 'dy': array of shape (n_members, n_combos, n_points) - deflection y
            - 'dz': array of shape (n_members, n_combos, n_points) - deflection z

        Examples
        --------
        >>> model.analyze()
        >>> forces = model.get_all_member_forces_array(n_points=20)
        >>> # Get max moment across all members and combos
        >>> max_moment = np.max(np.abs(forces['Mz']))
        >>> # Get forces for specific member by index
        >>> idx = forces['member_names'].index('Stud_5')
        >>> stud5_shear = forces['Fy'][idx, :, :]

        Notes
        -----
        This method uses the optimized per-member extraction and stacks results
        into contiguous arrays for efficient downstream processing.
        """
        from numpy import empty

        # Default to all combos and all members
        if combo_names is None:
            combo_names = list(self.load_combos.keys())
        if member_names is None:
            member_names = list(self.members.keys())

        n_members = len(member_names)
        n_combos = len(combo_names)

        # Get per-member results
        forces_dict = self.get_all_member_forces(combo_names, member_names, n_points)

        # Pre-allocate output arrays
        x_all = empty((n_members, n_points))
        Fx_all = empty((n_members, n_combos, n_points))
        Fy_all = empty((n_members, n_combos, n_points))
        Fz_all = empty((n_members, n_combos, n_points))
        Mx_all = empty((n_members, n_combos, n_points))
        My_all = empty((n_members, n_combos, n_points))
        Mz_all = empty((n_members, n_combos, n_points))
        dx_all = empty((n_members, n_combos, n_points))
        dy_all = empty((n_members, n_combos, n_points))
        dz_all = empty((n_members, n_combos, n_points))

        # Stack results into contiguous arrays
        for i, member_name in enumerate(member_names):
            forces = forces_dict[member_name]
            x_all[i, :] = forces['x']
            Fx_all[i, :, :] = forces['Fx']
            Fy_all[i, :, :] = forces['Fy']
            Fz_all[i, :, :] = forces['Fz']
            Mx_all[i, :, :] = forces['Mx']
            My_all[i, :, :] = forces['My']
            Mz_all[i, :, :] = forces['Mz']
            dx_all[i, :, :] = forces['dx']
            dy_all[i, :, :] = forces['dy']
            dz_all[i, :, :] = forces['dz']

        return {
            'member_names': member_names,
            'combo_names': combo_names,
            'x': x_all,
            'Fx': Fx_all,
            'Fy': Fy_all,
            'Fz': Fz_all,
            'Mx': Mx_all,
            'My': My_all,
            'Mz': Mz_all,
            'dx': dx_all,
            'dy': dy_all,
            'dz': dz_all,
        }

    def unique_name(self, dictionary, prefix):
        """Returns the next available unique name for a dictionary of objects.

        :param dictionary: The dictionary to get a unique name for.
        :type dictionary: dict
        :param prefix: The prefix to use for the unique name.
        :type prefix: str
        :return: A unique name for the dictionary.
        :rtype: str
        """

        # Select a trial value for the next available name
        name = prefix + str(len(dictionary) + 1)
        i = 2
        while name in dictionary.keys():
            name = prefix + str(len(dictionary) + i)
            i += 1

        # Return the next available name
        return name


    def rename(self):
        """
        Renames all the nodes and elements in the model.
        """

        # Rename each node in the model
        temp = self.nodes.copy()
        id = 1
        for old_key in temp.keys():
            new_key = 'N' + str(id)
            self.nodes[new_key] = self.nodes.pop(old_key)
            self.nodes[new_key].name = new_key
            id += 1

        # Rename each spring in the model
        temp = self.springs.copy()
        id = 1
        for old_key in temp.keys():
            new_key = 'S' + str(id)
            self.springs[new_key] = self.springs.pop(old_key)
            self.springs[new_key].name = new_key
            id += 1

        # Rename each member in the model
        temp = self.members.copy()
        id = 1
        for old_key in temp.keys():
            new_key = 'M' + str(id)
            self.members[new_key] = self.members.pop(old_key)
            self.members[new_key].name = new_key
            id += 1

        # Rename each plate in the model
        temp = self.plates.copy()
        id = 1
        for old_key in temp.keys():
            new_key = 'P' + str(id)
            self.plates[new_key] = self.plates.pop(old_key)
            self.plates[new_key].name = new_key
            id += 1

        # Rename each quad in the model
        temp = self.quads.copy()
        id = 1
        for old_key in temp.keys():
            new_key = 'Q' + str(id)
            self.quads[new_key] = self.quads.pop(old_key)
            self.quads[new_key].name = new_key
            id += 1

    def orphaned_nodes(self):
        """
        Returns a list of the names of nodes that are not attached to any elements.
        """

        # Initialize a list of orphaned nodes
        orphans = []

        # Step through each node in the model
        for node in self.nodes.values():

            orphaned = False

            # Check to see if the node is attached to any elements
            quads = [quad.name for quad in self.quads.values() if quad.i_node == node or quad.j_node == node or quad.m_node == node or quad.n_node == node]
            plates = [plate.name for plate in self.plates.values() if plate.i_node == node or plate.j_node == node or plate.m_node == node or plate.n_node == node]
            members = [member.name for member in self.members.values() if member.i_node == node or member.j_node == node]
            springs = [spring.name for spring in self.springs.values() if spring.i_node == node or spring.j_node == node]

            # Determine if the node is orphaned
            if quads == [] and plates == [] and members == [] and springs == []:
                orphaned = True

            # Add the orphaned nodes to the list of orphaned nodes
            if orphaned == True:
                orphans.append(node.name)

        return orphans

    def diagnose(self, verbose: bool = True, include_info: bool = False) -> 'DiagnosticReport':
        """
        Run comprehensive diagnostics on the model to identify potential issues.

        This method checks for common modeling problems that could prevent
        successful analysis or cause unexpected results. It's recommended to
        run this before attempting analysis on a new or modified model.

        Parameters
        ----------
        verbose : bool, optional
            If True, includes detailed descriptions and suggestions for each
            issue found (default: True).
        include_info : bool, optional
            If True, includes informational messages in addition to errors
            and warnings (default: False).

        Returns
        -------
        DiagnosticReport
            A report object containing all findings. The report can be printed
            or accessed programmatically.

        Examples
        --------
        >>> model = FEModel3D()
        >>> # ... build model ...
        >>> report = model.diagnose()
        >>> print(report.format())

        >>> # Check if model is ready for analysis
        >>> if not report.has_errors:
        ...     model.analyze()
        ... else:
        ...     print("Fix errors before analyzing")
        """
        from Pynite.Diagnostics import ModelDiagnostics

        diagnostics = ModelDiagnostics(self)
        report = diagnostics.run_full_diagnosis()

        # Print the report
        print(report.format(verbose=verbose, include_info=include_info))

        return report

    def check_connectivity(self) -> str:
        """
        Get a summary of the model's structural connectivity.

        This method analyzes how nodes are connected through structural
        elements and identifies any disconnected components or floating nodes.

        Returns
        -------
        str
            A formatted string describing the connectivity of the model.

        Examples
        --------
        >>> model = FEModel3D()
        >>> # ... build model ...
        >>> print(model.check_connectivity())
        Connectivity Summary:
          - 1 connected component(s)
            Component 1: 10 nodes (with supports)
        """
        from Pynite.Diagnostics import get_connectivity_summary

        summary = get_connectivity_summary(self)
        print(summary)
        return summary

    def connectivity_graph(self, format: str = 'text') -> str:
        """
        Generate a graph representation of the model's node-member connectivity.

        This method produces a visual/textual representation showing which nodes
        connect to which elements, useful for debugging connectivity issues.

        Parameters
        ----------
        format : str, optional
            Output format (default: 'text'):
            - 'text': Readable adjacency list with node markers
            - 'dot': Graphviz DOT format for visualization

        Returns
        -------
        str
            Graph representation in the specified format.

        Examples
        --------
        >>> model = FEModel3D()
        >>> # ... build model ...
        >>> print(model.connectivity_graph())

        >>> # Export to Graphviz for visualization:
        >>> with open('model.dot', 'w') as f:
        ...     f.write(model.connectivity_graph(format='dot'))
        >>> # Then run: dot -Tpng model.dot -o model.png
        """
        from Pynite.Diagnostics import get_connectivity_graph

        graph = get_connectivity_graph(self, format=format)
        if format == 'text':
            print(graph)
        return graph
