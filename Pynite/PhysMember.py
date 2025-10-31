from __future__ import annotations # Allows more recent type hints features

import hashlib
from math import isclose, sqrt
from typing import Callable, Dict, List, Literal, Tuple, TYPE_CHECKING

import numpy as np
from numpy import array

from Pynite.Member3D import Member3D

if TYPE_CHECKING:

    from Pynite.Node3D import Node3D
    from Pynite.FEModel3D import FEModel3D
    import numpy.typing as npt
    from numpy import float64
    from numpy.typing import NDArray

class PhysMember(Member3D):
    """
    A physical member.

    Physical members can detect internal nodes and subdivide themselves into sub-members at those
    nodes.
    """

    def __init__(self, model: FEModel3D, name: str, i_node: Node3D, j_node: Node3D, material_name: str, section_name: str, rotation: float = 0.0,
                 tension_only: bool = False, comp_only: bool = False) -> None:

        super().__init__(model, name, i_node, j_node, material_name, section_name, rotation, tension_only, comp_only)
        self.sub_members: Dict[str, Member3D] = {}
        # Track the last discretization inputs so we can skip regeneration unless something changed
        self._discretize_signature: Tuple[int, float, float, float, float, float, float] | None = None
        self._result_array_cache: dict[Tuple[str, str, str, Tuple], NDArray[float64]] = {}
        self._x_cache: dict[Tuple[int, float], NDArray[float64]] = {}
        self._submember_result_cache: dict[
            Tuple[str, str, str], Dict[str, Dict[Tuple, NDArray[float64]]]
        ] = {}
        self._load_case_scaling: dict[str, Tuple[str, float]] | None = None

    def discretize(self) -> None:
        """
        Subdivides the physical member into sub-members at each node along the physical member
        """

        model_rev = getattr(self.model, '_node_revision', None)
        signature = (model_rev, self.i_node.X, self.i_node.Y, self.i_node.Z,
                     self.j_node.X, self.j_node.Y, self.j_node.Z)
        if self._discretize_signature == signature and self.sub_members:
            # No topology change since the last run; keep the existing sub-member layout
            return

        # Clear out any old sub_members
        self.sub_members = {}
        self._result_array_cache.clear()
        self._x_cache.clear()
        self._submember_result_cache.clear()

        # Start a new list of nodes along the member
        int_nodes: List[Tuple[Node3D, float]] = []

        # Create a vector from the i-node to the j-node
        Xi, Yi, Zi = self.i_node.X, self.i_node.Y, self.i_node.Z
        Xj, Yj, Zj = self.j_node.X, self.j_node.Y, self.j_node.Z

        dx = Xj - Xi
        dy = Yj - Yi
        dz = Zj - Zi

        length_sq = dx*dx + dy*dy + dz*dz
        if isclose(length_sq, 0.0):
            # Degenerate member - nothing to discretize
            int_nodes.append((self.i_node, 0.0))
            int_nodes.append((self.j_node, 0.0))
        else:
            length = sqrt(length_sq)

            # Add the i-node and j-node to the list
            int_nodes.append((self.i_node, 0.0))
            int_nodes.append((self.j_node, length))

            # Pull cached coordinate arrays when available to avoid repeated object lookups
            coords = getattr(self.model, '_node_coord_array', None)
            nodes_by_id = getattr(self.model, '_nodes_by_id', None)

            # Allow small angular deviation for floating point noise
            colinear_tol = 1e-12

            if coords is not None and nodes_by_id is not None:
                coords_x = coords[:, 0]
                coords_y = coords[:, 1]
                coords_z = coords[:, 2]

                axis_tol = 1e-9
                abs_dx = abs(dx)
                abs_dy = abs(dy)
                abs_dz = abs(dz)

                lookup = getattr(self.model, '_axis_node_lookup', None)
                round_digits = getattr(self.model, '_coord_round_digits', 9)
                handled = False

                if abs_dy <= axis_tol and abs_dz <= axis_tol:
                    # Member runs along the global X axis
                    handled = True
                    # Use the precomputed axis buckets when available to avoid scanning every node
                    if lookup:
                        key = (round(Yi, round_digits), round(Zi, round_digits))
                        nodes_on_line = lookup['x'].get(key)
                        if nodes_on_line:
                            for node in nodes_on_line:
                                if node is self.i_node or node is self.j_node:
                                    continue
                                if dx > 0 and Xi < node.X < Xj:
                                    int_nodes.append((node, node.X - Xi))
                                elif dx < 0 and Xj < node.X < Xi:
                                    int_nodes.append((node, Xi - node.X))
                    else:
                        axis_mask = (np.abs(coords_y - Yi) <= axis_tol) & (np.abs(coords_z - Zi) <= axis_tol)
                        axis_mask[self.i_node.ID] = False
                        axis_mask[self.j_node.ID] = False
                        candidate_ids = np.nonzero(axis_mask)[0]
                        if candidate_ids.size:
                            cx = (coords_x - Xi)[axis_mask]
                            between = ((cx*dx) > 0.0) & (np.abs(cx) < length)
                            candidate_ids = candidate_ids[between]
                            if candidate_ids.size:
                                distances = np.abs(cx[between])
                                for node_id, dist in zip(candidate_ids.tolist(), distances.tolist()):
                                    int_nodes.append((nodes_by_id[node_id], dist))
                elif abs_dx <= axis_tol and abs_dz <= axis_tol:
                    # Member runs along the global Y axis
                    handled = True
                    # Use the precomputed axis buckets when available to avoid scanning every node
                    if lookup:
                        key = (round(Xi, round_digits), round(Zi, round_digits))
                        nodes_on_line = lookup['y'].get(key)
                        if nodes_on_line:
                            for node in nodes_on_line:
                                if node is self.i_node or node is self.j_node:
                                    continue
                                if dy > 0 and Yi < node.Y < Yj:
                                    int_nodes.append((node, node.Y - Yi))
                                elif dy < 0 and Yj < node.Y < Yi:
                                    int_nodes.append((node, Yi - node.Y))
                    else:
                        axis_mask = (np.abs(coords_x - Xi) <= axis_tol) & (np.abs(coords_z - Zi) <= axis_tol)
                        axis_mask[self.i_node.ID] = False
                        axis_mask[self.j_node.ID] = False
                        candidate_ids = np.nonzero(axis_mask)[0]
                        if candidate_ids.size:
                            cy = (coords_y - Yi)[axis_mask]
                            between = ((cy*dy) > 0.0) & (np.abs(cy) < length)
                            candidate_ids = candidate_ids[between]
                            if candidate_ids.size:
                                distances = np.abs(cy[between])
                                for node_id, dist in zip(candidate_ids.tolist(), distances.tolist()):
                                    int_nodes.append((nodes_by_id[node_id], dist))
                elif abs_dx <= axis_tol and abs_dy <= axis_tol:
                    # Member runs along the global Z axis
                    handled = True
                    # Use the precomputed axis buckets when available to avoid scanning every node
                    if lookup:
                        key = (round(Xi, round_digits), round(Yi, round_digits))
                        nodes_on_line = lookup['z'].get(key)
                        if nodes_on_line:
                            for node in nodes_on_line:
                                if node is self.i_node or node is self.j_node:
                                    continue
                                if dz > 0 and Zi < node.Z < Zj:
                                    int_nodes.append((node, node.Z - Zi))
                                elif dz < 0 and Zj < node.Z < Zi:
                                    int_nodes.append((node, Zi - node.Z))
                    else:
                        axis_mask = (np.abs(coords_x - Xi) <= axis_tol) & (np.abs(coords_y - Yi) <= axis_tol)
                        axis_mask[self.i_node.ID] = False
                        axis_mask[self.j_node.ID] = False
                        candidate_ids = np.nonzero(axis_mask)[0]
                        if candidate_ids.size:
                            cz = (coords_z - Zi)[axis_mask]
                            between = ((cz*dz) > 0.0) & (np.abs(cz) < length)
                            candidate_ids = candidate_ids[between]
                            if candidate_ids.size:
                                distances = np.abs(cz[between])
                                for node_id, dist in zip(candidate_ids.tolist(), distances.tolist()):
                                    int_nodes.append((nodes_by_id[node_id], dist))

                if not handled:
                    # General skew member: test each node's projection onto the member axis
                    axis_mask = np.ones(len(coords), dtype=bool)
                    axis_mask[self.i_node.ID] = False
                    axis_mask[self.j_node.ID] = False

                    candidate_ids_all = np.nonzero(axis_mask)[0]
                    if candidate_ids_all.size:
                        offset_x = (coords_x - Xi)[axis_mask]
                        offset_y = (coords_y - Yi)[axis_mask]
                        offset_z = (coords_z - Zi)[axis_mask]

                        projections = offset_x*dx + offset_y*dy + offset_z*dz
                        node_len_sq = offset_x*offset_x + offset_y*offset_y + offset_z*offset_z
                        cross_x = dy*offset_z - dz*offset_y
                        cross_y = dz*offset_x - dx*offset_z
                        cross_z = dx*offset_y - dy*offset_x
                        cross_sq = cross_x*cross_x + cross_y*cross_y + cross_z*cross_z

                        between = (projections > 0.0) & (projections < length_sq)
                        nonzero_len = node_len_sq > 0.0
                        near_axis = cross_sq <= colinear_tol * length_sq * node_len_sq

                        mask = between & nonzero_len & near_axis
                        candidate_ids = candidate_ids_all[mask]

                        if candidate_ids.size:
                            distances = projections[mask] / length
                            for node_id, dist in zip(candidate_ids.tolist(), distances.tolist()):
                                int_nodes.append((nodes_by_id[node_id], dist))
            else:
                # Fallback path when cached arrays are unavailable: loop through nodes
                tol = 1e-9
                min_x = min(Xi, Xj) - tol
                max_x = max(Xi, Xj) + tol
                min_y = min(Yi, Yj) - tol
                max_y = max(Yi, Yj) + tol
                min_z = min(Zi, Zj) - tol
                max_z = max(Zi, Zj) + tol

                for node in self.model.nodes.values():

                    if node is self.i_node or node is self.j_node:
                        continue

                    X, Y, Z = node.X, node.Y, node.Z

                    if (X < min_x or X > max_x or
                        Y < min_y or Y > max_y or
                        Z < min_z or Z > max_z):
                        continue

                    vx = X - Xi
                    vy = Y - Yi
                    vz = Z - Zi

                    node_len_sq = vx*vx + vy*vy + vz*vz
                    if node_len_sq == 0.0:
                        continue

                    proj = vx*dx + vy*dy + vz*dz
                    if proj <= 0.0 or proj >= length_sq:
                        continue

                    cross_x = dy*vz - dz*vy
                    cross_y = dz*vx - dx*vz
                    cross_z = dx*vy - dy*vx
                    cross_sq = cross_x*cross_x + cross_y*cross_y + cross_z*cross_z

                    if cross_sq > colinear_tol * length_sq * node_len_sq:
                        continue

                    distance_along = proj / length
                    int_nodes.append((node, distance_along))

        # Create a list of sorted intermediate nodes by distance from the i-node
        int_nodes = sorted(int_nodes, key=lambda x: x[1])

        # Break up the member into sub-members at each intermediate node
        for i in range(len(int_nodes) - 1):

            # Generate the sub-member's name (physical member name + a, b, c, etc.)
            name = self.name + chr(i+97)

            # Find the i and j nodes for the sub-member, and their positions along the physical
            # member's local x-axis
            i_node = int_nodes[i][0]
            j_node = int_nodes[i+1][0]
            xi = int_nodes[i][1]
            xj = int_nodes[i+1][1]

            # Create a new sub-member
            new_sub_member = Member3D(self.model, name, i_node, j_node, self.material.name, self.section.name, self.rotation, self.tension_only, self.comp_only)

            # Flag the sub-member as active
            for combo_name in self.model.load_combos.keys():
                new_sub_member.active[combo_name] = True

            # Apply end releases if applicable
            if i == 0:
                new_sub_member.Releases[0:6] = self.Releases[0:6]
            if i == len(int_nodes) - 2:
                new_sub_member.Releases[6:12] = self.Releases[6:12]

            # Add distributed to the sub-member
            for dist_load in self.DistLoads:

                # Find the start and end points of the distributed load in the physical member's
                # local coordinate system
                x1_load = dist_load[3]
                x2_load = dist_load[4]

                # Determine if the distributed load should be applied to this segment
                if x1_load <= xj and x2_load > xi: 

                    direction = dist_load[0]
                    w1 = dist_load[1]
                    w2 = dist_load[2]
                    case = dist_load[5]

                    # Equation describing the load as a function of x
                    w = lambda x: (w2 - w1)/(x2_load - x1_load)*(x - x1_load) + w1

                    # Chop up the distributed load for the sub-member
                    if x1_load > xi:
                        x1 = x1_load - xi
                    else:
                        x1 = 0
                        w1 = w(xi)

                    if x2_load < xj:
                        x2 = x2_load - xi
                    else:
                        x2 = xj - xi
                        w2 = w(xj)

                    # Add the load to the sub-member
                    new_sub_member.DistLoads.append([direction, w1, w2, x1, x2, case])

            # Add point loads to the sub-member
            for pt_load in self.PtLoads:

                direction = pt_load[0]
                P = pt_load[1]
                x = pt_load[2]
                case = pt_load[3]

                # Determine if the point load should be applied to this segment
                if x >= xi and x < xj or (isclose(x, xj) and isclose(xj, self.L())):

                    x = x - xi

                    # Add the load to the sub-member
                    new_sub_member.PtLoads.append([direction, P, x, case])

            # Add the new sub-member to the sub-member dictionary for this physical member
            self.sub_members[name] = new_sub_member

        # Remember the inputs that produced this discretization so we can skip redundant work
        self._discretize_signature = signature

    def shear(self, Direction: Literal['Fy', 'Fz'], x: float, combo_name: str = 'Combo 1') -> float:
        """
        Returns the shear at a point along the member's length.

        Parameters
        ----------
        Direction : string
            The direction in which to find the shear. Must be one of the following:
                'Fy' = Shear acting on the local y-axis.
                'Fz' = Shear acting on the local z-axis.
        x : number
            The location at which to find the shear.
        combo_name : string
            The name of the load combination to get the results for (not the combination itself).
        """

        member, x_mod = self.find_member(x)
        return member.shear(Direction, x_mod, combo_name)

    def max_shear(self, Direction: Literal['Fy', 'Fz'], combo_name: str = 'Combo 1') -> float:
        """
        Returns the maximum shear in the member for the given direction

        Parameters
        ----------
        Direction : string
            The direction in which to find the maximum shear. Must be one of the following:
                'Fy' = Shear acting on the local y-axis
                'Fz' = Shear acting on the local z-axis
        combo_name : string
            The name of the load combination to get the results for (not the combination itself).
        """

        Vmax = None
        for member in self.sub_members.values():
            V = member.max_shear(Direction, combo_name)
            if Vmax is None or V > Vmax:
                Vmax = V
        return Vmax

    def min_shear(self, Direction: Literal['Fy', 'Fz'], combo_name: str = 'Combo 1') -> float:
        """
        Returns the minimum shear in the member for the given direction

        Parameters
        ----------
        Direction : string
            The direction in which to find the minimum shear. Must be one of the following:
                'Fy' = Shear acting on the local y-axis
                'Fz' = Shear acting on the local z-axis
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """

        Vmin = None
        for member in self.sub_members.values():
            V = member.min_shear(Direction, combo_name)
            if Vmin is None or V < Vmin:
                Vmin = V
        return Vmin

    def plot_shear(self, Direction: Literal['Fy', 'Fz'], combo_name: str = 'Combo 1', n_points: int = 20) -> None:
        """
        Plots the shear diagram for the member

        Parameters
        ----------
        Direction : string
            The direction in which to plot the shear force. Must be one of the following:
                'Fy' = Shear in the local y-axis.
                'Fz' = Shear in the local z-axis.
        combo_name : string
            The name of the load combination to get the results for (not the combination itself).
        n_points: int
            The number of points used to generate the plot
        """

        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "Matplotlib is required for plotting features.\n"
                "Install with: pip install PyNiteFEA[plotting]"
            ) from e

        fig, ax = plt.subplots()
        ax.axhline(0, color='black', lw=1)
        ax.grid()

        # Generate the shear diagram
        V_array = self.shear_array(Direction, n_points, combo_name)
        x = V_array[0]
        V = V_array[1]

        plt.plot(x, V)
        plt.ylabel('Shear')
        plt.xlabel('Location')
        plt.title('Member ' + self.name + '\n' + combo_name)
        plt.show()

    def _prepare_result_x_points(self, n_points: int, x_array=None) -> NDArray[float64]:
        """
        Normalize user-specified sampling points for result extraction.
        """
        L = self.L()
        if x_array is None:
            key = (int(n_points), float(np.round(L, 12)))
            cached = self._x_cache.get(key)
            if cached is None:
                cached = np.linspace(0.0, L, int(n_points), dtype='float64')
                cached.setflags(write=False)
                self._x_cache[key] = cached
            return cached

        x_vals = np.array(x_array, dtype='float64', copy=False)
        if x_vals.ndim != 1:
            raise ValueError("x_array must be a 1D array of coordinates")
        if x_vals.size and ((x_vals < 0.0).any() or (x_vals > L).any()):
            raise ValueError(f"All x values must be in the range 0 to {L}")
        x_vals.setflags(write=False)
        return x_vals

    def _collect_submember_results(
        self,
        family: str,
        direction: str,
        combo_name: str,
        x_vals: NDArray[float64],
        default_spacing: bool,
        x_signature: Tuple,
        evaluator: Callable[[Member3D, NDArray[float64]], NDArray[float64]],
    ) -> NDArray[float64]:
        """
        Evaluate submember response arrays and stitch them together for the full member.
        Results are memoized per submember so repeated queries with the same sampling scheme
        avoid re-evaluating the underlying segment polynomials.
        """
        if x_vals.size == 0:
            empty = np.empty((2, 0), dtype='float64')
            empty.setflags(write=False)
            return empty

        submembers = tuple(self.sub_members.values())
        if not submembers:
            empty = np.empty((2, 0), dtype='float64')
            empty.setflags(write=False)
            return empty

        lengths = np.fromiter((member.L() for member in submembers), dtype='float64')
        starts = np.zeros_like(lengths)
        if lengths.size > 1:
            np.cumsum(lengths[:-1], dtype='float64', out=starts[1:])
        ends = starts + lengths

        load_combo = self.model.load_combos[combo_name]
        cache_group_key = (family, direction, combo_name)
        group_cache = self._submember_result_cache.setdefault(cache_group_key, {})

        start_indices = np.searchsorted(x_vals, starts, side='left')
        end_indices = np.searchsorted(x_vals, ends, side='left')
        if end_indices.size:
            end_indices[-1] = np.searchsorted(x_vals, ends[-1], side='right')

        segments: List[NDArray[float64]] = []

        def make_slice_key(
            start_offset: float,
            start_idx: int,
            end_idx: int,
            segment_x: NDArray[float64],
        ) -> Tuple:
            if default_spacing:
                return ('lin', float(start_offset), start_idx, end_idx, x_signature)
            if segment_x.size == 0:
                return ('empty', float(start_offset))
            contiguous = np.ascontiguousarray(segment_x, dtype='float64')
            digest = hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).digest()
            return ('custom', float(start_offset), int(segment_x.size), digest)

        for idx, submember in enumerate(submembers):
            start_idx = start_indices[idx]
            end_idx = end_indices[idx]
            if end_idx <= start_idx:
                continue

            if submember._solved_combo is None or submember._solved_combo.name != combo_name:
                submember._segment_member(combo_name)
                submember._solved_combo = load_combo
                group_cache.pop(submember.name, None)

            segment_x = x_vals[start_idx:end_idx]
            if segment_x.size == 0:
                continue

            local_x = segment_x - starts[idx]
            sub_cache = group_cache.setdefault(submember.name, {})
            slice_key = make_slice_key(starts[idx], start_idx, end_idx, segment_x)

            cached_result = sub_cache.get(slice_key)
            if cached_result is not None:
                segments.append(cached_result)
                continue

            result = evaluator(submember, local_x.astype('float64', copy=False))
            if result is None or result.size == 0:
                continue

            if result.shape[0] != 2:
                raise ValueError("Result evaluator must return a 2xN array.")

            result = np.array(result, dtype='float64', copy=False, order='C')
            if not result.flags.writeable or not result.flags.owndata:
                result = result.copy(order='C')
            result[0] += starts[idx]
            result.setflags(write=False)
            sub_cache[slice_key] = result
            segments.append(result)

        if not segments:
            empty = np.empty((2, 0), dtype='float64')
            empty.setflags(write=False)
            return empty

        if len(segments) == 1:
            return segments[0]

        total_points = int(sum(segment.shape[1] for segment in segments))
        merged = np.empty((2, total_points), dtype='float64')
        cursor = 0
        for segment in segments:
            width = segment.shape[1]
            merged[:, cursor:cursor + width] = segment
            cursor += width
        merged.setflags(write=False)
        return merged

    def _build_x_signature(self, default_spacing: bool, x_vals: NDArray[float64]) -> Tuple:
        size = int(x_vals.size)
        if default_spacing:
            last = float(x_vals[-1]) if size else 0.0
            return ('lin', size, last)
        return ('custom', int(x_vals.size), hash(x_vals.tobytes()))

    def _ensure_load_case_scaling(self) -> None:
        if self._load_case_scaling is not None:
            return

        scaling: dict[str, Tuple[str, float]] = {}
        structure_map: dict[Tuple, Tuple[str, np.ndarray]] = {}
        tol = 1e-12

        # Organize distributed loads by case for quick lookup
        dist_by_case: dict[str, List[Tuple[str, float, float, float, float]]] = {}
        for direction, w1, w2, x1, x2, case in self.DistLoads:
            dist_by_case.setdefault(case, []).append((direction, float(x1), float(x2), float(w1), float(w2)))

        point_cases = {load[3] for load in self.PtLoads}
        case_names = sorted(set(dist_by_case.keys()) | point_cases)

        for case in case_names:
            # Point loads or absence of distributed loads make scaling detection unreliable—treat as unique
            if case in point_cases:
                scaling[case] = (case, 1.0)
                continue

            case_loads = dist_by_case.get(case)
            if not case_loads:
                scaling[case] = (case, 1.0)
                continue

            case_loads.sort(key=lambda item: (item[0], item[1], item[2]))
            structure_key = tuple((item[0], item[1], item[2]) for item in case_loads)

            magnitudes = np.empty(len(case_loads) * 2, dtype='float64')
            for idx, (_, _, _, w1, w2) in enumerate(case_loads):
                magnitudes[2 * idx] = w1
                magnitudes[2 * idx + 1] = w2

            base_info = structure_map.get(structure_key)
            if base_info is None:
                structure_map[structure_key] = (case, magnitudes)
                scaling[case] = (case, 1.0)
                continue

            base_case, base_vec = base_info
            nonzero = np.abs(base_vec) > tol

            if not np.any(nonzero):
                # Base loads are effectively zero—treat the new case as distinct
                structure_map[structure_key] = (case, magnitudes)
                scaling[case] = (case, 1.0)
                continue

            if np.any(np.abs(magnitudes[~nonzero]) > tol):
                # New case introduces loads where the base case had none—treat as unique
                structure_map[structure_key] = (case, magnitudes)
                scaling[case] = (case, 1.0)
                continue

            ratios = magnitudes[nonzero] / base_vec[nonzero]
            ratio_span = float(np.max(ratios) - np.min(ratios))

            if ratio_span <= 1e-9:
                scaling[case] = (base_case, float(ratios[0]))
            else:
                # Different distribution shape—treat as a unique base
                structure_map[structure_key] = (case, magnitudes)
                scaling[case] = (case, 1.0)

        self._load_case_scaling = scaling

    def _get_case_array(
        self,
        family: str,
        direction: str,
        x_vals: NDArray[float64],
        default_spacing: bool,
        x_signature: Tuple,
        case_name: str,
        evaluator: Callable[[Member3D, NDArray[float64]], NDArray[float64]],
    ) -> NDArray[float64] | None:
        case_key = self._result_cache_key(family, direction, case_name, x_signature)
        cached = self._result_array_cache.get(case_key)
        if cached is not None:
            return cached

        self._ensure_load_case_scaling()
        case_scaling = self._load_case_scaling or {}
        ref_case, scale_factor = case_scaling.get(case_name, (case_name, 1.0))

        base_combo = self.model._find_single_case_combo(ref_case)
        if base_combo is None:
            if case_name in self.model.load_combos:
                base_array = self._collect_submember_results(
                    family,
                    direction,
                    case_name,
                    x_vals,
                    default_spacing,
                    x_signature,
                    evaluator,
                )
                self._result_array_cache[case_key] = base_array
                return base_array
            return None

        base_key = self._result_cache_key(family, direction, base_combo, x_signature)
        base_array = self._result_array_cache.get(base_key)
        if base_array is None:
            base_array = self._collect_submember_results(
                family,
                direction,
                base_combo,
                x_vals,
                default_spacing,
                x_signature,
                evaluator,
            )
            self._result_array_cache[base_key] = base_array

        if ref_case != case_name or abs(scale_factor - 1.0) > 1e-9:
            scaled_array = self._scale_result_array(base_array, scale_factor)
            self._result_array_cache[case_key] = scaled_array
            return scaled_array

        self._result_array_cache[case_key] = base_array
        return base_array
    def _result_cache_key(
        self,
        family: str,
        direction: str,
        combo_name: str,
        x_signature: Tuple,
    ) -> Tuple[str, str, str, Tuple]:
        return (family, direction, combo_name, x_signature)

    @staticmethod
    def _scale_result_array(base_array: NDArray[float64], factor: float) -> NDArray[float64]:
        if abs(factor - 1.0) <= 1e-9:
            return base_array
        scaled = np.empty_like(base_array)
        scaled[0] = base_array[0]
        if base_array.shape[1]:
            np.multiply(base_array[1], factor, out=scaled[1])
        else:
            scaled[1] = base_array[1]
        scaled.setflags(write=False)
        return scaled

    def _compute_superposed_result(
        self,
        family: str,
        direction: str,
        combo_name: str,
        x_vals: NDArray[float64],
        default_spacing: bool,
        evaluator: Callable[[Member3D, NDArray[float64]], NDArray[float64]],
    ) -> NDArray[float64]:
        x_signature = self._build_x_signature(default_spacing, x_vals)
        cache_key = self._result_cache_key(family, direction, combo_name, x_signature)
        cached = self._result_array_cache.get(cache_key)
        if cached is not None:
            return cached

        combo = self.model.load_combos[combo_name]
        factors = [
            (case, float(factor))
            for case, factor in combo.factors.items()
            if abs(factor) > 1e-12
        ]

        if len(factors) == 1:
            case_name, factor = factors[0]
            case_array = self._get_case_array(
                family,
                direction,
                x_vals,
                default_spacing,
                x_signature,
                case_name,
                evaluator,
            )
            if case_array is not None:
                scaled_array = self._scale_result_array(case_array, factor)
                self._result_array_cache[cache_key] = scaled_array
                return scaled_array

        accumulator: NDArray[float64] | None = None
        x_reference: NDArray[float64] | None = None
        temp_buffer: NDArray[float64] | None = None

        for case_name, factor in factors:
            case_array = self._get_case_array(
                family,
                direction,
                x_vals,
                default_spacing,
                x_signature,
                case_name,
                evaluator,
            )
            if case_array is None:
                accumulator = None
                break

            if x_reference is None:
                x_reference = case_array[0]
                accumulator = np.array(case_array[1], dtype='float64', copy=True)
                accumulator *= factor
                temp_buffer = np.empty_like(accumulator)
            else:
                if case_array[0].size != x_reference.size or not np.array_equal(case_array[0], x_reference):
                    raise ValueError("Mismatched x-coordinates when superposing member results.")
                np.multiply(case_array[1], factor, out=temp_buffer)
                accumulator += temp_buffer

        if accumulator is not None and x_reference is not None:
            result = np.empty((2, x_reference.size), dtype='float64')
            result[0] = x_reference
            result[1] = accumulator
            result.setflags(write=False)
        else:
            result = self._collect_submember_results(
                family,
                direction,
                combo_name,
                x_vals,
                default_spacing,
                x_signature,
                evaluator,
            )

        self._result_array_cache[cache_key] = result
        return result

    def shear_array(self, Direction: Literal['Fy', 'Fz'], n_points: int, combo_name='Combo 1', x_array=None) -> NDArray[float64]:
        """
        Returns the array of the shear in the physical member for the given direction

        Parameters
        ----------
        Direction : string
            The direction to plot the shear for. Must be one of the following:
                'Fy' = Shear acting on the local y-axis.
                'Fz' = Shear acting on the local z-axis.
        n_points: int
            The number of points in the array to generate over the full length of the member.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        x_array : array = None
            A custom array of x values that may be provided by the user, otherwise an array is generated. Values must be provided in local member coordinates (between 0 and L) and be in ascending order
        """

        x_vals = self._prepare_result_x_points(n_points, x_array)
        default_spacing = x_array is None

        if Direction == 'Fz':
            segment_attr = 'SegmentsY'
        elif Direction == 'Fy':
            segment_attr = 'SegmentsZ'
        else:
            raise ValueError(f"Direction must be 'Fy' or 'Fz'. {Direction} was given.")

        def evaluator(submember: Member3D, local_x: NDArray[float64]) -> NDArray[float64]:
            segments = getattr(submember, segment_attr)
            return submember._extract_vector_results(segments, local_x, 'shear')

        return self._compute_superposed_result('shear', Direction, combo_name, x_vals, default_spacing, evaluator)

    def moment(self, Direction: Literal['My', 'Mz'], x: float, combo_name: str = 'Combo 1') -> float:
        """
        Returns the moment at a point along the member's length

        Parameters
        ----------
        Direction : string
            The direction in which to find the moment. Must be one of the following:
                'My' = Moment about the local y-axis.
                'Mz' = moment about the local z-axis.
        x : number
            The location at which to find the moment.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """

        member, x_mod = self.find_member(x)
        return member.moment(Direction, x_mod, combo_name)

    def max_moment(self, Direction: Literal['My', 'Mz'], combo_name: str = 'Combo 1') -> float:
        """
        Returns the maximum moment in the member for the given direction.
        
        Parameters
        ----------
        Direction : string
            The direction in which to find the maximum moment. Must be one of the following:
                'My' = Moment about the local y-axis.
                'Mz' = Moment about the local z-axis.
        combo_name : string
            The name of the load combination to get the results for (not the combination itself).
        """

        Mmax = None
        for member in self.sub_members.values():
            M = member.max_moment(Direction, combo_name)
            if Mmax is None or M > Mmax:
                Mmax = M
        return Mmax

    def min_moment(self, Direction: Literal['My', 'Mz'], combo_name: str = 'Combo 1') -> float:
        """
        Returns the minimum moment in the member for the given direction
        
        Parameters
        ----------
        Direction : string
            The direction in which to find the minimum moment. Must be one of the following:
                'My' = Moment about the local y-axis.
                'Mz' = Moment about the local z-axis.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """
        
        Mmin = None
        for member in self.sub_members.values():
            M = member.min_moment(Direction, combo_name)
            if Mmin is None or M < Mmin:
                Mmin = M
        return Mmin

    def plot_moment(self, Direction: Literal['My', 'Mz'], combo_name: str = 'Combo 1', n_points: int = 20) -> None:
        """
        Plots the moment diagram for the member

        Parameters
        ----------

        Direction : string
            The direction in which to plot the moment. Must be one of the following:
                'My' = Moment about the local y-axis.
                'Mz' = moment about the local z-axis.
        combo_name : string
            The name of the load combination to get the results for (not the combination itself).
        n_points: int
            The number of points used to generate the plot
        """

        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "Matplotlib is required for plotting features.\n"
                "Install with: pip install PyNiteFEA[plotting]"
            ) from e

        fig, ax = plt.subplots()
        ax.axhline(0, color='black', lw=1)
        ax.grid()

        # Generate the moment diagram
        M_array = self.moment_array(Direction, n_points, combo_name)
        x = M_array[0]
        M = M_array[1]

        plt.plot(x, M)
        plt.ylabel('Moment')
        plt.xlabel('Location')
        plt.title('Member ' + self.name + '\n' + combo_name)
        plt.show()

    def moment_array(self, Direction: Literal['My', 'Mz'], n_points: int, combo_name='Combo 1', x_array=None) -> NDArray[float64]:
        """
        Returns the array of the moment in the physical member for the given direction

        Parameters
        ----------
        Direction : string
            The direction to plot the moment for. Must be one of the following:
                'My' = Moment acting about the local y-axis (usually the weak-axis).
                'Mz' = Moment acting about the local z-axis (usually the strong-axis).
        n_points: int
            The number of points in the array to generate over the full length of the member.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        x_array : array = None
            A custom array of x values that may be provided by the user, otherwise an array is generated.
            Values must be provided in local member coordinates (between 0 and L) and be in ascending order
        """

        x_vals = self._prepare_result_x_points(n_points, x_array)
        default_spacing = x_array is None

        if Direction == 'My':
            segment_attr = 'SegmentsY'
        elif Direction == 'Mz':
            segment_attr = 'SegmentsZ'
        else:
            raise ValueError(f"Direction must be 'My' or 'Mz'. {Direction} was given.")

        include_pdelta = self.model.solution == 'P-Delta'

        def evaluator(submember: Member3D, local_x: NDArray[float64]) -> NDArray[float64]:
            segments = getattr(submember, segment_attr)
            return submember._extract_vector_results(segments, local_x, 'moment', include_pdelta)

        return self._compute_superposed_result('moment', Direction, combo_name, x_vals, default_spacing, evaluator)

    def torque(self, x: float, combo_name: str = 'Combo 1') -> float:
        """
        Returns the torsional moment at a point along the member's length
        
        Parameters
        ----------
        x : number
            The location at which to find the torque
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """
        
        member, x_mod = self.find_member(x)
        return member.torque(x_mod, combo_name)

    def max_torque(self, combo_name: str = 'Combo 1') -> float:
        
        Tmax = None
        for member in self.sub_members.values():
            T = member.max_torque(combo_name)
            if Tmax is None or T > Tmax:
                Tmax = T
        return Tmax

    def min_torque(self, combo_name: str = 'Combo 1') -> float:
        """
        Returns the minimum torsional moment in the member.

        Parameters
        ----------
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """

        Tmin = None
        for member in self.sub_members.values():
            T = member.min_torque(combo_name)
            if Tmin is None or T < Tmin:
                Tmin = T
        return Tmin

    def plot_torque(self, combo_name: str = 'Combo 1', n_points: int = 20) -> None:
        """
        Plots the torque diagram for the member

        Parameters
        ----------
        combo_name : string
            The name of the load combination to get the results for (not the combination itself).
        n_points: int
            The number of points used to generate the plot
        """

        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "Matplotlib is required for plotting features.\n"
                "Install with: pip install PyNiteFEA[plotting]"
            ) from e

        fig, ax = plt.subplots()
        ax.axhline(0, color='black', lw=1)
        ax.grid()

        # Generate the torque diagram
        T_array = self.torque_array(n_points, combo_name)
        x = T_array[0]
        T = T_array[1]

        plt.plot(x, T)
        plt.ylabel('Torque')
        plt.xlabel('Location')
        plt.title('Member ' + self.name + '\n' + combo_name)
        plt.show()

    def torque_array(self, n_points: int, combo_name='Combo 1', x_array=None) -> NDArray[float64]:
        """
        Returns the array of the torque in the physical member.

        Parameters
        ----------
        n_points: int
            The number of points in the array to generate over the full length of the member.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        x_array : array = None
            A custom array of x values that may be provided by the user, otherwise an array is generated. Values must be provided in local member coordinates (between 0 and L) and be in ascending order
        """

        x_vals = self._prepare_result_x_points(n_points, x_array)
        default_spacing = x_array is None

        def evaluator(submember: Member3D, local_x: NDArray[float64]) -> NDArray[float64]:
            return submember._extract_vector_results(submember.SegmentsX, local_x, 'torque')

        return self._compute_superposed_result('torque', 'T', combo_name, x_vals, default_spacing, evaluator)

    def axial(self, x: float, combo_name: str = 'Combo 1') -> float:
        """
        Returns the axial force at a point along the member's length.
        
        Parameters
        ----------
        x : number
            The location at which to find the axial force.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """

        member, x_mod = self.find_member(x)
        return member.axial(x_mod, combo_name)

    def max_axial(self, combo_name: str = 'Combo 1') -> float:
        
        Pmax = None
        for member in self.sub_members.values():
            P = member.max_axial(combo_name)
            if Pmax is None or P > Pmax:
                Pmax = P
        return Pmax

    def min_axial(self, combo_name: str = 'Combo 1') -> float:

        Pmin = None
        for member in self.sub_members.values():
            P = member.min_axial(combo_name)
            if Pmin is None or P < Pmin:
                Pmin = P
        return Pmin

    def plot_axial(self, combo_name: str = 'Combo 1', n_points: int = 20) -> None:
        """
        Plots the axial force diagram for the member

        Parameters
        ----------
        combo_name : string
            The name of the load combination to get the results for (not the combination itself).
        n_points: int
            The number of points used to generate the plot
        """

        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "Matplotlib is required for plotting features.\n"
                "Install with: pip install PyNiteFEA[plotting]"
            ) from e

        fig, ax = plt.subplots()
        ax.axhline(0, color='black', lw=1)
        ax.grid()

        # Generate the axial force array
        P_array = self.axial_array(n_points, combo_name)
        x = P_array[0]
        P = P_array[1]

        plt.plot(x, P)
        plt.ylabel('Axial Force')
        plt.xlabel('Location')
        plt.title('Member ' + self.name + '\n' + combo_name)
        plt.show()

    def axial_array(self, n_points: int, combo_name='Combo 1', x_array=None) -> NDArray[float64]:
        """
        Returns the array of the axial force in the physical member.

        Parameters
        ----------
        n_points: int
            The number of points in the array to generate over the full length of the member.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        x_array : array = None
            A custom array of x values that may be provided by the user, otherwise an array is generated. Values must be provided in local member coordinates (between 0 and L) and be in ascending order
        """

        x_vals = self._prepare_result_x_points(n_points, x_array)
        default_spacing = x_array is None

        def evaluator(submember: Member3D, local_x: NDArray[float64]) -> NDArray[float64]:
            return submember._extract_vector_results(submember.SegmentsZ, local_x, 'axial')

        return self._compute_superposed_result('axial', 'P', combo_name, x_vals, default_spacing, evaluator)

    def deflection(self, Direction: Literal['dx', 'dy', 'dz'], x: float, combo_name: str = 'Combo 1') -> float:
        """
        Returns the deflection at a point along the member's length.
        
        Parameters
        ----------
        Direction : string
            The direction in which to find the deflection. Must be one of the following:
                'dx' = Deflection in the local x-axis.
                'dy' = Deflection in the local y-axis.
                'dz' = Deflection in the local z-axis.
        x : number
            The location at which to find the deflection.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """

        member, x_mod = self.find_member(x)
        return member.deflection(Direction, x_mod, combo_name)

    def max_deflection(self, Direction: Literal['dx', 'dy', 'dz'], combo_name: str = 'Combo 1') -> float:
        """
        Returns the maximum deflection in the member.
        
        Parameters
        ----------
        Direction : {'dy', 'dz'}
            The direction in which to find the maximum deflection.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """

        dmax = None
        for member in self.sub_members.values():
            d = member.max_deflection(Direction, combo_name)
            if dmax is None or d > dmax:
                dmax = d
        return dmax

    def min_deflection(self, Direction: Literal['dx', 'dy', 'dz'], combo_name: str = 'Combo 1') -> float:
        """
        Returns the minimum deflection in the member.
        
        Parameters
        ----------
        Direction : {'dy', 'dz'}
            The direction in which to find the minimum deflection.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """

        dmin = None
        for member in self.sub_members.values():
            d = member.min_deflection(Direction, combo_name)
            if dmin is None or d < dmin:
                dmin = d
        return dmin

    def rel_deflection(self, Direction: Literal['dx', 'dy', 'dz'], x: float, combo_name: str = 'Combo 1') -> float:
        """
        Returns the relative deflection at a point along the member's length
        
        Parameters
        ----------

        Direction : string
            The direction in which to find the relative deflection. Must be one of the following:
                'dy' = Deflection in the local y-axis
                'dz' = Deflection in the local x-axis
        x : number
            The location at which to find the relative deflection
        combo_name : string
            The name of the load combination to get the results for (not the combination itself).
        """
        
        member, x_mod = self.find_member(x)
        return member.rel_deflection(Direction, x_mod, combo_name)

    def plot_deflection(self, Direction: Literal['dx', 'dy', 'dz'], combo_name: str = 'Combo 1', n_points: int = 20) -> None:
        """
        Plots the deflection diagram for the member

        Parameters
        ----------
        Direction : string
            The direction in which to plot the deflection. Must be one of the following:
                'dy' = Deflection in the local y-axis.
                'dz' = Deflection in the local z-axis.
        combo_name : string
            The name of the load combination to get the results for (not the combination itself).
        n_points: int
            The number of points used to generate the plot
        """

        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "Matplotlib is required for plotting features.\n"
                "Install with: pip install PyNiteFEA[plotting]"
            ) from e

        fig, ax = plt.subplots()
        ax.axhline(0, color='black', lw=1)
        ax.grid()

        d_array = self.deflection_array(Direction, n_points, combo_name)
        x = d_array[0]
        d = d_array[1]

        plt.plot(x, d)
        plt.ylabel('Deflection')
        plt.xlabel('Location')
        plt.title('Member ' + self.name + '\n' + combo_name)
        plt.show()

    def deflection_array(self, Direction: Literal['dx', 'dy', 'dz'], n_points: int, combo_name='Combo 1', x_array=None) -> NDArray[float64]:
        """
        Returns the array of the deflection in the physical member for the given direction

        Parameters
        ----------
        Direction : string
            The direction to plot the deflection for. Must be one of the following:
                'dx' = Deflection in the local x-direction (axial deflection)
                'dy' = Deflection in the local y-direction (usually the strong-axis).
                'dz' = Deflection in the local z-direction (usually the weak-axis).
        n_points: int
            The number of points in the array to generate over the full length of the member.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        x_array : array = None
            A custom array of x values that may be provided by the user, otherwise an array is generated. Values must be provided in local member coordinates (between 0 and L) and be in ascending order
        """

        x_vals = self._prepare_result_x_points(n_points, x_array)
        default_spacing = x_array is None

        if Direction == 'dx':
            segment_attr = 'SegmentsZ'
            result_name = 'axial_deflection'
        elif Direction == 'dy':
            segment_attr = 'SegmentsZ'
            result_name = 'deflection'
        elif Direction == 'dz':
            segment_attr = 'SegmentsY'
            result_name = 'deflection'
        else:
            raise ValueError(f"Direction must be 'dx', 'dy', or 'dz'. {Direction} was given.")

        def evaluator(submember: Member3D, local_x: NDArray[float64]) -> NDArray[float64]:
            segments = getattr(submember, segment_attr)
            return submember._extract_vector_results(segments, local_x, result_name)

        return self._compute_superposed_result('deflection', Direction, combo_name, x_vals, default_spacing, evaluator)

    def find_member(self, x: float) -> Tuple[Member3D, float]:
        """
        Returns the sub-member that the physical member's local point 'x' lies on, and 'x' modified for that sub-member's local coordinate system.
        """

        # Initialize a summation of sub-member lengths
        L = 0

        # Step through each sub-member (in order from start to end)
        for i, member in enumerate(self.sub_members.values()):

            # Sum the sub-member's length
            L += member.L()

            # Check if 'x' lies on this sub-member
            if x < L or (isclose(x, L) and i == len(self.sub_members.values()) - 1):

                # Return the sub-member, and a modified value for 'x' relative to the sub-member's
                # i-node
                return member, x - (L - member.L())

                # Exit the 'for' loop
                break
        else:
            raise ValueError(f"Location x={x} does not lie on this member")
