from __future__ import annotations # Allows more recent type hints features
from typing import Dict, List, Literal, Tuple, TYPE_CHECKING, Union
from Pynite.Member3D import Member3D

if TYPE_CHECKING:

    from Pynite.Node3D import Node3D
    from Pynite.FEModel3D import FEModel3D
    import numpy.typing as npt
    from numpy import float64
    from numpy.typing import NDArray

from numpy import array, dot, linspace, hstack, empty
from numpy.linalg import norm
from math import isclose, acos

class PhysMember(Member3D):
    """
    A physical member.

    Physical members can detect internal nodes and subdivide themselves into sub-members at those
    nodes.
    """
    
    # '__plt' is used to store the 'pyplot' from matplotlib once it gets imported. Setting it to 'None' for now allows us to defer importing it until it's actually needed.
    __plt = None  
    
    def __init__(self, model: FEModel3D, name: str, i_node: Node3D, j_node: Node3D, material_name: str, section_name: str, rotation: float = 0.0,
                 tension_only: bool = False, comp_only: bool = False) -> None:
        
        super().__init__(model, name, i_node, j_node, material_name, section_name, rotation, tension_only, comp_only)
        self.sub_members: Dict[str, Member3D] = {}


    def discretize(self) -> None:
        """
        Subdivides the physical member into sub-members at each node along the physical member
        using spatial acceleration (KDTree) for performance.
        """
        # Clear any existing sub-members and initialize the intermediate nodes list
        self.sub_members = {}
        int_nodes: List[Tuple[Node3D, float]] = []

        # Extract endpoint coordinates for geometric calculations
        Xi, Yi, Zi = self.i_node.X, self.i_node.Y, self.i_node.Z
        Xj, Yj, Zj = self.j_node.X, self.j_node.Y, self.j_node.Z
        
        # Calculate the member vector and total length
        vector_ij = array([Xj - Xi, Yj - Yi, Zj - Zi])
        L = float(norm(vector_ij))

        # Initialize intermediate nodes list with the member endpoints
        # Each tuple contains (node_object, distance_from_i_node)
        int_nodes.append((self.i_node, 0.0))
        int_nodes.append((self.j_node, L))

        # Use spatial acceleration (KDTree) to as an efficient prefilter to find nearby nodes
        # Candidate nodes are nodes that are no more than L/8 away from the member line
        # This avoids checking every node in large models and can greatly reduce the number of nodes checked
        if self.model._kd_tree is not None:
            # Calculate optimized search strategy using eighth-points to maximize volume reduction
            # Instead of searching a sphere centered at midpoint with radius L/2,
            # use four smaller spheres at eighth points with radius L/8
            # This reduces total search volume by ~90% while still capturing all relevant nodes
            
            eighth_L = L / 8
            buffer = 1e-6  # Add a small buffer to account for numerical precision
            
            # Calculate the four eighth points along the member
            # Points at 12.5%, 37.5%, 62.5%, and 87.5% ensure complete coverage
            eighth_positions = [0.125, 0.375, 0.625, 0.875]
            dx, dy, dz = Xj - Xi, Yj - Yi, Zj - Zi
            
            # Generate all eighth points efficiently
            eighth_points = []
            for pos in eighth_positions:
                point = array([Xi + pos * dx, Yi + pos * dy, Zi + pos * dz])
                eighth_points.append(point)
            
            # Search radius for each eighth-sphere (L/8 + buffer)
            radius = eighth_L + buffer

            # Query all four eighth-point regions and combine results
            all_indices = []
            for point in eighth_points:
                indices = self.model._kd_tree.query_ball_point(point, r=radius, p=2)
                all_indices.extend(indices)
            
            # Combine and deduplicate indices using set operations for efficiency
            idxs = list(set(all_indices))

            # Pre-allocate lists for better performance and avoid repeated lookups
            candidate_nodes = []
            kd_tree_node_names = self.model._kd_tree_node_names
            model_nodes = self.model.nodes
            
            # Batch convert indices to nodes (faster than list comprehension with nested lookups)
            for idx in idxs:
                node_name = kd_tree_node_names[idx]
                candidate_nodes.append(model_nodes[node_name])
        else:
            # Fallback: if no KDTree available, check all nodes in the model
            # This is slower but ensures compatibility
            candidate_nodes = list(self.model.nodes.values())

        # Check each candidate node to see if it lies exactly on the member line
        # Pre-compute values that don't change in the loop for better performance
        vector_ij_norm_sq = L * L  # Avoid repeated norm calculations
        tolerance_sq = 1e-12      # Square of tolerance to avoid sqrt in distance calculation
        
        for node in candidate_nodes:
            # Skip the member's own endpoints (fast identity check)
            if node is self.i_node or node is self.j_node:
                continue

            # Calculate vector from i-node to the candidate node
            dx, dy, dz = node.X - Xi, node.Y - Yi, node.Z - Zi
            
            # Project the candidate node onto the member's axis using dot product
            # proj_length = dot(vector_in, vector_ij) / L
            proj_length = (dx * vector_ij[0] + dy * vector_ij[1] + dz * vector_ij[2]) / L

            # Quick bounds check - must be within member length (excluding endpoints)
            if not (0 < proj_length < L):
                continue

            # Calculate perpendicular distance squared to avoid expensive sqrt operation
            # Use the formula: |v x u|² = |v|²|u|² - (v·u)²
            proj_length_ratio = proj_length / L
            proj_x = Xi + proj_length_ratio * vector_ij[0]
            proj_y = Yi + proj_length_ratio * vector_ij[1] 
            proj_z = Zi + proj_length_ratio * vector_ij[2]
            
            # Calculate squared distance from node to projection point
            offset_x, offset_y, offset_z = node.X - proj_x, node.Y - proj_y, node.Z - proj_z
            distance_to_line_sq = offset_x*offset_x + offset_y*offset_y + offset_z*offset_z

            # If the node is essentially on the line (within numerical tolerance)
            if distance_to_line_sq < tolerance_sq:
                # Add this node as an intermediate node with its position along the member
                int_nodes.append((node, proj_length))

        # Sort intermediate nodes by distance from the i-node
        int_nodes.sort(key=lambda x: x[1])

        # Create sub-members between each pair of consecutive intermediate nodes
        # Each sub-member spans from one node to the next along the physical member
        for i in range(len(int_nodes) - 1):
            # Generate unique sub-member name using alphabetic suffix (a, b, c, etc.)
            name = self.name + chr(i + 97)
            i_node, xi = int_nodes[i]      # Start node and its position along member
            j_node, xj = int_nodes[i + 1]  # End node and its position along member

            # Create a new sub-member spanning from i_node to j_node
            # This inherits all material and section properties from the parent physical member
            new_sub_member = Member3D(
                self.model, name, i_node, j_node, self.material.name,
                self.section.name, self.rotation, self.tension_only, self.comp_only
            )

            # Activate the sub-member for all existing load combinations in the model
            for combo_name in self.model.load_combos:
                new_sub_member.active[combo_name] = True

            # Transfer end releases from the physical member to appropriate sub-members
            # First sub-member gets the i-end (start) releases from the physical member
            if i == 0:
                new_sub_member.Releases[0:6] = self.Releases[0:6]
            # Last sub-member gets the j-end (end) releases from the physical member
            if i == len(int_nodes) - 2:
                new_sub_member.Releases[6:12] = self.Releases[6:12]

            # Helper function to linearly interpolate distributed load intensity at any position
            def interpolate_load(x: float) -> float:
                return (w2 - w1) / (x2_load - x1_load) * (x - x1_load) + w1

            # Distribute the physical member's distributed loads to applicable sub-members
            for dist_load in self.DistLoads:
                direction, w1, w2, x1_load, x2_load, case = dist_load
                
                # Check if this distributed load overlaps with the current sub-member
                if x1_load <= xj and x2_load > xi:
                    # Calculate the portion of the load that applies to this sub-member
                    # Convert from physical member coordinates to sub-member local coordinates
                    x1 = max(0, x1_load - xi)           # Start position relative to sub-member
                    x2 = min(xj - xi, x2_load - xi)     # End position relative to sub-member
                    
                    # Start with the original load intensities
                    w1_adjacent = w1
                    w2_adjacent = w2
                    
                    # If the load starts before this sub-member, interpolate the starting intensity
                    if x1_load < xi:
                        w1_adjacent = interpolate_load(xi)
                    
                    # If the load extends beyond this sub-member, interpolate the ending intensity
                    if x2_load > xj:
                        w2_adjacent = interpolate_load(xj)
                    
                    # Add the adjusted distributed load to the sub-member
                    new_sub_member.DistLoads.append((direction, w1_adjacent, w2_adjacent, x1, x2, case))

            # Distribute the physical member's point loads to applicable sub-members
            for pt_load in self.PtLoads:
                direction, P, x, case = pt_load
                
                # Check if this point load falls within the current sub-member's span
                # Special case: include loads exactly at the end if this is the last sub-member
                if xi <= x < xj or (isclose(x, xj) and isclose(xj, L)):
                    # Convert load position from physical member coordinates to sub-member coordinates
                    new_sub_member.PtLoads.append((direction, P, x - xi, case))

            # Store the completed sub-member in the dictionary
            self.sub_members[name] = new_sub_member
    
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

        # Import 'pyplot' if not already done
        if PhysMember.__plt is None:
            from matplotlib import pyplot as plt
            PhysMember.__plt = plt
        
        fig, ax = PhysMember.__plt.subplots()
        ax.axhline(0, color='black', lw=1)
        ax.grid()

        # Generate the shear diagram
        V_array = self.shear_array(Direction, n_points, combo_name)
        x = V_array[0]
        V = V_array[1]

        PhysMember.__plt.plot(x, V)
        PhysMember.__plt.ylabel('Shear')
        PhysMember.__plt.xlabel('Location')
        PhysMember.__plt.title('Member ' + self.name + '\n' + combo_name)
        PhysMember.__plt.show()

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

        # `v_array2` will be used to store the shear values for the overall member
        v_array2 = empty((2, 1))

        # Create an array of locations along the physical member to obtain results at
        L = self.L()
        if x_array is None:
            x_array = linspace(0, L, n_points)
        else:
            if any(x_array < 0) or any(x_array > L):
                raise ValueError(f"All x values must be in the range 0 to {L}")

        # Step through each submember in the physical member
        x_o = 0
        for i, submember in enumerate(self.sub_members.values()):

            # Segment the submember into segments with mathematically continuous loads if not already done
            if submember._solved_combo is None or combo_name != submember._solved_combo.name:
                submember._segment_member(combo_name)
                submember._solved_combo = self.model.load_combos[combo_name]

            # Check if this is the last submember
            if i == len(self.sub_members.values()) - 1:

                # Find any points from `x_array` that lie along this submember
                filter = (x_array >= x_o) & (x_array <= x_o + submember.L())

            # Not the last submember
            else:

                # Find any points from `x_array` that lie along this submember
                filter = (x_array >= x_o) & (x_array < x_o + submember.L())

            x_subm_array = x_array[filter] - x_o

            # Check which axis is of interest
            if Direction == 'Fz':
                v_array = self._extract_vector_results(submember.SegmentsY, x_subm_array, 'shear')
            elif Direction == 'Fy':
                v_array = self._extract_vector_results(submember.SegmentsZ, x_subm_array, 'shear')
            else:
                raise ValueError(f"Direction must be 'Fy' or 'Fz'. {Direction} was given.")

            # Adjust from the submember's coordinate system to the physical member's coordinate system
            v_array[0] = [x_o + x for x in v_array[0]]

            # Add the submember shear values to the overall member shear values in `v_array2`
            if i != 0:
                v_array2 = hstack((v_array2, v_array))
            else:
                v_array2 = v_array

            # Get the starting position of the next submember
            x_o += submember.L()

        # Return the results
        return v_array2

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

        # Import 'pyplot' if not already done
        if PhysMember.__plt is None:
            from matplotlib import pyplot as plt
            PhysMember.__plt = plt

        fig, ax = PhysMember.__plt.subplots()
        ax.axhline(0, color='black', lw=1)
        ax.grid()

        # Generate the moment diagram
        M_array = self.moment_array(Direction, n_points, combo_name)
        x = M_array[0]
        M = M_array[1]

        PhysMember.__plt.plot(x, M)
        PhysMember.__plt.ylabel('Moment')
        PhysMember.__plt.xlabel('Location')
        PhysMember.__plt.title('Member ' + self.name + '\n' + combo_name)
        PhysMember.__plt.show()

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

        # `m_array2` will be used to store the moment values for the overall member
        m_array2 = empty((2, 1))

        # Create an array of locations along the physical member to obtain results at
        L = self.L()
        if x_array is None:
            x_array = linspace(0, L, n_points)
        else:
            if any(x_array < 0) or any(x_array > L):
                raise ValueError(f"All x values must be in the range 0 to {L}")

        # Step through each submember in the physical member
        x_o = 0
        for i, submember in enumerate(self.sub_members.values()):

            # Segment the submember into segments with mathematically continuous loads if not already done
            if submember._solved_combo is None or combo_name != submember._solved_combo.name:
                submember._segment_member(combo_name)
                submember._solved_combo = self.model.load_combos[combo_name]

            # Check if this is the last submember
            if i == len(self.sub_members.values()) - 1:

                # Find any points from `x_array` that lie along this submember
                filter = (x_array >= x_o) & (x_array <= x_o + submember.L())

            # Not the last submember
            else:

                # Find any points from `x_array` that lie along this submember
                filter = (x_array >= x_o) & (x_array < x_o + submember.L())

            x_subm_array = x_array[filter] - x_o

            # Check which axis is of interest
            if Direction == 'My':
                m_array = self._extract_vector_results(submember.SegmentsY, x_subm_array, 'moment')
            elif Direction == 'Mz':
                m_array = self._extract_vector_results(submember.SegmentsZ, x_subm_array, 'moment')
            else:
                raise ValueError(f"Direction must be 'My' or 'Mz'. {Direction} was given.")

            # Adjust from the submember's coordinate system to the physical member's coordinate system
            m_array[0] = [x_o + x for x in m_array[0]]

            # Add the submember moment values to the overall member shear values in `m_array2`
            if i != 0:
                m_array2 = hstack((m_array2, m_array))
            else:
                m_array2 = m_array

            # Get the starting position of the next submember
            x_o += submember.L()

        # Return the results
        return m_array2
    
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

        # Import 'pyplot' if not already done
        if PhysMember.__plt is None:
            from matplotlib import pyplot as plt
            PhysMember.__plt = plt

        fig, ax = PhysMember.__plt.subplots()
        ax.axhline(0, color='black', lw=1)
        ax.grid()

        # Generate the torque diagram
        T_array = self.torque_array(n_points, combo_name)
        x = T_array[0]
        T = T_array[1]

        PhysMember.__plt.plot(x, T)
        PhysMember.__plt.ylabel('Torque')
        PhysMember.__plt.xlabel('Location')
        PhysMember.__plt.title('Member ' + self.name + '\n' + combo_name)
        PhysMember.__plt.show()

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

        # `t_array2` will be used to store the torque values for the overall member
        t_array2 = empty((2, 1))

        # Create an array of locations along the physical member to obtain results at
        L = self.L()
        if x_array is None:
            x_array = linspace(0, L, n_points)
        else:
            if any(x_array < 0) or any(x_array > L):
                raise ValueError(f"All x values must be in the range 0 to {L}")

        # Step through each submember in the physical member
        x_o = 0
        for i, submember in enumerate(self.sub_members.values()):

            # Segment the submember into segments with mathematically continuous loads if not already done
            if submember._solved_combo is None or combo_name != submember._solved_combo.name:
                submember._segment_member(combo_name)
                submember._solved_combo = self.model.load_combos[combo_name]

            # Check if this is the last submember
            if i == len(self.sub_members.values()) - 1:

                # Find any points from `x_array` that lie along this submember
                filter = (x_array >= x_o) & (x_array <= x_o + submember.L())

            # Not the last submember
            else:

                # Find any points from `x_array` that lie along this submember
                # x_subm_array = [x - x_o for x in x_array if x >= x_o and x < x_o + submember.L()]
                filter = (x_array >= x_o) & (x_array < x_o + submember.L())

            x_subm_array = x_array[filter] - x_o

            # Check which axis is of interest
            t_array = self._extract_vector_results(submember.SegmentsZ, x_subm_array, 'torque')

            # Adjust from the submember's coordinate system to the physical member's coordinate system
            t_array[0] = [x_o + x for x in t_array[0]]

            # Add the submember torque values to the overall member shear values in `t_array2`
            if i != 0:
                t_array2 = hstack((t_array2, t_array))
            else:
                t_array2 = t_array

            # Get the starting position of the next submember
            x_o += submember.L()

        # Return the results
        return t_array2

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

        # Import 'pyplot' if not already done
        if PhysMember.__plt is None:
            from matplotlib import pyplot as plt
            PhysMember.__plt = plt

        fig, ax = PhysMember.__plt.subplots()
        ax.axhline(0, color='black', lw=1)
        ax.grid()

        # Generate the axial force array
        P_array = self.axial_array(n_points, combo_name)
        x = P_array[0]
        P = P_array[1]

        PhysMember.__plt.plot(x, P)
        PhysMember.__plt.ylabel('Axial Force')
        PhysMember.__plt.xlabel('Location')
        PhysMember.__plt.title('Member ' + self.name + '\n' + combo_name)
        PhysMember.__plt.show()

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

        # `a_array2` will be used to store the axial force values for the overall member
        a_array2 = empty((2, 1))

        # Create an array of locations along the physical member to obtain results at
        L = self.L()
        if x_array is None:
            x_array = linspace(0, L, n_points)
        else:
            if any(x_array < 0) or any(x_array > L):
                raise ValueError(f"All x values must be in the range 0 to {L}")

        # Step through each submember in the physical member
        x_o = 0
        for i, submember in enumerate(self.sub_members.values()):

            # Segment the submember into segments with mathematically continuous loads if not already done
            if submember._solved_combo is None or combo_name != submember._solved_combo.name:
                submember._segment_member(combo_name)
                submember._solved_combo = self.model.load_combos[combo_name]

            # Check if this is the last submember
            if i == len(self.sub_members.values()) - 1:

                # Find any points from `x_array` that lie along this submember
                filter = (x_array >= x_o) & (x_array <= x_o + submember.L())

            # Not the last submember
            else:

                # Find any points from `x_array` that lie along this submember
                filter = (x_array >= x_o) & (x_array < x_o + submember.L())

            x_subm_array = x_array[filter] - x_o

            # Check which axis is of interest
            a_array = self._extract_vector_results(submember.SegmentsZ, x_subm_array, 'axial')

            # Adjust from the submember's coordinate system to the physical member's coordinate system
            a_array[0] = [x_o + x for x in a_array[0]]

            # Add the submember axial values to the overall member shear values in `a_array2`
            if i != 0:
                a_array2 = hstack((a_array2, a_array))
            else:
                a_array2 = a_array

            # Get the starting position of the next submember
            x_o += submember.L()
        
        # Return the results
        return a_array2
    
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

        # Import 'pyplot' if not already done
        if PhysMember.__plt is None:
            from matplotlib import pyplot as plt
            PhysMember.__plt = plt

        fig, ax = PhysMember.__plt.subplots()
        ax.axhline(0, color='black', lw=1)
        ax.grid()

        d_array = self.deflection_array(Direction, n_points, combo_name)
        x = d_array[0]
        d = d_array[1]

        PhysMember.__plt.plot(x, d)
        PhysMember.__plt.ylabel('Deflection')
        PhysMember.__plt.xlabel('Location')
        PhysMember.__plt.title('Member ' + self.name + '\n' + combo_name)
        PhysMember.__plt.show()

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

        # `d_array2` will be used to store the deflection values for the overall member
        d_array2 = empty((2, 1))

        # Create an array of locations along the physical member to obtain results at
        L = self.L()
        if x_array is None:
            # Create an array of evenly spaced points
            x_array = linspace(0, L, n_points)
        else:
            # Ensure the requested points are within the member
            if any(x_array < 0) or any(x_array > L):
                raise ValueError(f"All x values must be in the range 0 to {L}")

        # Step through each submember in the physical member
        x_o = 0
        for i, submember in enumerate(self.sub_members.values()):

            # Segment the submember into segments with mathematically continuous loads if not already done
            if submember._solved_combo is None or combo_name != submember._solved_combo.name:
                submember._segment_member(combo_name)
                submember._solved_combo = self.model.load_combos[combo_name]

            # Check if this is the last submember
            if i == len(self.sub_members.values()) - 1:

                # Find any points from `x_array` that lie along this submember
                filter = (x_array >= x_o) & (x_array <= x_o + submember.L())

            # Not the last submember
            else:

                # Find any points from `x_array` that lie along this submember
                filter = (x_array >= x_o) & (x_array < x_o + submember.L())

            x_subm_array = x_array[filter] - x_o

            # Check which axis is of interest
            if Direction == 'dx':
                d_array = self._extract_vector_results(submember.SegmentsX, x_subm_array, 'axial_deflection')
            elif Direction == 'dy':
                d_array = self._extract_vector_results(submember.SegmentsZ, x_subm_array, 'deflection')
            elif Direction == 'dz':
                d_array = self._extract_vector_results(submember.SegmentsY, x_subm_array, 'deflection')
            else:
                raise ValueError(f"Direction must be 'dy' or 'dz'. {Direction} was given.")

            # Adjust from the submember's coordinate system to the physical member's coordinate system
            d_array[0] = [x_o + x for x in d_array[0]]

            # Add the submember deflection values to the overall member shear values in `d_array2`
            if i != 0:
                d_array2 = hstack((d_array2, d_array))
            else:
                d_array2 = d_array

            # Get the starting position of the next submember
            x_o += submember.L()

        # Return the results
        return d_array2

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
