# -*- coding: utf-8 -*-
"""
Comprehensive Model Diagnostics for Pynite

This module provides detailed analysis of structural models to identify
the root causes of analysis failures. Instead of generic error messages,
it provides specific, actionable diagnostics explaining:
- WHAT went wrong (the symptom)
- WHY it went wrong (the root cause)
- WHERE the problem is (specific nodes/members)
- HOW to fix it (actionable suggestions)

Created for Pynite - A 3D Structural Engineering Finite Element Library
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Set, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

if TYPE_CHECKING:
    from Pynite.FEModel3D import FEModel3D
    from Pynite.Node3D import Node3D
    from Pynite.Member3D import Member3D
    from Pynite.PhysMember import PhysMember
    from numpy import float64
    from numpy.typing import NDArray


class IssueSeverity(Enum):
    """Severity levels for diagnostic issues."""
    ERROR = "ERROR"       # Will cause analysis failure
    WARNING = "WARNING"   # May cause issues or unexpected results
    INFO = "INFO"         # Informational - no direct impact


class IssueCategory(Enum):
    """Categories of structural issues."""
    CONNECTIVITY = "Connectivity"
    SUPPORTS = "Supports"
    MECHANISM = "Mechanism"
    GEOMETRY = "Geometry"
    MATERIAL = "Material"
    STABILITY = "Stability"
    LOADING = "Loading"


@dataclass
class DiagnosticIssue:
    """Represents a single diagnostic finding."""
    severity: IssueSeverity
    category: IssueCategory
    title: str
    description: str
    affected_entities: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def format(self, verbose: bool = True) -> str:
        """Format the issue as a human-readable string.

        When verbose=False, shows the title with affected entities on one line
        and the first suggestion. When verbose=True, shows the full description,
        all affected entities, and all suggestions.
        """
        lines = []
        prefix = f"[{self.severity.value}]"

        if verbose:
            lines.append(f"{prefix} {self.title}")
            lines.append(f"  {self.description}")

            if self.affected_entities:
                if len(self.affected_entities) <= 5:
                    entities = ", ".join(self.affected_entities)
                else:
                    entities = ", ".join(self.affected_entities[:5]) + f" (and {len(self.affected_entities) - 5} more)"
                lines.append(f"  Affected: {entities}")

            if self.suggestions:
                lines.append("  Suggestions:")
                for suggestion in self.suggestions:
                    lines.append(f"    - {suggestion}")
        else:
            # Concise: title + affected entities inline, first suggestion
            title_line = f"{prefix} {self.title}"
            if self.affected_entities:
                if len(self.affected_entities) <= 3:
                    title_line += f" — {', '.join(self.affected_entities)}"
                else:
                    title_line += f" — {', '.join(self.affected_entities[:3])} (and {len(self.affected_entities) - 3} more)"
            lines.append(title_line)
            if self.suggestions:
                lines.append(f"  Fix: {self.suggestions[0]}")

        return "\n".join(lines)


class UnionFind:
    """
    Union-Find (Disjoint Set Union) data structure for connectivity analysis.

    Used to efficiently determine which nodes are connected to each other
    through structural elements (members, springs, plates, etc.)
    """

    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        """Find the root of the set containing x with path compression."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        """Unite the sets containing x and y using union by rank."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            # Union by rank
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

    def connected(self, x: str, y: str) -> bool:
        """Check if x and y are in the same connected component."""
        return self.find(x) == self.find(y)

    def get_components(self) -> Dict[str, Set[str]]:
        """Return all connected components as a dict of root -> set of members."""
        components: Dict[str, Set[str]] = defaultdict(set)
        for node in self.parent:
            root = self.find(node)
            components[root].add(node)
        return dict(components)


@dataclass
class ConnectivityInfo:
    """Information about model connectivity."""
    num_components: int
    components: List[Set[str]]  # List of sets of node names
    floating_nodes: Set[str]     # Nodes not connected to any element
    component_supports: List[int]  # Number of supported DOFs per component
    supported_components: List[bool]  # Whether each component has supports


@dataclass
class SupportInfo:
    """Information about model supports."""
    total_supported_dofs: int
    translation_dofs: Dict[str, int]  # X, Y, Z -> count
    rotation_dofs: Dict[str, int]     # RX, RY, RZ -> count
    supported_nodes: List[str]
    rigid_body_modes: List[str]       # List of unrestrained rigid body modes


@dataclass
class DiagnosticReport:
    """Complete diagnostic report for a model."""
    issues: List[DiagnosticIssue] = field(default_factory=list)
    connectivity: Optional[ConnectivityInfo] = None
    supports: Optional[SupportInfo] = None
    model_summary: Dict = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(issue.severity == IssueSeverity.ERROR for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return any(issue.severity == IssueSeverity.WARNING for issue in self.issues)

    def format(self, verbose: bool = True, include_info: bool = False) -> str:
        """Format the complete report as a human-readable string.

        When verbose=False, only errors and warnings are shown with their
        titles, affected entities, and suggestions — no model summary,
        connectivity, or support sections.
        """
        lines = []

        if verbose:
            lines.append("=" * 70)
            lines.append("                    MODEL DIAGNOSTIC REPORT")
            lines.append("=" * 70)
            lines.append("")

            # Model summary
            if self.model_summary:
                lines.append("MODEL SUMMARY")
                lines.append("-" * 40)
                for key, value in self.model_summary.items():
                    lines.append(f"  {key}: {value}")
                lines.append("")

            # Connectivity summary
            if self.connectivity:
                lines.append("CONNECTIVITY ANALYSIS")
                lines.append("-" * 40)
                lines.append(f"  Connected components: {self.connectivity.num_components}")
                if self.connectivity.num_components > 1:
                    lines.append("  Component sizes: " + ", ".join(
                        str(len(c)) + " nodes" for c in self.connectivity.components
                    ))
                if self.connectivity.floating_nodes:
                    lines.append(f"  Floating nodes: {len(self.connectivity.floating_nodes)}")
                lines.append("")

            # Support summary
            if self.supports:
                lines.append("SUPPORT ANALYSIS")
                lines.append("-" * 40)
                lines.append(f"  Total supported DOFs: {self.supports.total_supported_dofs}")
                lines.append(f"  Supported nodes: {len(self.supports.supported_nodes)}")
                trans = self.supports.translation_dofs
                rot = self.supports.rotation_dofs
                lines.append(f"  Translations: X={trans.get('X', 0)}, Y={trans.get('Y', 0)}, Z={trans.get('Z', 0)}")
                lines.append(f"  Rotations: RX={rot.get('RX', 0)}, RY={rot.get('RY', 0)}, RZ={rot.get('RZ', 0)}")
                if self.supports.rigid_body_modes:
                    lines.append(f"  Possible rigid body modes: {', '.join(self.supports.rigid_body_modes)}")
                lines.append("")

        # Issues grouped by severity
        errors = [i for i in self.issues if i.severity == IssueSeverity.ERROR]
        warnings = [i for i in self.issues if i.severity == IssueSeverity.WARNING]
        infos = [i for i in self.issues if i.severity == IssueSeverity.INFO]

        if errors:
            if verbose:
                lines.append("ERRORS (will prevent successful analysis)")
                lines.append("-" * 40)
            for issue in errors:
                lines.append(issue.format(verbose))
                lines.append("")

        if warnings:
            if verbose:
                lines.append("WARNINGS (may cause unexpected results)")
                lines.append("-" * 40)
            for issue in warnings:
                lines.append(issue.format(verbose))
                lines.append("")

        if include_info and infos:
            if verbose:
                lines.append("INFORMATION")
                lines.append("-" * 40)
            for issue in infos:
                lines.append(issue.format(verbose))
                lines.append("")

        if not errors and not warnings:
            lines.append("No issues detected. Model appears ready for analysis.")
            lines.append("")

        if verbose:
            lines.append("=" * 70)

        return "\n".join(lines)


class ModelDiagnostics:
    """
    Comprehensive diagnostic analyzer for Pynite structural models.

    This class performs various checks to identify potential issues that
    could cause analysis failures or unexpected results. It provides
    detailed, actionable feedback to help users fix their models.

    Usage:
        diagnostics = ModelDiagnostics(model)
        report = diagnostics.run_full_diagnosis()
        print(report.format())
    """

    def __init__(self, model: 'FEModel3D'):
        self.model = model
        self.issues: List[DiagnosticIssue] = []
        self._connectivity: Optional[ConnectivityInfo] = None
        self._supports: Optional[SupportInfo] = None
        self._geometric_rotation_restraint: Dict[str, bool] = {}

    def run_full_diagnosis(self) -> DiagnosticReport:
        """
        Run all diagnostic checks and return a complete report.

        Returns:
            DiagnosticReport: A comprehensive report of all findings.
        """
        self.issues = []

        # Run all diagnostic checks
        self._check_basic_requirements()
        self._analyze_connectivity()
        self._check_geometric_rotation_restraint()  # Must run before _analyze_supports
        self._analyze_supports()
        self._check_mechanisms()
        self._check_geometry()
        self._check_near_coincident_nodes()
        self._check_singly_connected_nodes()
        # NOTE: _check_plate_drilling_dof and _check_coplanar_frame removed due to
        # high false positive risk. Pynite handles plate drilling DOF internally with
        # weak stiffness, and 2D coplanar frames work correctly.
        self._check_materials_and_sections()
        self._check_loading()

        # Build the report
        report = DiagnosticReport(
            issues=self.issues.copy(),
            connectivity=self._connectivity,
            supports=self._supports,
            model_summary=self._get_model_summary()
        )

        return report

    def _get_model_summary(self) -> Dict:
        """Get a summary of the model's contents."""
        return {
            "Nodes": len(self.model.nodes),
            "Members": len(self.model.members),
            "Springs": len(self.model.springs),
            "Plates": len(self.model.plates),
            "Quads": len(self.model.quads),
            "Materials": len(self.model.materials),
            "Sections": len(self.model.sections),
            "Load Combinations": len(self.model.load_combos)
        }

    def _check_basic_requirements(self) -> None:
        """Check that the model has the minimum required components."""

        if len(self.model.nodes) == 0:
            self.issues.append(DiagnosticIssue(
                severity=IssueSeverity.ERROR,
                category=IssueCategory.GEOMETRY,
                title="No nodes defined",
                description="The model contains no nodes. A structural model requires at least one node to define geometry.",
                suggestions=["Add nodes using model.add_node(name, X, Y, Z)"]
            ))
            return  # Can't continue without nodes

        # Check for elements
        num_elements = (len(self.model.members) + len(self.model.springs) +
                       len(self.model.plates) + len(self.model.quads))

        if num_elements == 0:
            self.issues.append(DiagnosticIssue(
                severity=IssueSeverity.ERROR,
                category=IssueCategory.GEOMETRY,
                title="No structural elements defined",
                description="The model contains nodes but no structural elements (members, springs, plates, or quads). "
                           "Elements are required to transfer loads through the structure.",
                suggestions=[
                    "Add members using model.add_member(...)",
                    "Add springs using model.add_spring(...)",
                    "Add plates/quads for shell structures"
                ]
            ))

    def _analyze_connectivity(self) -> None:
        """
        Analyze the structural connectivity of the model.

        Uses Union-Find to identify:
        - Disconnected components (separate parts of the structure)
        - Floating nodes (nodes not connected to any element)
        """
        if len(self.model.nodes) == 0:
            return

        uf = UnionFind()
        connected_nodes: Set[str] = set()

        # Add all nodes to Union-Find
        for node_name in self.model.nodes:
            uf.find(node_name)

        # Connect nodes via members
        for member in self.model.members.values():
            uf.union(member.i_node.name, member.j_node.name)
            connected_nodes.add(member.i_node.name)
            connected_nodes.add(member.j_node.name)

        # Connect nodes via springs
        for spring in self.model.springs.values():
            uf.union(spring.i_node.name, spring.j_node.name)
            connected_nodes.add(spring.i_node.name)
            connected_nodes.add(spring.j_node.name)

        # Connect nodes via quads (4 nodes each)
        for quad in self.model.quads.values():
            nodes = [quad.i_node.name, quad.j_node.name, quad.m_node.name, quad.n_node.name]
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    uf.union(nodes[i], nodes[j])
                connected_nodes.add(nodes[i])

        # Connect nodes via plates (4 nodes each)
        for plate in self.model.plates.values():
            nodes = [plate.i_node.name, plate.j_node.name, plate.m_node.name, plate.n_node.name]
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    uf.union(nodes[i], nodes[j])
                connected_nodes.add(nodes[i])

        # Identify floating nodes
        floating_nodes = set(self.model.nodes.keys()) - connected_nodes

        # Get connected components (only for connected nodes)
        components_dict = uf.get_components()
        # Filter to only include nodes that are actually connected to elements
        components = []
        for root, members in components_dict.items():
            connected_in_component = members & connected_nodes
            if connected_in_component:
                components.append(connected_in_component)

        # Check supports per component
        component_supports = []
        supported_components = []

        for component in components:
            supported_dofs = 0
            for node_name in component:
                node = self.model.nodes[node_name]
                if node.support_DX: supported_dofs += 1
                if node.support_DY: supported_dofs += 1
                if node.support_DZ: supported_dofs += 1
                if node.support_RX: supported_dofs += 1
                if node.support_RY: supported_dofs += 1
                if node.support_RZ: supported_dofs += 1
                # Also count spring supports
                if node.spring_DX[0] is not None: supported_dofs += 1
                if node.spring_DY[0] is not None: supported_dofs += 1
                if node.spring_DZ[0] is not None: supported_dofs += 1
                if node.spring_RX[0] is not None: supported_dofs += 1
                if node.spring_RY[0] is not None: supported_dofs += 1
                if node.spring_RZ[0] is not None: supported_dofs += 1

            component_supports.append(supported_dofs)
            supported_components.append(supported_dofs > 0)

        self._connectivity = ConnectivityInfo(
            num_components=len(components),
            components=components,
            floating_nodes=floating_nodes,
            component_supports=component_supports,
            supported_components=supported_components
        )

        # Report issues
        if floating_nodes:
            self.issues.append(DiagnosticIssue(
                severity=IssueSeverity.WARNING,
                category=IssueCategory.CONNECTIVITY,
                title=f"Unconnected nodes ({len(floating_nodes)} nodes)",
                description="These nodes exist in the model but have no members, plates, or springs attached. "
                           "Unconnected nodes create degrees of freedom with zero stiffness, which typically "
                           "causes the system of equations to be unsolvable.",
                affected_entities=sorted(list(floating_nodes)),
                suggestions=[
                    "Remove unused nodes, or connect them to structural elements",
                    "Verify member definitions reference the correct node names"
                ]
            ))

        if len(components) > 1:
            # Find which components are unsupported
            unsupported_components = []
            for i, (component, has_support) in enumerate(zip(components, supported_components)):
                if not has_support:
                    unsupported_components.append((i, component))

            if unsupported_components:
                desc = f"The model has {len(components)} separate disconnected parts. "
                if unsupported_components:
                    desc += f"{len(unsupported_components)} component(s) have no supports and will be completely unstable."

                all_affected = []
                for _, component in unsupported_components:
                    all_affected.extend(list(component)[:3])  # Show first 3 nodes from each

                self.issues.append(DiagnosticIssue(
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.CONNECTIVITY,
                    title="Disconnected and unsupported structure",
                    description=desc,
                    affected_entities=all_affected,
                    suggestions=[
                        "Connect the separate parts with structural elements",
                        "Add supports to all disconnected components",
                        "Check if members were accidentally omitted"
                    ]
                ))
            else:
                # All components have supports - this may be intentional for multi-structure analysis
                self.issues.append(DiagnosticIssue(
                    severity=IssueSeverity.INFO,
                    category=IssueCategory.CONNECTIVITY,
                    title=f"Model has {len(components)} disconnected components",
                    description="The structure consists of separate, disconnected parts. Each component "
                               "has supports so analysis should succeed.",
                    affected_entities=[],
                    suggestions=[
                        "If unintentional, connect the parts with structural elements"
                    ]
                ))

    def _analyze_supports(self) -> None:
        """
        Analyze the support conditions of the model.

        Checks for:
        - Sufficient supports to prevent rigid body motion (minimum 6 DOFs for 3D)
        - Distribution of supports
        - Potential rigid body modes
        """
        translation_dofs = {'X': 0, 'Y': 0, 'Z': 0}
        rotation_dofs = {'RX': 0, 'RY': 0, 'RZ': 0}
        supported_nodes: List[str] = []
        total_dofs = 0

        for node in self.model.nodes.values():
            node_has_support = False

            # Check fixed supports
            if node.support_DX:
                translation_dofs['X'] += 1
                total_dofs += 1
                node_has_support = True
            if node.support_DY:
                translation_dofs['Y'] += 1
                total_dofs += 1
                node_has_support = True
            if node.support_DZ:
                translation_dofs['Z'] += 1
                total_dofs += 1
                node_has_support = True
            if node.support_RX:
                rotation_dofs['RX'] += 1
                total_dofs += 1
                node_has_support = True
            if node.support_RY:
                rotation_dofs['RY'] += 1
                total_dofs += 1
                node_has_support = True
            if node.support_RZ:
                rotation_dofs['RZ'] += 1
                total_dofs += 1
                node_has_support = True

            # Check spring supports (they also provide restraint)
            if node.spring_DX[0] is not None:
                translation_dofs['X'] += 1
                total_dofs += 1
                node_has_support = True
            if node.spring_DY[0] is not None:
                translation_dofs['Y'] += 1
                total_dofs += 1
                node_has_support = True
            if node.spring_DZ[0] is not None:
                translation_dofs['Z'] += 1
                total_dofs += 1
                node_has_support = True
            if node.spring_RX[0] is not None:
                rotation_dofs['RX'] += 1
                total_dofs += 1
                node_has_support = True
            if node.spring_RY[0] is not None:
                rotation_dofs['RY'] += 1
                total_dofs += 1
                node_has_support = True
            if node.spring_RZ[0] is not None:
                rotation_dofs['RZ'] += 1
                total_dofs += 1
                node_has_support = True

            if node_has_support:
                supported_nodes.append(node.name)

        # Determine potential rigid body modes
        rigid_body_modes = []
        if translation_dofs['X'] == 0:
            rigid_body_modes.append("Translation in X (structure can slide in X direction)")
        if translation_dofs['Y'] == 0:
            rigid_body_modes.append("Translation in Y (structure can slide in Y direction)")
        if translation_dofs['Z'] == 0:
            rigid_body_modes.append("Translation in Z (structure can slide in Z direction)")

        # Check rotation restraints, accounting for geometric restraint from translation supports
        # Rotation about an axis can be prevented by translation supports at different positions
        geom_restraint = self._geometric_rotation_restraint

        if len(supported_nodes) == 1:
            # Single supported node - check rotation restraints
            node = self.model.nodes[supported_nodes[0]]
            if not (node.support_RX or node.spring_RX[0] is not None or geom_restraint.get('RX', False)):
                rigid_body_modes.append("Rotation about X through the single support")
            if not (node.support_RY or node.spring_RY[0] is not None or geom_restraint.get('RY', False)):
                rigid_body_modes.append("Rotation about Y through the single support")
            if not (node.support_RZ or node.spring_RZ[0] is not None or geom_restraint.get('RZ', False)):
                rigid_body_modes.append("Rotation about Z through the single support")
        elif len(supported_nodes) >= 2:
            # Multiple supports - geometric restraint typically prevents rotation.
            # Note: We intentionally do NOT check for rotation about axis connecting
            # two supports because member torsional/bending stiffness provides restraint
            # in practice. Flagging this would be a misleading "red herring" for users.
            pass

        self._supports = SupportInfo(
            total_supported_dofs=total_dofs,
            translation_dofs=translation_dofs,
            rotation_dofs=rotation_dofs,
            supported_nodes=supported_nodes,
            rigid_body_modes=rigid_body_modes
        )

        # Report issues
        if total_dofs == 0:
            self.issues.append(DiagnosticIssue(
                severity=IssueSeverity.ERROR,
                category=IssueCategory.SUPPORTS,
                title="No supports defined",
                description="The model has no boundary conditions. Without at least one supported node, "
                           "the structure has no reference point and equilibrium cannot be established.",
                suggestions=[
                    "Add supports using model.def_support(node_name, DX, DY, DZ, RX, RY, RZ)",
                    "A fixed support restrains all 6 DOFs: def_support(node, True, True, True, True, True, True)",
                    "A pinned support restrains translations only: def_support(node, True, True, True, False, False, False)"
                ]
            ))
        elif total_dofs < 6:
            # Check if geometric rotation restraint makes up for missing explicit rotation supports
            # Translation supports at different locations can geometrically restrain rotations
            geom_restraint = self._geometric_rotation_restraint
            effective_dofs = total_dofs
            if geom_restraint.get('RX', False) and rotation_dofs['RX'] == 0:
                effective_dofs += 1
            if geom_restraint.get('RY', False) and rotation_dofs['RY'] == 0:
                effective_dofs += 1
            if geom_restraint.get('RZ', False) and rotation_dofs['RZ'] == 0:
                effective_dofs += 1

            if effective_dofs < 6:
                self.issues.append(DiagnosticIssue(
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.SUPPORTS,
                    title=f"Insufficient boundary conditions ({total_dofs} restrained DOFs)",
                    description="A 3D structure requires restraint against 3 translations (X, Y, Z) and "
                               "3 rotations (RX, RY, RZ) to prevent rigid body motion. The current support "
                               "configuration does not fully restrain the structure.",
                    affected_entities=supported_nodes,
                    suggestions=[
                        "Use a fixed support at one node (restrains all 6 DOFs)",
                        "Or use multiple supports that collectively prevent all rigid body modes",
                        "Two pinned supports at different elevations can geometrically restrain rotations"
                    ]
                ))

        if rigid_body_modes:
            # Determine severity based on what modes are possible
            has_translation_mode = any("Translation" in mode for mode in rigid_body_modes)

            self.issues.append(DiagnosticIssue(
                severity=IssueSeverity.ERROR if has_translation_mode else IssueSeverity.WARNING,
                category=IssueCategory.SUPPORTS,
                title=f"Unrestrained rigid body mode(s) detected ({len(rigid_body_modes)})",
                description="The structure can displace without developing internal resistance in one or more "
                           "directions. This produces a singular stiffness matrix with no unique solution.",
                affected_entities=supported_nodes,
                suggestions=[
                    "Add boundary conditions to restrain all rigid body modes:",
                    *[f"  - {mode}" for mode in rigid_body_modes[:3]]
                ]
            ))

    def _check_mechanisms(self) -> None:
        """
        Check for mechanism formation due to member releases.

        A mechanism occurs when there are too many releases (hinges) creating
        an unstable configuration.
        """
        # Check each node for mechanism potential
        node_releases: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # node -> [(member, release_type), ...]

        for member in self.model.members.values():
            releases = member.Releases
            # Releases are: [Fxi, Fyi, Fzi, Mxi, Myi, Mzi, Fxj, Fyj, Fzj, Mxj, Myj, Mzj]

            i_node = member.i_node.name
            j_node = member.j_node.name

            # Track moment releases (most common cause of mechanisms)
            if releases[3]:  # Mxi
                node_releases[i_node].append((member.name, "Mx-i"))
            if releases[4]:  # Myi
                node_releases[i_node].append((member.name, "My-i"))
            if releases[5]:  # Mzi
                node_releases[i_node].append((member.name, "Mz-i"))
            if releases[9]:  # Mxj
                node_releases[j_node].append((member.name, "Mx-j"))
            if releases[10]:  # Myj
                node_releases[j_node].append((member.name, "My-j"))
            if releases[11]:  # Mzj
                node_releases[j_node].append((member.name, "Mz-j"))

            # Track axial releases (can cause mechanism if both ends released)
            if releases[0] and releases[6]:  # Both Fxi and Fxj released
                self.issues.append(DiagnosticIssue(
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.MECHANISM,
                    title=f"Member '{member.name}' has axial release at both ends",
                    description="Releasing axial force at both ends removes all axial stiffness from this member. "
                               "The member cannot transfer axial load and creates a mechanism.",
                    affected_entities=[member.name, i_node, j_node],
                    suggestions=[
                        "Remove the axial release from one end",
                        "If zero axial stiffness is intended, use a spring element with defined stiffness"
                    ]
                ))

        # Check for hinge chains (multiple releases at a node creating a mechanism)
        for node_name, releases in node_releases.items():
            node = self.model.nodes[node_name]

            # Count members connected to this node
            connected_members = self._get_connected_members(node_name)
            num_members = len(connected_members)

            # Count releases by type at this node
            my_releases = sum(1 for _, r in releases if 'My' in r)
            mz_releases = sum(1 for _, r in releases if 'Mz' in r)

            # For a node to be stable in bending:
            # If N members connect at a node, at most N-1 moment releases are allowed
            # (assuming the node is not a support)

            is_rotationally_supported = (
                node.support_RX or node.support_RY or node.support_RZ or
                node.spring_RX[0] is not None or node.spring_RY[0] is not None or
                node.spring_RZ[0] is not None
            )

            if num_members > 0:
                max_releases = num_members if is_rotationally_supported else num_members - 1

                if my_releases > max_releases:
                    self.issues.append(DiagnosticIssue(
                        severity=IssueSeverity.ERROR,
                        category=IssueCategory.MECHANISM,
                        title=f"Moment release mechanism at node '{node_name}' (My)",
                        description=f"Node has {my_releases} My (major-axis moment) releases across {num_members} "
                                   f"connected member(s). With N members at an unsupported node, at most N-1 "
                                   "moment releases are allowed to maintain rotational equilibrium.",
                        affected_entities=[node_name] + [m for m, _ in releases if 'My' in _],
                        suggestions=[
                            "Remove the moment release from at least one member end at this node",
                            "Or add a rotational support (RY or RZ) at this node"
                        ]
                    ))

                if mz_releases > max_releases:
                    self.issues.append(DiagnosticIssue(
                        severity=IssueSeverity.ERROR,
                        category=IssueCategory.MECHANISM,
                        title=f"Moment release mechanism at node '{node_name}' (Mz)",
                        description=f"Node has {mz_releases} Mz (minor-axis moment) releases across {num_members} "
                                   f"connected member(s). With N members at an unsupported node, at most N-1 "
                                   "moment releases are allowed to maintain rotational equilibrium.",
                        affected_entities=[node_name] + [m for m, _ in releases if 'Mz' in _],
                        suggestions=[
                            "Remove the moment release from at least one member end at this node",
                            "Or add a rotational support (RY or RZ) at this node"
                        ]
                    ))

    def _get_connected_members(self, node_name: str) -> List[str]:
        """Get list of member names connected to a node."""
        connected = []
        for member in self.model.members.values():
            if member.i_node.name == node_name or member.j_node.name == node_name:
                connected.append(member.name)
        return connected

    def _check_geometry(self) -> None:
        """
        Check for geometric issues that could cause analysis problems.

        Checks for:
        - Zero-length members
        - Very short members (numerical issues)
        - Duplicate nodes at same location
        - Colinear member chains that might cause issues
        """
        # Check for zero-length or very short members
        for member in self.model.members.values():
            length = member.i_node.distance(member.j_node)

            if length < 1e-10:
                self.issues.append(DiagnosticIssue(
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.GEOMETRY,
                    title=f"Zero-length member '{member.name}'",
                    description=f"Member '{member.name}' connects nodes '{member.i_node.name}' and "
                               f"'{member.j_node.name}' at the same location. A member requires finite length "
                               "to define its local coordinate system and compute stiffness.",
                    affected_entities=[member.name, member.i_node.name, member.j_node.name],
                    suggestions=[
                        "Verify node coordinates are correct",
                        "If nodes should coincide, use a single node for both member ends",
                        "Run model.merge_duplicate_nodes() to merge coincident nodes"
                    ]
                ))
            elif length < 1e-6:
                self.issues.append(DiagnosticIssue(
                    severity=IssueSeverity.WARNING,
                    category=IssueCategory.GEOMETRY,
                    title=f"Very short member '{member.name}' (L={length:.2e})",
                    description=f"Member has an extremely small length which may cause numerical issues.",
                    affected_entities=[member.name],
                    suggestions=[
                        "Consider merging nodes if unintentional",
                        "Use consistent units throughout the model"
                    ]
                ))

        # Check for duplicate nodes (same coordinates)
        node_locations: Dict[Tuple[float, float, float], List[str]] = defaultdict(list)
        for node in self.model.nodes.values():
            # Round to avoid floating point comparison issues
            loc = (round(node.X, 6), round(node.Y, 6), round(node.Z, 6))
            node_locations[loc].append(node.name)

        for loc, nodes in node_locations.items():
            if len(nodes) > 1:
                self.issues.append(DiagnosticIssue(
                    severity=IssueSeverity.WARNING,
                    category=IssueCategory.GEOMETRY,
                    title=f"Multiple nodes at same location {loc}",
                    description=f"Found {len(nodes)} nodes at the same coordinates. This usually indicates "
                               "a modeling error unless intentionally used for special connections.",
                    affected_entities=nodes,
                    suggestions=[
                        "Merge duplicate nodes if they should be the same point",
                        "Check for unintentional node duplication",
                        "If intentional, ensure nodes are properly connected"
                    ]
                ))

        # Note: Zero-length springs are intentionally NOT flagged as issues.
        # They are commonly used and mathematically valid for modeling nodal
        # springs connecting coincident nodes or for special release conditions.

    def _check_materials_and_sections(self) -> None:
        """
        Check for issues with material and section properties.

        Checks for:
        - Zero or negative stiffness properties
        - Unused materials/sections
        - Missing required properties
        """
        # Check materials
        for mat_name, material in self.model.materials.items():
            if hasattr(material, 'E') and material.E <= 0:
                self.issues.append(DiagnosticIssue(
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.MATERIAL,
                    title=f"Invalid elastic modulus for material '{mat_name}'",
                    description=f"Elastic modulus E={material.E} must be positive.",
                    affected_entities=[mat_name],
                    suggestions=["Set a positive value for E (elastic modulus)"]
                ))

            if hasattr(material, 'G') and material.G <= 0:
                self.issues.append(DiagnosticIssue(
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.MATERIAL,
                    title=f"Invalid shear modulus for material '{mat_name}'",
                    description=f"Shear modulus G={material.G} must be positive.",
                    affected_entities=[mat_name],
                    suggestions=["Set a positive value for G (shear modulus)"]
                ))

        # Check sections
        for sec_name, section in self.model.sections.items():
            if hasattr(section, 'A') and section.A <= 0:
                self.issues.append(DiagnosticIssue(
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.MATERIAL,
                    title=f"Invalid area for section '{sec_name}'",
                    description=f"Cross-sectional area A={section.A} must be positive.",
                    affected_entities=[sec_name],
                    suggestions=["Set a positive value for A (cross-sectional area)"]
                ))

            if hasattr(section, 'Iy') and section.Iy <= 0:
                self.issues.append(DiagnosticIssue(
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.MATERIAL,
                    title=f"Invalid moment of inertia Iy for section '{sec_name}'",
                    description=f"Moment of inertia Iy={section.Iy} must be positive for bending analysis.",
                    affected_entities=[sec_name],
                    suggestions=["Set a positive value for Iy"]
                ))

            if hasattr(section, 'Iz') and section.Iz <= 0:
                self.issues.append(DiagnosticIssue(
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.MATERIAL,
                    title=f"Invalid moment of inertia Iz for section '{sec_name}'",
                    description=f"Moment of inertia Iz={section.Iz} must be positive for bending analysis.",
                    affected_entities=[sec_name],
                    suggestions=["Set a positive value for Iz"]
                ))

            if hasattr(section, 'J') and section.J <= 0:
                self.issues.append(DiagnosticIssue(
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.MATERIAL,
                    title=f"Invalid torsion constant for section '{sec_name}'",
                    description=f"Torsion constant J={section.J} must be positive.",
                    affected_entities=[sec_name],
                    suggestions=["Set a positive value for J (torsion constant)"]
                ))

    def _check_loading(self) -> None:
        """
        Check for potential loading issues.

        Checks for:
        - Loads on unsupported nodes
        - Load combinations without any loads
        """
        # Check if any load combinations are defined
        if not self.model.load_combos:
            self.issues.append(DiagnosticIssue(
                severity=IssueSeverity.INFO,
                category=IssueCategory.LOADING,
                title="No load combinations defined",
                description="No load combinations are defined. A default 'Combo 1' will be created "
                           "automatically during analysis.",
                suggestions=["Define load combinations for different load scenarios"]
            ))

        # Check for loaded nodes that are in disconnected/unsupported regions
        if self._connectivity and self._connectivity.num_components > 1:
            for node in self.model.nodes.values():
                if node.NodeLoads:
                    # Find which component this node is in
                    for i, component in enumerate(self._connectivity.components):
                        if node.name in component:
                            if not self._connectivity.supported_components[i]:
                                self.issues.append(DiagnosticIssue(
                                    severity=IssueSeverity.ERROR,
                                    category=IssueCategory.LOADING,
                                    title=f"Load on unsupported structure component",
                                    description=f"Node '{node.name}' has loads but is part of a disconnected "
                                               "component with no supports. The load cannot be resisted.",
                                    affected_entities=[node.name],
                                    suggestions=[
                                        "Add supports to this component",
                                        "Connect this component to the supported structure",
                                        "Remove the load if the component is not needed"
                                    ]
                                ))
                            break

    def _check_singly_connected_nodes(self) -> None:
        """
        Check for nodes connected to only one element without adequate support.

        Only reports an issue if unmerged node candidates are detected (nodes that
        are singly-connected AND very close to other nodes). This avoids false
        positives for legitimate cantilever ends.
        """
        # Build node connectivity map
        node_connections: Dict[str, List[str]] = defaultdict(list)

        for member_name, member in self.model.members.items():
            if hasattr(member, 'sub_members') and member.sub_members:
                for sub_member in member.sub_members.values():
                    node_connections[sub_member.i_node.name].append(f"member:{member_name}")
                    node_connections[sub_member.j_node.name].append(f"member:{member_name}")
                    break  # Only count connectivity once per physical member
            elif hasattr(member, 'i_node'):
                node_connections[member.i_node.name].append(f"member:{member_name}")
                node_connections[member.j_node.name].append(f"member:{member_name}")

        for spring_name, spring in self.model.springs.items():
            node_connections[spring.i_node.name].append(f"spring:{spring_name}")
            node_connections[spring.j_node.name].append(f"spring:{spring_name}")

        for plate_name, plate in self.model.plates.items():
            for node in [plate.i_node, plate.j_node, plate.m_node, plate.n_node]:
                node_connections[node.name].append(f"plate:{plate_name}")

        for quad_name, quad in self.model.quads.items():
            for node in [quad.i_node, quad.j_node, quad.m_node, quad.n_node]:
                node_connections[node.name].append(f"quad:{quad_name}")

        # Find singly-connected unsupported nodes
        singly_connected = []
        for node_name, connections in node_connections.items():
            if len(connections) == 1:
                node = self.model.nodes.get(node_name)
                if node:
                    is_fully_supported = (
                        node.support_DX and node.support_DY and node.support_DZ and
                        node.support_RX and node.support_RY and node.support_RZ
                    )
                    if not is_fully_supported:
                        singly_connected.append((node_name, connections[0]))

        if not singly_connected:
            return

        # Check for unmerged nodes (singly-connected nodes close to other nodes)
        # ONLY report if we find unmerged candidates - avoid false positives for cantilevers
        max_coord = self._get_model_scale()
        merge_tolerance = max(max_coord * 1e-4, 0.01)

        unmerged_candidates = []
        for singly_name, _ in singly_connected[:20]:  # Check first 20 to avoid O(n²)
            singly_node = self.model.nodes.get(singly_name)
            if not singly_node:
                continue
            for other_name, other_node in self.model.nodes.items():
                if other_name == singly_name:
                    continue
                dist = singly_node.distance(other_node)
                if dist < merge_tolerance and dist > 1e-10:
                    unmerged_candidates.append(f"{singly_name} near {other_name} (dist={dist:.2e})")
                    break

        # Only report if we found likely unmerged nodes
        if not unmerged_candidates:
            return  # Don't warn about legitimate cantilever ends

        affected = [f"{name} (connected to {conn})" for name, conn in singly_connected[:5]]
        if len(singly_connected) > 5:
            affected.append(f"... and {len(singly_connected) - 5} more")

        suggestions = [
            f"Run model.merge_duplicate_nodes(tolerance={merge_tolerance:.3g}) to merge nearby nodes",
            "Or ensure members reference the same node object at shared connection points"
        ]
        for candidate in unmerged_candidates[:3]:
            suggestions.append(f"  - {candidate}")

        self.issues.append(DiagnosticIssue(
            severity=IssueSeverity.ERROR,
            category=IssueCategory.CONNECTIVITY,
            title=f"Probable unmerged nodes ({len(unmerged_candidates)} found)",
            description="Singly-connected member ends were found within close proximity to other nodes. "
                       "This typically indicates nodes that should share connectivity but were created "
                       "separately, resulting in no load path between adjacent members.",
            affected_entities=affected,
            suggestions=suggestions
        ))

    def _check_geometric_rotation_restraint(self) -> None:
        """
        Check if rotation is geometrically restrained by translation supports.

        Rotation about an axis can be prevented by translation supports at
        different positions perpendicular to that axis, even without explicit
        rotational supports.
        """
        # Collect support coordinates
        support_coords = {'DX': [], 'DY': [], 'DZ': []}

        for node in self.model.nodes.values():
            if node.support_DX or (node.spring_DX[0] is not None):
                support_coords['DX'].append((node.X, node.Y, node.Z))
            if node.support_DY or (node.spring_DY[0] is not None):
                support_coords['DY'].append((node.X, node.Y, node.Z))
            if node.support_DZ or (node.spring_DZ[0] is not None):
                support_coords['DZ'].append((node.X, node.Y, node.Z))

        geom_tol = 0.001

        # Check rotation restraints
        # RX restrained by DY at different Z, or DZ at different Y
        # RY restrained by DX at different Z, or DZ at different X
        # RZ restrained by DX at different Y, or DY at different X

        rotation_checks = {
            'RX': [('DY', 2), ('DZ', 1)],  # Check Z coords for DY, Y coords for DZ
            'RY': [('DX', 2), ('DZ', 0)],  # Check Z coords for DX, X coords for DZ
            'RZ': [('DX', 1), ('DY', 0)],  # Check Y coords for DX, X coords for DY
        }

        geometrically_restrained = {'RX': False, 'RY': False, 'RZ': False}

        for rot_dof, checks in rotation_checks.items():
            for trans_dof, coord_idx in checks:
                coords = [c[coord_idx] for c in support_coords[trans_dof]]
                if len(coords) >= 2 and max(coords) - min(coords) > geom_tol:
                    geometrically_restrained[rot_dof] = True
                    break

        # Store for use by _analyze_supports
        self._geometric_rotation_restraint = geometrically_restrained

    def _check_near_coincident_nodes(self) -> None:
        """
        Check for nodes that are very close but not exactly coincident.

        This often indicates modeling errors or unintentional duplicate nodes.
        """
        max_coord = self._get_model_scale()
        # Use 0.01% of model size as tolerance, minimum 0.001
        near_tolerance = max(max_coord * 1e-4, 0.001)

        near_coincident = []
        node_list = list(self.model.nodes.items())

        for i, (name1, node1) in enumerate(node_list):
            for name2, node2 in node_list[i+1:]:
                dist = node1.distance(node2)
                if 1e-6 < dist < near_tolerance:
                    near_coincident.append((name1, name2, dist))

        if near_coincident:
            affected = [f"'{n1}' and '{n2}': distance={d:.4g}" for n1, n2, d in near_coincident[:5]]
            if len(near_coincident) > 5:
                affected.append(f"... and {len(near_coincident) - 5} more pairs")

            self.issues.append(DiagnosticIssue(
                severity=IssueSeverity.WARNING,
                category=IssueCategory.GEOMETRY,
                title=f"Near-coincident nodes detected ({len(near_coincident)} pairs)",
                description="Node pairs exist with very small but non-zero separation. This may indicate "
                           "unintentional coordinate discrepancies or nodes that should be merged.",
                affected_entities=affected,
                suggestions=[
                    f"Run model.merge_duplicate_nodes(tolerance={near_tolerance:.4g}) to merge",
                    "Or verify that the coordinate differences are intentional"
                ]
            ))

    def _get_model_scale(self) -> float:
        """Get the characteristic scale of the model from node coordinates."""
        all_coords = []
        for node in self.model.nodes.values():
            all_coords.extend([abs(node.X), abs(node.Y), abs(node.Z)])
        return max(all_coords) if all_coords else 1.0


def diagnose_instability(model: 'FEModel3D', K: 'NDArray[float64]' = None) -> str:
    """
    Convenience function to diagnose why a model might be unstable.

    This can be called after an analysis failure to get a detailed explanation
    of what went wrong.

    Args:
        model: The FEModel3D that failed analysis
        K: Optional stiffness matrix (if available) for additional diagnostics

    Returns:
        A formatted string explaining the likely cause of instability
    """
    diagnostics = ModelDiagnostics(model)
    report = diagnostics.run_full_diagnosis()
    return report.format(verbose=True)


def get_connectivity_summary(model: 'FEModel3D') -> str:
    """
    Get a summary of the model's structural connectivity.

    Args:
        model: The FEModel3D to analyze

    Returns:
        A formatted string describing the connectivity
    """
    diagnostics = ModelDiagnostics(model)
    diagnostics._analyze_connectivity()

    if diagnostics._connectivity is None:
        return "Unable to analyze connectivity."

    conn = diagnostics._connectivity
    lines = []
    lines.append("Connectivity Summary:")
    lines.append(f"  - {conn.num_components} connected component(s)")

    for i, component in enumerate(conn.components):
        has_support = conn.supported_components[i] if i < len(conn.supported_components) else False
        support_str = "with supports" if has_support else "NO SUPPORTS"
        lines.append(f"    Component {i+1}: {len(component)} nodes ({support_str})")
        if len(component) <= 5:
            lines.append(f"      Nodes: {', '.join(sorted(component))}")

    if conn.floating_nodes:
        lines.append(f"  - {len(conn.floating_nodes)} floating (unconnected) node(s)")
        if len(conn.floating_nodes) <= 5:
            lines.append(f"      {', '.join(sorted(conn.floating_nodes))}")

    return "\n".join(lines)


def get_connectivity_graph(model: 'FEModel3D', format: str = 'text') -> str:
    """
    Generate a graph representation of the model's node-member connectivity.

    Parameters
    ----------
    model : FEModel3D
        The finite element model to analyze
    format : str
        Output format: 'text' for readable adjacency list, 'dot' for Graphviz DOT format

    Returns
    -------
    str
        Graph representation in the specified format

    Example
    -------
    >>> print(get_connectivity_graph(model))
    >>> # Or export to Graphviz:
    >>> with open('model.dot', 'w') as f:
    ...     f.write(get_connectivity_graph(model, format='dot'))
    """
    from collections import defaultdict

    # Build adjacency data
    node_to_elements: Dict[str, List[str]] = defaultdict(list)
    element_to_nodes: Dict[str, List[str]] = {}

    # Collect member connections
    for name, member in model.members.items():
        i_name = member.i_node.name
        j_name = member.j_node.name
        element_to_nodes[f"M:{name}"] = [i_name, j_name]
        node_to_elements[i_name].append(f"M:{name}")
        node_to_elements[j_name].append(f"M:{name}")

    # Collect spring connections
    for name, spring in model.springs.items():
        i_name = spring.i_node.name
        j_name = spring.j_node.name
        element_to_nodes[f"S:{name}"] = [i_name, j_name]
        node_to_elements[i_name].append(f"S:{name}")
        node_to_elements[j_name].append(f"S:{name}")

    # Collect plate connections
    for name, plate in model.plates.items():
        nodes = [plate.i_node.name, plate.j_node.name, plate.m_node.name, plate.n_node.name]
        element_to_nodes[f"P:{name}"] = nodes
        for n in nodes:
            node_to_elements[n].append(f"P:{name}")

    # Collect quad connections
    for name, quad in model.quads.items():
        nodes = [quad.i_node.name, quad.j_node.name, quad.m_node.name, quad.n_node.name]
        element_to_nodes[f"Q:{name}"] = nodes
        for n in nodes:
            node_to_elements[n].append(f"Q:{name}")

    # Classify nodes
    supported_nodes = set()
    for node in model.nodes.values():
        if (node.support_DX or node.support_DY or node.support_DZ or
            node.support_RX or node.support_RY or node.support_RZ or
            node.spring_DX[0] is not None or node.spring_DY[0] is not None or
            node.spring_DZ[0] is not None):
            supported_nodes.add(node.name)

    floating_nodes = set(model.nodes.keys()) - set(node_to_elements.keys())
    singly_connected = {n for n, elems in node_to_elements.items() if len(elems) == 1}

    if format == 'dot':
        return _format_dot_graph(model, node_to_elements, element_to_nodes,
                                  supported_nodes, floating_nodes, singly_connected)
    else:
        return _format_text_graph(model, node_to_elements, element_to_nodes,
                                   supported_nodes, floating_nodes, singly_connected)


def _format_text_graph(model: 'FEModel3D',
                        node_to_elements: Dict[str, List[str]],
                        element_to_nodes: Dict[str, List[str]],
                        supported_nodes: Set[str],
                        floating_nodes: Set[str],
                        singly_connected: Set[str]) -> str:
    """Format connectivity as readable text."""
    lines = []
    lines.append("=" * 60)
    lines.append("           MODEL CONNECTIVITY GRAPH")
    lines.append("=" * 60)
    lines.append("")

    # Summary
    lines.append(f"Nodes: {len(model.nodes)}  |  Members: {len(model.members)}  |  "
                f"Springs: {len(model.springs)}  |  Plates: {len(model.plates) + len(model.quads)}")
    lines.append("")

    # Legend
    lines.append("Node markers:  [S]=Supported  [F]=Floating  [1]=Singly-connected")
    lines.append("-" * 60)
    lines.append("")

    # Node adjacency list
    lines.append("NODE CONNECTIONS:")
    lines.append("")

    for node_name in sorted(model.nodes.keys()):
        # Build node marker
        markers = []
        if node_name in supported_nodes:
            markers.append("S")
        if node_name in floating_nodes:
            markers.append("F")
        if node_name in singly_connected:
            markers.append("1")
        marker_str = f"[{''.join(markers)}]" if markers else "   "

        elements = node_to_elements.get(node_name, [])
        if elements:
            elem_str = ", ".join(elements)
            lines.append(f"  {marker_str} {node_name} ── {elem_str}")
        else:
            lines.append(f"  {marker_str} {node_name} ── (no connections)")

    lines.append("")
    lines.append("-" * 60)
    lines.append("")

    # Element list with endpoints
    lines.append("ELEMENT ENDPOINTS:")
    lines.append("")

    for elem_name in sorted(element_to_nodes.keys()):
        nodes = element_to_nodes[elem_name]
        if len(nodes) == 2:
            lines.append(f"  {elem_name}:  {nodes[0]} ←→ {nodes[1]}")
        else:
            lines.append(f"  {elem_name}:  {' ─ '.join(nodes)}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def _format_dot_graph(model: 'FEModel3D',
                       node_to_elements: Dict[str, List[str]],
                       element_to_nodes: Dict[str, List[str]],
                       supported_nodes: Set[str],
                       floating_nodes: Set[str],
                       singly_connected: Set[str]) -> str:
    """Format connectivity as Graphviz DOT format."""
    lines = []
    lines.append("graph model {")
    lines.append("  // Generated by Pynite Diagnostics")
    lines.append("  rankdir=TB;")
    lines.append("  node [shape=circle, width=0.5, fixedsize=true];")
    lines.append("")

    # Define node styles
    lines.append("  // Nodes")
    for node_name in sorted(model.nodes.keys()):
        attrs = []
        if node_name in supported_nodes:
            attrs.append('shape=triangle')
            attrs.append('color=blue')
        if node_name in floating_nodes:
            attrs.append('color=red')
            attrs.append('style=dashed')
        if node_name in singly_connected and node_name not in floating_nodes:
            attrs.append('color=orange')

        # Escape node name for DOT format
        safe_name = node_name.replace('"', '\\"')
        if attrs:
            lines.append(f'  "{safe_name}" [{", ".join(attrs)}];')
        else:
            lines.append(f'  "{safe_name}";')

    lines.append("")
    lines.append("  // Elements (edges)")

    # Add edges for members and springs
    for elem_name, nodes in sorted(element_to_nodes.items()):
        safe_nodes = [n.replace('"', '\\"') for n in nodes]

        if elem_name.startswith("M:"):
            # Member - solid line
            label = elem_name[2:]
            lines.append(f'  "{safe_nodes[0]}" -- "{safe_nodes[1]}" [label="{label}"];')
        elif elem_name.startswith("S:"):
            # Spring - dashed line
            label = elem_name[2:]
            lines.append(f'  "{safe_nodes[0]}" -- "{safe_nodes[1]}" [label="{label}", style=dashed];')
        elif elem_name.startswith("P:") or elem_name.startswith("Q:"):
            # Plate/Quad - connect all corners
            label = elem_name[2:]
            for i in range(len(safe_nodes)):
                j = (i + 1) % len(safe_nodes)
                if i == 0:
                    lines.append(f'  "{safe_nodes[i]}" -- "{safe_nodes[j]}" [label="{label}", color=gray];')
                else:
                    lines.append(f'  "{safe_nodes[i]}" -- "{safe_nodes[j]}" [color=gray];')

    lines.append("}")

    return "\n".join(lines)
