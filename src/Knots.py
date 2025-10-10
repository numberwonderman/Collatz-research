import numpy as np
import matplotlib.pyplot as plt

class CelticKnot:
    def __init__(self, width, height):
        # Primary grid size (number of squares)
        self.width = width
        self.height = height
        self.primary_grid = self._create_primary_grid()
        self.secondary_grid = self._create_secondary_grid()
        self.tertiary_grid = self._create_tertiary_grid()
        self.skeleton_edges = []
        self.band_labels = {}
        self.breaklines = set()  # Set of breakline edges (pairs of points)

    def _create_primary_grid(self):
        # Grid points at integer coordinates
        return [(x, y) for y in range(self.height + 1) for x in range(self.width + 1)]

    def _create_secondary_grid(self):
        # Secondary points inside primary grid squares, offset by 0.5 in both directions
        return [(x + 0.5, y + 0.5) for y in range(self.height) for x in range(self.width)]

    def _create_tertiary_grid(self):
        # Combination of primary and secondary points
        return self.primary_grid + self.secondary_grid

    def add_breakline(self, point1, point2):
        """
        Add a breakline between two adjacent grid points.
        These edges block skeleton lines from crossing.
        """
        if not self._are_adjacent(point1, point2):
            raise ValueError("Breaklines can only connect adjacent grid points")
        if self._crosses_existing_breakline(point1, point2):
            raise ValueError("Breaklines cannot cross existing breaklines")
        self.breaklines.add(frozenset([point1, point2]))

    def _are_adjacent(self, p1, p2):
        # Adjacent means horizontal or vertical neighbors by 1
        x1, y1 = p1
        x2, y2 = p2
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        return (dx == 1 and dy == 0) or (dx == 0 and dy == 1)

    def _crosses_existing_breakline(self, p1, p2):
        # Simple check: no crossing breaklines allowed
        # Could be extended for comprehensive intersection tests
        for bl in self.breaklines:
            pts = list(bl)
            if self._lines_cross(p1, p2, pts[0], pts[1]):
                return True
        return False

    def _lines_cross(self, a1, a2, b1, b2):
        # Check if two line segments (a1-a2 and b1-b2) intersect
        def ccw(pA, pB, pC):
            return (pC[1] - pA[1]) * (pB[0] - pA[0]) > (pB[1] - pA[1]) * (pC[0] - pA[0])
        return (ccw(a1, b1, b2) != ccw(a2, b1, b2)) and (ccw(a1, a2, b1) != ccw(a1, a2, b2))

    def generate_skeleton(self):
        self.skeleton_edges = []
        for y in range(self.height):
            for x in range(self.width):
                # cell corners
                bottom_left = (x, y)
                bottom_right = (x + 1, y)
                top_left = (x, y + 1)
                top_right = (x + 1, y + 1)

                # Determine diagonal orientation
                if (x + y) % 2 == 0:
                    diag = (bottom_left, top_right)
                    # Breaklines that form the cell edges for this diagonal:
                    cell_edges = [
                        frozenset((bottom_left, bottom_right)),
                        frozenset((bottom_left, top_left)),
                        frozenset((top_right, bottom_right)),
                        frozenset((top_right, top_left))
                    ]
                else:
                    diag = (top_left, bottom_right)
                    cell_edges = [
                        frozenset((top_left, top_right)),
                        frozenset((top_left, bottom_left)),
                        frozenset((bottom_right, top_right)),
                        frozenset((bottom_right, bottom_left))
                    ]

                # Blocking logic for breaklines
                is_blocked = False
                # 1. Check if any cell edge is a breakline (Cell-Wall Block)
                if any(edge in self.breaklines for edge in cell_edges):
                    is_blocked = True

                # 2. Check for crossing (Less common, for non-standard routing)
                if not is_blocked and self._crosses_breakline(diag):
                    is_blocked = True

                if not is_blocked:
                    self.skeleton_edges.append(diag)

    def _crosses_breakline(self, edge):
        # Checks for line segment intersection with any breakline
        for bl in self.breaklines:
            if self._lines_cross(edge[0], edge[1], list(bl)[0], list(bl)[1]):
                return True
        return False

    def label_bands(self):
        graph = {}
        point_ids = {}
        current_id = 0

        def get_point_id(pt):
            nonlocal current_id
            if pt not in point_ids:
                point_ids[pt] = current_id
                graph[current_id] = []
                current_id += 1
            return point_ids[pt]

        for p1, p2 in self.skeleton_edges:
            id1 = get_point_id(p1)
            id2 = get_point_id(p2)
            graph[id1].append(id2)
            graph[id2].append(id1)

        visited = set()
        bands = {}
        band_num = 0

        def dfs(node):
            stack = [node]
            component = []
            while stack:
                n = stack.pop()
                if n not in visited:
                    visited.add(n)
                    component.append(n)
                    stack.extend(graph[n])
            return component

        for node in graph:
            if node not in visited:
                band_num += 1
                comp = dfs(node)
                for n in comp:
                    bands[n] = band_num

        self.band_labels = {}
        for (p1, p2) in self.skeleton_edges:
            id1 = point_ids[p1]
            id2 = point_ids[p2]
            self.band_labels[(p1, p2)] = bands[id1]

    def plot(self):
        plt.figure(figsize=(8, 8))
        colors = plt.cm.get_cmap('tab10')
        for ((p1, p2), band) in self.band_labels.items():
            color = colors((band-1) % 10)
            xs = [p1[0], p2[0]]
            ys = [p1[1], p2[1]]
            plt.plot(xs, ys, color=color, linewidth=3)
        plt.title('Celtic Knot Skeleton Bands')
        plt.axis('equal')
        plt.axis('off')
        plt.show()

# Example usage:
knot = CelticKnot(4, 4)
knot.add_breakline((2, 2), (3, 2))  # Add breakline between adjacent points
knot.generate_skeleton()
knot.label_bands()
knot.plot()
