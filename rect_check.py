#!/usr/bin/env python3.11

import argparse

desc = """
The program loads a text file containing either 4 2D coordinates or 5 3D coordinates, 
where axes are comma-separated and points are newline-separated. 
- For 4 2D coordinates: It checks if the first three coordinates can form a rectangle 
and whether the fourth coordinate lies inside this rectangle. 
- For 5 3D coordinates: It verifies if the first three coordinates form a rectangle, 
if the fourth can complete a rectangular cuboid with this rectangle, and if the fifth point is within the cuboid.
"""

__author__ = "Marin Mrčela"
__email__ = "marin.mrcela@gmail.com"


class Point:
    """A point in 2-or-3-dimensional space."""

    def __init__(self, ct: tuple[float, float] | tuple[float, float, float]
                 | tuple[None, None] | tuple[None, None, None]):
        """
        Initialize a point with x,y and optional z coordinate.

        Parameters:
        ct :
        A tuple of two floats representing the coordinates of a point in 2D space,
        or a tuple of three floats representing the coordinates of a point in 3D space.
        """
        self.x = ct[0]
        self.y = ct[1]
        self.z = None

        if len(ct) == 3:
            self.z = ct[2]

    def __str__(self) -> str:
        if self.z is None:
            return f"({self.x}, {self.y})"
        else:
            return f"({self.x}, {self.y}, {self.z})"

    def __iter__(self):
        """Iterate over the coordinates of the point."""
        if self.z is None:
            yield self.x
            yield self.y
        else:
            yield self.x
            yield self.y
            yield self.z


class Base:
    """Loads, checks and parses input data."""
    def __init__(self, _file_path):
        self.file_path = _file_path
        self.mode = 0  # 2 for 2D, 3 for 3D
        self.parsed_lines = tuple()
        self.point_a = Point((None, None))
        self.point_b = Point((None, None))
        self.point_c = Point((None, None))
        self.point_d_arb = Point((None, None, None))
        self.point_x = Point((None, None))

    def load_file(self):
        """Load the input file containing points as coordinates.
        Example of a valid file:
        5, 1
        -3, -3
        -5, 1
        3, 5
        """
        with open(self.file_path, 'r') as file:
            self.parsed_lines = tuple(tuple(float(num) for num in line.strip().split(',')) for line in file.readlines())

    def check_input(self):
        """Checks the input data validity in following points:
        - the first line must contain 2 or 3 coordinates,
        - all the following lines must have the consistent number of coordinates,
        - number of lines must be 4 for 2D or 5 for 3D."""

        # Mode is set according to the first line (point)
        self.mode = len(self.parsed_lines[0])
        if self.mode not in (2, 3):
            raise ValueError(f"Invalid number of coordinates in line 1; expecting 2 or 3 comma-separated coordinates.")

        dim_err_msg = "Inconsistent number of coordinates in line"
        pts_err_msg = "Invalid number of points; expecting"

        if self.mode == 2:
            if len(self.parsed_lines) != 4:
                raise ValueError(f"{pts_err_msg} 4.")
            for i in range(1, 4):
                if len(self.parsed_lines[i]) != 2:
                    raise ValueError(f"{dim_err_msg} {i + 1}.")
        if self.mode == 3:
            if len(self.parsed_lines) != 5:
                raise ValueError(f"{pts_err_msg} 5.")
            for i in range(1, 5):
                if len(self.parsed_lines[i]) != 3:
                    raise ValueError(f"{dim_err_msg} {i + 1}.")

    def parse_input(self):
        """Parses raw coordinates to Point instances."""
        self.point_a = Point(self.parsed_lines[0])
        self.point_b = Point(self.parsed_lines[1])
        self.point_c = Point(self.parsed_lines[2])
        if self.mode == 2:
            self.point_x = Point(self.parsed_lines[3])
        else:
            self.point_d_arb = Point(self.parsed_lines[3])
            self.point_x = Point(self.parsed_lines[4])

    def main(self):
        self.load_file()
        self.check_input()
        self.parse_input()


class Rectangle:
    def __init__(self, bs_instance):
        self.point_a = bs_instance.point_a
        self.point_b = bs_instance.point_b
        self.point_c = bs_instance.point_c
        self.point_d = Point((None, None))
        self.point_x = bs_instance.point_x
        self.is_rectangle = False

    @staticmethod
    def vector_2d(point_a: Point, point_b: Point) -> tuple[float, float]:
        """Calculates 2D vector defined by two endpoints."""
        return point_b.x - point_a.x, point_b.y - point_a.y

    @staticmethod
    def dot_product_2d(vector_a: tuple[float, float], vector_b: tuple[float, float]) -> float:
        """Dot product of two two-dimensional vectors."""
        return vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]

    def check_rectangle_2d(self):
        """Check if A, B, and C can form the vertices of a rectangle."""

        not_rect_msg = f"Points A{self.point_a}, B{self.point_b} and C{self.point_c} are not vertices of a rectangle."
        rr_msg = "Order of points has been rearranged to form a rectangle."

        # Calculate vectors BA and BC
        ba_vector = self.vector_2d(self.point_b, self.point_a)
        bc_vector = self.vector_2d(self.point_b, self.point_c)
        ac_vector = self.vector_2d(self.point_a, self.point_c)

        if is_null_vector(ba_vector) or is_null_vector(bc_vector) or is_null_vector(ac_vector):
            raise ValueError(f"{not_rect_msg} Points overlap.")

        # Calculate dot product
        dot_product_ba_bc = self.dot_product_2d(ba_vector, bc_vector)
        dot_product_ba_ac = self.dot_product_2d(ba_vector, ac_vector)
        dot_product_ac_bc = self.dot_product_2d(ac_vector, bc_vector)

        # If dot product is zero, vectors are perpendicular

        if dot_product_ba_bc == 0:
            # Points are perpendicular and in correct order
            pass
        elif dot_product_ba_ac == 0:
            # Points are perpendicular but in incorrect order
            tmp = self.point_a
            self.point_a = self.point_b
            self.point_b = tmp
            print(rr_msg)
        elif dot_product_ac_bc == 0:
            # Points are perpendicular but in incorrect order
            tmp = self.point_b
            self.point_b = self.point_c
            self.point_c = tmp
            print(rr_msg)
        else:
            raise ValueError(f"{not_rect_msg} Points do not form a right angle.")

        self.is_rectangle = True
        is_rect_msg = f"Points A{self.point_a}, B{self.point_b} and C{self.point_c} are vertices of a rectangle."
        print(is_rect_msg)

    def find_point_d_2d(self):
        """Finds vertex D by applying vector BC to point A."""
        bc_vector = self.vector_2d(self.point_b, self.point_c)
        self.point_d = Point((self.point_a.x + bc_vector[0], self.point_a.y + bc_vector[1]))

    def print_point_d_2d(self):
        print(f"Final vertex is D{self.point_d}.")

    def check_point_x_2d(self):
        """Check whether point X is inside or outside rectangle."""
        ab_vector = self.vector_2d(self.point_a, self.point_b)
        ax_vector = self.vector_2d(self.point_a, self.point_x)
        bc_vector = self.vector_2d(self.point_b, self.point_c)
        bx_vector = self.vector_2d(self.point_b, self.point_x)

        # Using projections of point X on vectors AB and BC
        if (0 <= self.dot_product_2d(ab_vector, ax_vector) <= self.dot_product_2d(ab_vector, ab_vector)
                and 0 <= self.dot_product_2d(bc_vector, bx_vector) <= self.dot_product_2d(bc_vector, bc_vector)):
            print(f"Point X{self.point_x} is inside the rectangle.")
        else:
            print(f"Point X{self.point_x} is outside the rectangle.")

    def get_diagonal_2d(self):
        """Calculate the length of the rectangle diagonals."""
        diagonal_2d = ((self.point_c.x - self.point_a.x) ** 2 + (self.point_c.y - self.point_a.y) ** 2) ** 0.5
        print(f"Length of diagonals is {diagonal_2d}.")

    def main(self):
        self.check_rectangle_2d()
        self.find_point_d_2d()
        self.print_point_d_2d()
        self.check_point_x_2d()
        self.get_diagonal_2d()


class RectCuboid:
    def __init__(self, bs_instance):
        self.point_a = bs_instance.point_a
        self.point_b = bs_instance.point_b
        self.point_c = bs_instance.point_c
        self.point_d = Point((None, None, None))
        self.point_e = Point((None, None, None))
        self.point_f = Point((None, None, None))
        self.point_g = Point((None, None, None))
        self.point_h = Point((None, None, None))
        self.point_d_arb = bs_instance.point_d_arb
        self.point_x = bs_instance.point_x
        self.is_rectangle = False
        self.is_rect_cuboid = False

    @staticmethod
    def vector_3d(point_a: Point, point_b: Point) -> tuple[float, float, float]:
        """Calculates 3D vector defined by two endpoints."""
        return point_b.x - point_a.x, point_b.y - point_a.y, point_b.z - point_a.z

    @staticmethod
    def dot_product_3d(vector_a: tuple[float, float, float], vector_b: tuple[float, float, float]) -> float:
        """Dot product of two three-dimensional vectors."""
        return vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1] + vector_a[2] * vector_b[2]

    @staticmethod
    def cross_product_3d(v1, v2):
        """Calculates the cross product of two vectors."""

        i = v1[1] * v2[2] - v1[2] * v2[1]
        j = v1[2] * v2[0] - v1[0] * v2[2]
        k = v1[0] * v2[1] - v1[1] * v2[0]
        return i, j, k

    @staticmethod
    def calculate_distance_3d(v1, v2):
        """Calculates distance between two points in 3D space."""
        return ((v2.x - v1.x) ** 2 + (v2.y - v1.y) ** 2 + (v2.z - v1.z) ** 2) ** 0.5

    @staticmethod
    def calculate_upper_vertex(point, vec):
        return Point((point.x + vec[0], point.y + vec[1], point.z + vec[2]))

    def check_rectangle_3d(self):
        """Check if A, B, and C can form the vertices of a rectangle."""

        not_rect_msg = f"Points A{self.point_a}, B{self.point_b} and C{self.point_c} are not vertices of a rectangle."
        rr_msg = "Order of points has been rearranged to form a rectangle."

        # Calculate vectors BA and BC
        ba_vector = self.vector_3d(self.point_b, self.point_a)
        bc_vector = self.vector_3d(self.point_b, self.point_c)
        ac_vector = self.vector_3d(self.point_a, self.point_c)

        if is_null_vector(ba_vector) or is_null_vector(bc_vector) or is_null_vector(ac_vector):
            raise ValueError(f"{not_rect_msg} Points overlap.")

        # Calculate dot product
        dot_product_ba_bc = self.dot_product_3d(ba_vector, bc_vector)
        dot_product_ba_ac = self.dot_product_3d(ba_vector, ac_vector)
        dot_product_ac_bc = self.dot_product_3d(ac_vector, bc_vector)

        # If dot product is zero, vectors are perpendicular

        if dot_product_ba_bc == 0:
            # Points are perpendicular and in correct order
            pass
        elif dot_product_ba_ac == 0:
            # Points are perpendicular but in incorrect order
            tmp = self.point_a
            self.point_a = self.point_b
            self.point_b = tmp
            print(rr_msg)
        elif dot_product_ac_bc == 0:
            # Points are perpendicular but in incorrect order
            tmp = self.point_b
            self.point_b = self.point_c
            self.point_c = tmp
            print(rr_msg)
        else:
            raise ValueError(f"{not_rect_msg} Points do not form a right angle.")

        self.is_rectangle = True
        is_rect_msg = f"Points A{self.point_a}, B{self.point_b} and C{self.point_c} are vertices of a rectangle."
        print(is_rect_msg)

    def find_point_d_3d(self):
        """Finds vertex D by applying vector BC to point A."""
        bc_vector = self.vector_3d(self.point_b, self.point_c)
        self.point_d = Point((self.point_a.x + bc_vector[0], self.point_a.y + bc_vector[1],
                              self.point_a.z + bc_vector[2]))

    def print_point_d_3d(self):
        print(f"Final vertex of the base rectangle is D{self.point_d}.")

    def are_parallel_3d(self, v1, v2):
        """
        Checks if two vectors are parallel by dot product method.
        If dot product is equal to scalar product of their magnitudes
        (cos 0) or equal to its negative value (cos pi), vectors are parallel.
        """

        dp = self.dot_product_3d(v1, v2)
        # Vector magnitude
        mag_v1 = (sum(val ** 2 for val in v1)) ** 0.5
        mag_v2 = (sum(val ** 2 for val in v2)) ** 0.5

        if dp == mag_v1 * mag_v2:
            # Vectors are parallel
            return True
        elif dp == -mag_v1 * mag_v2:
            # Vectors are parallel but in opposite directions
            return True
        else:
            return False

    def check_point_d_arb(self):
        """
        Checks whether arbitrary point D' can be a vertex of a rectangular cuboid
        by calculating normal vector to plane ABC and searching for a parallel vector
        among AD', BD', CD' and DD'. Finds upper vertices.
        """
        not_rcu_msg = f"Arbitrary point D'{self.point_d_arb} does not form a rectangular cuboid."
        is_rcu_msg = f"Arbitrary point D'{self.point_d_arb} forms a rectangular cuboid."

        ab_vector = self.vector_3d(self.point_a, self.point_b)
        bc_vector = self.vector_3d(self.point_b, self.point_c)
        normal = self.cross_product_3d(ab_vector, bc_vector)

        a_darb_vector = self.vector_3d(self.point_a, self.point_d_arb)
        b_darb_vector = self.vector_3d(self.point_b, self.point_d_arb)
        c_darb_vector = self.vector_3d(self.point_c, self.point_d_arb)
        d_darb_vector = self.vector_3d(self.point_d, self.point_d_arb)

        # Checks if D' overlaps with one of rectangle vertices
        if any(is_null_vector(vec) for vec in (a_darb_vector, b_darb_vector, c_darb_vector, d_darb_vector)):
            raise ValueError(not_rcu_msg)

        # Checks if any vector is parallel to normal
        # If it is, all the upper vertices are calculated
        if any(self.are_parallel_3d(vec, normal) for vec in
               (a_darb_vector, b_darb_vector, c_darb_vector, d_darb_vector)):
            if self.are_parallel_3d(a_darb_vector, normal):
                self.point_e = self.point_d_arb
                self.point_f = self.calculate_upper_vertex(self.point_b, a_darb_vector)
                self.point_g = self.calculate_upper_vertex(self.point_c, a_darb_vector)
                self.point_h = self.calculate_upper_vertex(self.point_d, a_darb_vector)
            elif self.are_parallel_3d(b_darb_vector, normal):
                self.point_e = self.calculate_upper_vertex(self.point_a, b_darb_vector)
                self.point_f = self.point_d_arb
                self.point_g = self.calculate_upper_vertex(self.point_c, b_darb_vector)
                self.point_h = self.calculate_upper_vertex(self.point_d, b_darb_vector)
            elif self.are_parallel_3d(c_darb_vector, normal):
                self.point_e = self.calculate_upper_vertex(self.point_a, c_darb_vector)
                self.point_f = self.calculate_upper_vertex(self.point_b, c_darb_vector)
                self.point_g = self.point_d_arb
                self.point_h = self.calculate_upper_vertex(self.point_d, c_darb_vector)
            else:
                self.point_e = self.calculate_upper_vertex(self.point_a, d_darb_vector)
                self.point_f = self.calculate_upper_vertex(self.point_b, d_darb_vector)
                self.point_g = self.calculate_upper_vertex(self.point_c, d_darb_vector)
                self.point_h = self.point_d_arb
            self.is_rect_cuboid = True
            print(is_rcu_msg)

        else:
            raise ValueError(not_rcu_msg)

    def check_point_x_3d(self):
        """
        Projects the vector between the centre of the cuboid and the
        observed point onto cuboid axis and determines whether the
        projection extends beyond the cuboid dimensions along that axis,
        which means the point is outside the cuboid.
        """

        def norm(v):
            return sum(x ** 2 for x in v) ** 0.5

        def dot(v1, v2):
            return sum(x * y for x, y in zip(v1, v2))

        vec_ab = self.vector_3d(self.point_a, self.point_b)
        len_ab = norm(vec_ab)
        vec_ab = (x / len_ab for x in vec_ab)

        vec_ad = self.vector_3d(self.point_a, self.point_d)
        len_ad = norm(vec_ad)
        vec_ad = (x / len_ad for x in vec_ad)

        vec_ae = self.vector_3d(self.point_a, self.point_e)
        len_ae = norm(vec_ae)
        vec_ae = (x / len_ae for x in vec_ae)

        cuboid_centre = Point(((self.point_a.x + self.point_g.x)/2,
                              (self.point_a.y + self.point_g.y)/2, (self.point_a.z + self.point_g.z)/2))

        def is_outside(vec, dir_vec, dir_len):
            return abs(dot(vec, dir_vec)) * 2 > dir_len

        vec_cc_px = self.vector_3d(cuboid_centre, self.point_x)

        is_outside_ab = is_outside(vec_cc_px, vec_ab, len_ab)
        is_outside_ad = is_outside(vec_cc_px, vec_ad, len_ad)
        is_outside_ae = is_outside(vec_cc_px, vec_ae, len_ae)

        if is_outside_ab or is_outside_ad or is_outside_ae:
            print(f"Point X{self.point_x} is outside the cuboid.")
        else:
            print(f"Point X{self.point_x} is inside the cuboid.")

    def print_diagonal_3d(self):
        diagonal_len = self.calculate_distance_3d(self.point_a, self.point_g)
        print(f"Length of space diagonals is {diagonal_len:.4f}.")

    def print_all_vertices_3d(self):
        print(f"Cuboid vertices are A{self.point_a}, B{self.point_b}, C{self.point_c}, D{self.point_d},"
              f"\nE{self.point_e}, F{self.point_f}, G{self.point_g}, H{self.point_h}.")

    def main(self):
        self.check_rectangle_3d()
        self.find_point_d_3d()
        self.check_point_d_arb()
        self.print_all_vertices_3d()
        self.check_point_x_3d()
        self.print_diagonal_3d()


def is_null_vector(vector: tuple) -> bool:
    """Checks if a vector is a null vector."""

    if all(component == 0 for component in vector):
        return True
    else:
        return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('file_path', type=str, nargs='?', default='coordinates.txt',
                        help='Path to the text file to be processed. Default is "coordinates.txt"')

    args = parser.parse_args()

    bs = Base(args.file_path)
    bs.main()

    if bs.mode == 2:
        rect = Rectangle(bs)
        rect.main()
    else:
        rcbd = RectCuboid(bs)
        rcbd.main()
