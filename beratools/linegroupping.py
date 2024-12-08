import networkit as nk
from shapely import force_2d, STRtree, get_point
from shapely.ops import split
import numpy as np
from shapely.geometry import Point, MultiPolygon, Polygon, LineString, MultiLineString
from enum import IntEnum, unique
from collections import defaultdict
from itertools import chain
from typing import Union
import geopandas as gpd

from dataclasses import dataclass, field
from mergelines import MergeLines

ANGLE_TOLERANCE = np.pi / 10
TURN_ANGLE_TOLERANCE = np.pi * 0.5  # (little bigger than right angle)
GROUP_ATTRIBUTE = "group"
TRIM_THRESHOLD = 0.05
TRANSECT_LENGTH = 10


def points_in_line(line):
    point_list = []
    try:
        for point in list(line.coords):  # loops through every point in a line
            # loops through every vertex of every segment
            if point:  # adds all the vertices to segment_list, which creates an array
                point_list.append(Point(point[0], point[1]))
    except Exception as e:
        print(e)

    return point_list

def get_angle(line, end_index):
    """
    Calculate the angle of the first or last segment
    line: LineString
    end_index: 0 or -1 of the line vertices. Consider the multipart.
    """
    pts = points_in_line(line)

    if end_index == 0:
        pt_1 = pts[0]
        pt_2 = pts[1]
    elif end_index == -1:
        pt_1 = pts[-1]
        pt_2 = pts[-2]

    delta_x = pt_2.x - pt_1.x
    delta_y = pt_2.y - pt_1.y
    angle = np.arctan2(delta_y, delta_x)

    return angle

@unique
class VertexClass(IntEnum):
    THREE_WAY_ZERO_PRIMARY_LINE = 1
    THREE_WAY_ONE_PRIMARY_LINE = 2
    FOUR_WAY_ZERO_PRIMARY_LINE = 3
    FOUR_WAY_ONE_PRIMARY_LINE = 4
    FOUR_WAY_TWO_PRIMARY_LINE = 5
    SINGLE_WAY = 6

@dataclass
class SingleLine():
    line_id: int = field(default=0)
    line: Union[LineString, MultiLineString] = field(default=None)
    sim_line: Union[LineString, MultiLineString] = field(default=None)
    vertex_index: int = field(default=0)
    group: int = field(default=0)

    def get_angle_for_line(self):
        return get_angle(self.sim_line, self.vertex_index)
    
    def end_transect(self):
        coords = self.line.coords
        end_seg = None
        if self.vertex_index == 0:
            end_seg = LineString([coords[0], coords[1]])
        elif self.vertex_index == -1:
            end_seg = LineString([coords[-1], coords[-2]])

        l_left = end_seg.offset_curve(10)        
        l_right = end_seg.offset_curve(-10)

        return LineString(
            [l_left.coords[0], l_right.coords[0]]
        )


class VertexNode:
    """
    """
    def __init__(self, line_id, line, sim_line, vertex_index, group=None) -> None:
        self.vertex = None
        self.line_list = []
        self.line_connected = []  # pairs of lines connected
        self.line_not_connected = []
        self.vertex_class = None

        if line:
            self.add_line(SingleLine(line_id, line, sim_line, vertex_index, group))

    def set_vertex(self, line, vertex_index):
        """ Set vertex coord """
        self.vertex = force_2d(get_point(line, vertex_index))
    
    def add_line(self, line_class):
        """ Common function for adding line when creating or merging other VertexNode """
        self.line_list.append(line_class)
        self.set_vertex(line_class.line, line_class.vertex_index)

    def get_line(self, line_id):
        for line in self.line_list:
            if line.line_id == line_id:
                return line.line

    def merge(self, vertex):
        """ merge other VertexNode if they have same vertex coords """
        self.add_line(vertex.line_list[0])

    def assign_vertex_class(self):
        if len(self.line_list) == 4:
            if len(self.line_connected) == 0:
                self.vertex_class = VertexClass.FOUR_WAY_ZERO_PRIMARY_LINE
            if len(self.line_connected) == 1:
                self.vertex_class = VertexClass.FOUR_WAY_ONE_PRIMARY_LINE
            if len(self.line_connected) == 2:
                self.vertex_class = VertexClass.FOUR_WAY_TWO_PRIMARY_LINE
        elif len(self.line_list) == 3:
            if len(self.line_connected) == 0:
                self.vertex_class = VertexClass.THREE_WAY_ZERO_PRIMARY_LINE
            if len(self.line_connected) == 1:
                self.vertex_class = VertexClass.THREE_WAY_ONE_PRIMARY_LINE
        elif len(self.line_list) == 1:
            self.vertex_class = VertexClass.SINGLE_WAY

    def has_group_attr(self):
        """If all values in group list are valid value, return True"""
        for i in self.line_list:
            if not i.group:
                return False

        return True

    def check_connectivity(self):
        if self.has_group_attr():
            self.group_line_by_attribute()
        else:
            self.group_line_by_angle()

        # record line not connected
        all_line_ids = {i.line_id for i in self.line_list} # set of line id
        self.line_not_connected = list(all_line_ids - set(chain(*self.line_connected)))
        
        self.assign_vertex_class()
        
    def group_line_by_attribute(self):
        group_line = defaultdict(list)
        for i in self.line_list:
            group_line[i.group].append(i.line_id)

        for value in group_line.values():
            if len(value) > 1:
                self.line_connected.append(value)

    def group_line_by_angle(self):
        """ generate connectivity of all lines """
        if len(self.line_list) == 1:
            return

        # if there are 2 and more lines
        new_angles = [i.get_angle_for_line() for i in self.line_list]
        angle_visited = [False] * len(new_angles)

        if len(self.line_list) == 2:
            angle_diff = abs(new_angles[0] - new_angles[1])
            angle_diff = angle_diff if angle_diff <= np.pi else angle_diff-np.pi

            #if angle_diff >= TURN_ANGLE_TOLERANCE:
            self.line_connected.append(
                (
                    self.line_list[0].line_id,
                    self.line_list[1].line_id,
                )
            )
            return

        # three and more lines
        for i, angle_1 in enumerate(new_angles):
            for j, angle_2 in enumerate(new_angles[i + 1 :]):
                if not angle_visited[i+j+1]:
                    angle_diff = abs(angle_1 - angle_2)
                    angle_diff = angle_diff if angle_diff <= np.pi else angle_diff-np.pi
                    if angle_diff < ANGLE_TOLERANCE or np.pi-ANGLE_TOLERANCE < abs(angle_1-angle_2) < np.pi+ANGLE_TOLERANCE:
                        angle_visited[j+i+1] = True  # tenth of PI
                        self.line_connected.append(
                            (
                                self.line_list[i].line_id,
                                self.line_list[i+j+1].line_id,
                            )
                        )

class LineGroupping:
    def __init__(self, in_line_file, in_poly_file):
        # remove empty and null geometry
        self.data = gpd.read_file(in_line_file)
        self.data = self.data[~self.data.geometry.isna() & ~self.data.geometry.is_empty]
        self.data.reset_index(inplace=True, drop=True)

        self.sim_geom = self.data.simplify(10)

        self.G = nk.Graph(len(self.data))
        self.merged_vertex_list = []
        self.has_groub_attr = False
        self.groups = [None] * len(self.data)
        self.vertex_of_concern = []

        self.polys = gpd.read_file(in_poly_file)
        self.merged = None  # merged lines

    def create_vertex_list(self):
        self.vertex_list = []

        # check if data has group column
        if GROUP_ATTRIBUTE in self.data.keys():
            self.groups = self.data[GROUP_ATTRIBUTE]
            self.has_groub_attr = True

        for idx, s_geom, geom, group in zip(
            *zip(*self.sim_geom.items()), self.data.geometry, self.groups
        ):
            self.vertex_list.append(VertexNode(idx, geom, s_geom, 0, group))
            self.vertex_list.append(VertexNode(idx, geom, s_geom, -1, group))

        v_points = []
        for i in self.vertex_list:
            v_points.append(i.vertex.buffer(1))  # small polygon around vertices

        v_index = STRtree(v_points)

        vertex_visited = [False] * len(self.vertex_list)

        for i, pt in enumerate(v_points):
            if vertex_visited[i]:
                continue

            s_list = v_index.query(pt)

            vertex = self.vertex_list[i]
            if len(s_list) > 1:
                for j in s_list:
                    if j != i:
                        vertex.merge(self.vertex_list[j])
                        vertex_visited[j] = True

            self.merged_vertex_list.append(vertex)
            vertex_visited[i] = True

        for i in self.merged_vertex_list:
            i.check_connectivity()

        for i in self.merged_vertex_list:
            if i.line_connected:
                # print(i.line_connected)
                for edge in i.line_connected:
                    self.G.addEdge(edge[0], edge[1])

    def group_lines(self):
        cc = nk.components.ConnectedComponents(self.G)
        cc.run()
        print("number of components ", cc.numberOfComponents())

        group = 0
        for i in range(cc.numberOfComponents()):
            component = cc.getComponents()[i]
            for id in component:
                self.groups[id] = group

            group += 1

    def find_vertex_for_poly_overlaps(self):
        concern_classes = (
            VertexClass.FOUR_WAY_ONE_PRIMARY_LINE,
            VertexClass.THREE_WAY_ONE_PRIMARY_LINE,
            VertexClass.SINGLE_WAY
        )
        self.vertex_of_concern = [
            i for i in self.merged_vertex_list if i.vertex_class in concern_classes
        ]

    def line_and_poly_cleanup(self):
        sindex_poly = self.polys.sindex

        for i in self.vertex_of_concern:
            idx = sindex_poly.query(i.vertex, predicate="within")
            if len(idx) == 0:
                continue
        
            if i.vertex_class == VertexClass.SINGLE_WAY:
                if len(idx) > 1:
                    print('Too many polygons')
                    continue

                # TODO use only last segment to reduce computation
                single_line = i.line_list[0]
                poly = self.polys.loc[*idx].geometry
                split_poly = split(poly, single_line.end_transect())
                
                if len(split_poly.geoms) != 2:
                    continue

                # check geom type
                none_poly = False
                for i in split_poly.geoms:
                    if i.geom_type != 'Polygon':
                        none_poly = True

                if none_poly:
                    continue
                
                # only two polygons in split_poly
                if split_poly.geoms[0].area > split_poly.geoms[1].area:
                    poly = split_poly.geoms[0]
                else:
                    poly = split_poly.geoms[1]
                    
                self.polys.at[idx[0], "geometry"] = poly 

                continue
            
            # other classes
            poly_trim = []
            primary_lines = []

            # retrieve primary lines
            for j in i.line_connected[0]:  # only one connected line is available
                primary_lines.append(i.get_line(j))

            for j in i.line_not_connected:  # only one connected line is available
                trim = PolygonTrimming(line_index = j, 
                                       line_cleanup = i.get_line(j))
                
                poly_trim.append(trim)

            polys = self.polys.loc[idx].geometry
            poly_primary = []
            for j, p in polys.items():
                if p.contains(primary_lines[0]) or p.contains(primary_lines[1]):
                    poly_primary.append(p)
                else:
                    for trim in poly_trim:
                        if p.contains(trim.line_cleanup):
                            trim.poly_cleanup = p
                            trim.poly_index = j

            poly_primary = MultiPolygon(poly_primary)
            for t in poly_trim:
                t.poly_primary = poly_primary

            for p in poly_trim:
                p.trim()
                self.polys.at[p.poly_index, 'geometry'] = p.poly_cleanup
                self.data.at[p.line_index, 'geometry'] = p.line_cleanup

    def run_groupping(self):    
        self.create_vertex_list()
        if not self.has_groub_attr:
            self.group_lines()

        self.find_vertex_for_poly_overlaps()
        self.data["group"] = self.groups  # assign group attribute

    def run_cleanup(self):    
        self.line_and_poly_cleanup()

    def run_line_merge(self):
        self.merged = self.data.dissolve(by=GROUP_ATTRIBUTE)
        self.merged.geometry = self.merged.line_merge()
        num = 0
        for i in self.merged.itertuples():
            num += 1
            if i.geometry.geom_type == "MultiLineString":
                worker = MergeLines(i.geometry)
                merged_line = worker.merge_all_lines()
                if merged_line:
                    self.merged.at[i.Index, "geometry"] = merged_line

    def save_file(self, out_line, out_poly):
        self.data.to_file(out_line)
        self.polys.to_file(out_poly)

@dataclass
class PolygonTrimming():
    """ Store polygon and line to trim. Primary polygon is used to trim both """

    poly_primary : MultiPolygon = field(default=None)
    poly_index: int = field(default=-1)
    poly_cleanup: Polygon = field(default=None)
    line_index: int = field(default=-1)
    line_cleanup: LineString = field(default=None)
    
    def trim(self):
        # TODO: chech why there is such cases
        if self.poly_cleanup is None:
            print('No polygon to trim.')
            return

        diff = self.poly_cleanup.difference(self.poly_primary)
        if diff.geom_type == "Polygon":
            self.poly_cleanup = diff
        elif diff.geom_type == "MultiPolygon":
            area = self.poly_cleanup.area
            reserved = []
            for i in diff.geoms:
                if i.area > TRIM_THRESHOLD * area:  # small part
                    reserved.append(i)

            if len(reserved) == 0:
                pass
            elif len(reserved) == 1:
                self.poly_cleanup = Polygon(*reserved)
            else:
                # TODO output all MultiPolygons which should be dealt with
                self.poly_cleanup = MultiPolygon(reserved)
        
        diff = self.line_cleanup.intersection(self.poly_cleanup)
        if diff.geom_type == "LineString":
            self.line_cleanup = diff
        elif diff.geom_type == "MultiLineString":
            length = self.line_cleanup.length
            reserved = []
            for i in diff.geoms:
                if i.length > TRIM_THRESHOLD * length:  # small part
                    reserved.append(i)

            if len(reserved) == 0:
                pass
            elif len(reserved) == 1:
                self.line_cleanup = LineString(*reserved)
            else:
                # TODO output all MultiPolygons which should be dealt with
                self.poly_cleanup = MultiLineString(reserved)
