import networkit as nk
import shapely
import numpy as np
from shapely.geometry import Point, MultiPolygon, Polygon
from enum import IntEnum, unique
from collections import defaultdict
from itertools import chain

ANGLE_TOLERANCE = np.pi / 10
TURN_ANGLE_TOLERANCE = np.pi * 0.5  # (little bigger than right angle)
GROUP_ATTRIBUTE = "group"


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
    # if np.isclose(pt_1.x, pt_2.x):
    #     angle = np.pi / 2
    #     if delta_y > 0:
    #         angle = np.pi / 2
    #     elif delta_y < 0:
    #         angle = -np.pi / 2
    # else:
    #     angle = np.arctan(delta_y / delta_x)
    angle = np.arctan2(delta_y, delta_x)

    # # arctan is in range [-pi/2, pi/2], regulate all angles to [[-pi/2, 3*pi/2]]
    # if delta_x < 0:
    #     angle += np.pi  # the second or fourth quadrant

    return angle

@unique
class VertexClass(IntEnum):
    THREE_WAY_ZERO_PRIMARY_LINE = 1
    THREE_WAY_ONE_PRIMARY_LINE = 2
    FOUR_WAY_ZERO_PRIMARY_LINE = 3
    FOUR_WAY_ONE_PRIMARY_LINE = 4
    FOUR_WAY_TWO_PRIMARY_LINE = 5

class VertexNode:
    """
    line_list: {'line_id': line_id,
                'sim_line': sim_line,
                'line': line,
                'vertex_index': 0 or -1, 
                'line_id': number}
    """
    def __init__(self, line_id, sim_line, line, vertex_index, group=None) -> None:
        self.vertex = None
        self.line_list = []  
        self.line_connected = []  # pairs of lines connected
        self.line_not_connected = []
        self.vertex_class = None

        if line:
            self.add_line(line_id, sim_line, line, vertex_index, group)

    def set_vertex(self, line, vertex_index):
        """ Set vertex coord """
        self.vertex = shapely.force_2d(shapely.get_point(line, vertex_index))
    
    def add_line(self, line_id, sim_line, line, vertex_index, group=None):
        """ Common function for adding line when creating or merging other VertexNode """
        self.line_list.append({'line_id': line_id, 
                               'sim_line': sim_line, 
                               'line': line, 
                               'vertex_index': vertex_index, 
                               'group': group})
        self.set_vertex(line, vertex_index)

    def get_line(self, line_id):
        for line in self.line_list:
            if line['line_id'] == line_id:
                return line['line']

    def merge(self, vertex):
        """ merge other VertexNode if they have same vertex coords """
        self.add_line(vertex.line_list[0]['line_id'],
                      vertex.line_list[0]['sim_line'],
                      vertex.line_list[0]['line'], 
                      vertex.line_list[0]['vertex_index'], 
                      vertex.line_list[0]['group'])

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

    def has_group_attr(self):
        """If all values in group list are valid value, return True"""
        for i in self.line_list:
            if not i['group']:
                return False

        return True

    def check_connectivity(self):
        if self.has_group_attr():
            self.group_line_by_attribute()
        else:
            self.group_line_by_angle()

        # record line not connected
        all_line_ids = {i['line_id'] for i in self.line_list}  # set of line id
        self.line_not_connected = list(all_line_ids - set(chain(*self.line_connected)))
        
        self.assign_vertex_class()
        
    def group_line_by_attribute(self):
        group_line = defaultdict(list)
        for i in self.line_list:
            group_line[i['group']].append(i['line_id'])

        for value in group_line.values():
            if len(value) > 1:
                self.line_connected.append(value)

    def group_line_by_angle(self):
        """ generate connectivity of all lines """
        if len(self.line_list) == 1:
            return

        # if there are 2 and more lines
        angles = [get_angle(i['sim_line'], i['vertex_index']) for i in self.line_list]
        angle_visited = [False]*len(angles)

        if len(self.line_list) == 2:
            angle_diff = abs(angles[0] - angles[1])
            angle_diff = angle_diff if angle_diff <= np.pi else angle_diff-np.pi

            #if angle_diff >= TURN_ANGLE_TOLERANCE:
            self.line_connected.append((self.line_list[0]['line_id'], self.line_list[1]['line_id']))
            return

        # three and more lines
        for i, angle_1 in enumerate(angles):
            for j, angle_2 in enumerate(angles[i+1:]):
                if not angle_visited[i+j+1]:
                    angle_diff = abs(angle_1 - angle_2)
                    angle_diff = angle_diff if angle_diff <= np.pi else angle_diff-np.pi
                    if angle_diff < ANGLE_TOLERANCE or np.pi-ANGLE_TOLERANCE < abs(angle_1-angle_2) < np.pi+ANGLE_TOLERANCE:
                        angle_visited[j+i+1] = True  # tenth of PI
                        self.line_connected.append((self.line_list[i]['line_id'], self.line_list[i+j+1]['line_id']))


class LineGroupping:
    def __init__(self, data):
        # remove empty and null geometry
        self.data = data
        self.sim_geom = self.data.simplify(10)

        self.G = nk.Graph(len(self.data))
        self.merged_vertex_list = []
        self.has_groub_attr = False
        self.groups = [None] * len(self.data)
        self.vertex_of_concern = []

        self.create_vertex_list()
        if not self.has_groub_attr:
            self.group_lines()

        self.find_vertex_for_poly_overlaps()

    def create_vertex_list(self):
        self.vertex_list = []

        # check if data has group column
        if GROUP_ATTRIBUTE in self.data.keys():
            self.groups = self.data[GROUP_ATTRIBUTE]
            self.has_groub_attr = True

        for idx, s_geom, geom, group in zip(
            *zip(*self.sim_geom.items()), self.data.geometry, self.groups
        ):
            self.vertex_list.append(VertexNode(idx, s_geom, geom, 0, group))
            self.vertex_list.append(VertexNode(idx, s_geom, geom, -1, group))

        v_points = []
        for i in self.vertex_list:
            v_points.append(i.vertex.buffer(1))  # small polygon around vertices

        v_index = shapely.STRtree(v_points)

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
        )
        self.vertex_of_concern = [
            i for i in self.merged_vertex_list if i.vertex_class in concern_classes
        ]

def cleanup_poly_and_line(p_cleanup, p_primary):
    p_cleanup_new = []
    for p in p_cleanup:
        diff = p["poly"].difference(p_primary)
        if diff.geom_type == "Polygon":
            p["poly"] = diff
            p["line"] = p["line"].intersection(p["poly"])
            p_cleanup_new.append(p)
        elif diff.geom_type == "MultiPolygon":
            area = p["poly"].area
            reserved = []
            for i in diff.geoms:
                if i.area > 0.01 * area:  # small part
                    reserved.append(i)

            if len(reserved) == 0:
                pass
            elif len(reserved) == 1:
                p["poly"] = Polygon(*reserved)
                p["line"] = p["line"].intersection(p["poly"])
                p_cleanup_new.append(p)
            else:
                p["poly"] = MultiPolygon(reserved)
                p["line"] = p["line"].intersection(p["poly"])
                p_cleanup_new.append(p)

    return p_cleanup_new