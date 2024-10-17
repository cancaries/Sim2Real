import math

def is_straight_path(polyline, tolerance=0.01):
    if len(polyline) < 3:
        return True  # 如果点少于3个，默认认为是直线
    start = polyline[0]
    end = polyline[-1]
    for point in polyline[1:-1]:
        if not is_point_on_line(start, end, point, tolerance):
            return False
    return True

def is_point_on_line(start, end, point, tolerance):
    x1, y1, _ = start
    x2, y2, _ = end
    x, y, _ = point
    # 向量法判断点是否在直线上
    if x2 - x1 == 0:  # 处理垂直线的情况
        return abs(x - x1) < tolerance
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    return abs(y - (k * x + b)) < tolerance


def detect_turns(polyline, angle_threshold=10):
    if len(polyline) < 3:
        return []
    turns = []
    for i in range(1, len(polyline) - 1):
        p1 = polyline[i - 1]
        p2 = polyline[i]
        p3 = polyline[i + 1]
        angle = calculate_angle(p1, p2, p3)
        if angle > angle_threshold:
            direction = calculate_turn_direction(p1, p2, p3)
            turns.append((i, direction))
    return turns


def calculate_angle(p1, p2, p3):
    def vector(p1, p2):
        return (p2[0] - p1[0], p2[1] - p1[1])
    
    def dot_product(v1, v2):
        return v1[0] * v2[0] + v1[1] * v2[1]
    
    def magnitude(v):
        return math.sqrt(v[0]**2 + v[1]**2)

    v1 = vector(p1, p2)
    v2 = vector(p2, p3)
    dot_prod = dot_product(v1, v2)
    mag_v1 = magnitude(v1)
    mag_v2 = magnitude(v2)
    
    if mag_v1 == 0 or mag_v2 == 0:
        return 0
    
    cos_angle = dot_prod / (mag_v1 * mag_v2)
    angle = math.acos(cos_angle) * (180.0 / math.pi)  # Convert to degrees
    return angle


def calculate_turn_direction(p1, p2, p3):
    def vector(p1, p2):
        return (p2[0] - p1[0], p2[1] - p1[1])
    
    v1 = vector(p1, p2)
    v2 = vector(p2, p3)
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    
    if cross_product > 0:
        return "left"
    elif cross_product < 0:
        return "right"
    else:
        return "straight"

def is_complex_path(polyline, angle_threshold=10, segment_length_threshold=5):
    turns = detect_turns(polyline, angle_threshold)
    if not turns:
        return False
    segments = []
    current_segment = [polyline[0]]
    for i, direction in turns:
        current_segment.append(polyline[i])
        segments.append(current_segment)
        current_segment = [polyline[i]]
    current_segment.append(polyline[-1])
    segments.append(current_segment)
    
    # Check if segments are long enough to be considered
    for segment in segments:
        if len(segment) >= segment_length_threshold:
            return True
    return False

def calculate_direction(points):
    # 计算最后几个点的方向向量
    x1, y1, _ = points[0]
    x2, y2, _ = points[-1]
    return (x2 - x1, y2 - y1)

def calculate_turn_relation(end_direction, exit_direction):
    ex, ey = end_direction
    nx, ny = exit_direction
    cross_product = ex * ny - ey * nx
    dot_product = ex * nx + ey * ny
    angle = math.atan2(cross_product, dot_product) * (180.0 / math.pi)
    if -45 <= angle <= 45:
        return 'LANEFOLLOW'
    elif angle > 45:
        return 'TURNLEFT'
    else:
        return 'TURNRIGHT'