import json
import os
import cv2 as cv
import numpy as np


#evaluate_camera.py

def get_polylines(camera_annotation, width, height, sampling_factor=0.2):
    """
    Given a set of camera parameters, this function adapts the camera to the desired image resolution and then
    projects the 3D points belonging to the terrain model in order to give a dictionary associating the classes
    observed and the points projected in the image.

    :param camera_annotation: camera parameters in their json/dictionary format
    :param width: image width for evaluation
    :param height: image height for evaluation
    :return: a dictionary with keys corresponding to a class observed in the image ( a line of the 3D model whose
    projection falls in the image) and values are then the list of 2D projected points.
    """

    cam = Camera(width, height)
    cam.from_json_parameters(camera_annotation)
    # if cam.image_width != width:
    #     cam.scale_resolution(width / cam.image_width)
    field = SoccerPitch()
    projections = dict()
    sides = [
        np.array([1, 0, 0]),
        np.array([1, 0, -width + 1]),
        np.array([0, 1, 0]),
        np.array([0, 1, -height + 1])
    ]
    for key, points in field.sample_field_points(sampling_factor).items():
        projections_list = []
        in_img = False
        prev_proj = np.zeros(3)
        for i, point in enumerate(points):
            ext = cam.project_point(point)
            if ext[2] < 1e-5:
                # point at infinity or behind camera
                continue
            if 0 <= ext[0] < width and 0 <= ext[1] < height:

                if not in_img and i > 0:

                    line = np.cross(ext, prev_proj)
                    in_img_intersections = []
                    dist_to_ext = []
                    for side in sides:
                        intersection = np.cross(line, side)
                        intersection /= intersection[2]
                        if 0 <= intersection[0] < width and 0 <= intersection[1] < height:
                            in_img_intersections.append(intersection)
                            dist_to_ext.append(np.sqrt(np.sum(np.square(intersection - ext))))
                    if in_img_intersections:
                        intersection = in_img_intersections[np.argmin(dist_to_ext)]

                        projections_list.append(
                            {
                                "x": intersection[0],
                                "y": intersection[1]
                            }
                        )

                projections_list.append(
                    {
                        "x": ext[0],
                        "y": ext[1]
                    }
                )
                in_img = True
            elif in_img:
                # first point out
                line = np.cross(ext, prev_proj)

                in_img_intersections = []
                dist_to_ext = []
                for side in sides:
                    intersection = np.cross(line, side)
                    intersection /= intersection[2]
                    if 0 <= intersection[0] < width and 0 <= intersection[1] < height:
                        in_img_intersections.append(intersection)
                        dist_to_ext.append(np.sqrt(np.sum(np.square(intersection - ext))))
                if in_img_intersections:
                    intersection = in_img_intersections[np.argmin(dist_to_ext)]

                    projections_list.append(
                        {
                            "x": intersection[0],
                            "y": intersection[1]
                        }
                    )
                in_img = False
            prev_proj = ext
        if len(projections_list):
            projections[key] = projections_list
    return projections


def distance_to_polyline(point, polyline):
    """
    Computes euclidian distance between a point and a polyline.
    :param point: 2D point
    :param polyline: a list of 2D point
    :return: the distance value
    """
    if 0 < len(polyline) < 2:
        dist = distance(point, polyline[0])
        return dist
    else:
        dist_to_segments = []
        point_np = np.array([point["x"], point["y"], 1])

        for i in range(len(polyline) - 1):
            origin_segment = np.array([
                polyline[i]["x"],
                polyline[i]["y"],
                1
            ])
            end_segment = np.array([
                polyline[i + 1]["x"],
                polyline[i + 1]["y"],
                1
            ])
            line = np.cross(origin_segment, end_segment)
            line /= np.sqrt(np.square(line[0]) + np.square(line[1]))

            # project point on line l
            projected = np.cross((np.cross(np.array([line[0], line[1], 0]), point_np)), line)
            projected = projected / projected[2]

            v1 = projected - origin_segment
            v2 = end_segment - origin_segment
            k = np.dot(v1, v2) / np.dot(v2, v2)
            if 0 < k < 1:

                segment_distance = np.sqrt(np.sum(np.square(projected - point_np)))
            else:
                d1 = distance(point, polyline[i])
                d2 = distance(point, polyline[i + 1])
                segment_distance = np.min([d1, d2])

            dist_to_segments.append(segment_distance)
        return np.min(dist_to_segments)


def evaluate_camera_prediction(projected_lines, groundtruth_lines, threshold):
    """
    Computes confusion matrices for a level of precision specified by the threshold.
    A groundtruth line is correctly classified if it lies at less than threshold pixels from a line of the prediction
    of the same class.
    Computes also the reprojection error of each groundtruth point : the reprojection error is the L2 distance between
    the point and the projection of the line.
    :param projected_lines: dictionary of detected lines classes as keys and associated predicted points as values
    :param groundtruth_lines: dictionary of annotated lines classes as keys and associated annotated points as values
    :param threshold: distance in pixels that distinguishes good matches from bad ones
    :return: confusion matrix, per class confusion matrix & per class reprojection errors
    """
    global_confusion_mat = np.zeros((2, 2), dtype=np.float32)
    per_class_confusion = {}
    dict_errors = {}
    detected_classes = set(projected_lines.keys())
    groundtruth_classes = set(groundtruth_lines.keys())

    false_positives_classes = detected_classes - groundtruth_classes
    for false_positive_class in false_positives_classes:
        # false_positives = len(projected_lines[false_positive_class])
        if "Circle" not in false_positive_class:
            # Count only extremities for lines, independently of soccer pitch sampling
            false_positives = 2.
        else:
            false_positives = 9.
        per_class_confusion[false_positive_class] = np.array([[0., false_positives], [0., 0.]])
        global_confusion_mat[0, 1] += 1

    false_negatives_classes = groundtruth_classes - detected_classes
    for false_negatives_class in false_negatives_classes:
        false_negatives = len(groundtruth_lines[false_negatives_class])
        per_class_confusion[false_negatives_class] = np.array([[0., 0.], [false_negatives, 0.]])
        global_confusion_mat[1, 0] += 1

    common_classes = detected_classes - false_positives_classes

    for detected_class in common_classes:

        detected_points = projected_lines[detected_class]
        groundtruth_points = groundtruth_lines[detected_class]

        per_class_confusion[detected_class] = np.zeros((2, 2))

        all_below_dist = 1
        for point in groundtruth_points:

            dist_to_poly = distance_to_polyline(point, detected_points)
            if dist_to_poly < threshold:
                per_class_confusion[detected_class][0, 0] += 1
            else:
                per_class_confusion[detected_class][0, 1] += 1
                all_below_dist *= 0

            if detected_class in dict_errors.keys():
                dict_errors[detected_class].append(dist_to_poly)
            else:
                dict_errors[detected_class] = [dist_to_poly]

        if all_below_dist:
            global_confusion_mat[0, 0] += 1
        else:
            global_confusion_mat[0, 1] += 1

    return global_confusion_mat, per_class_confusion, dict_errors


#evaluate_extremities.py

def distance(point1, point2):
    """
    Computes euclidian distance between 2D points
    :param point1
    :param point2
    :return: euclidian distance between point1 and point2
    """
    diff = np.array([point1['x'], point1['y']]) - np.array([point2['x'], point2['y']])
    sq_dist = np.square(diff)
    return np.sqrt(sq_dist.sum())


def mirror_labels(lines_dict):
    """
    Replace each line class key of the dictionary with its opposite element according to a central projection by the
    soccer pitch center
    :param lines_dict: dictionary whose keys will be mirrored
    :return: Dictionary with mirrored keys and same values
    """
    mirrored_dict = dict()
    for line_class, value in lines_dict.items():
        mirrored_dict[SoccerPitch.symetric_classes[line_class]] = value
    return mirrored_dict


def evaluate_detection_prediction(groundtruth_lines, detected_lines, threshold=2.):
    """
    Evaluates the prediction of extremities. The extremities associated to a class are unordered. The extremities of the
    "Circle central" element is not well-defined for this task, thus this class is ignored.
    Computes confusion matrices for a level of precision specified by the threshold.
    A groundtruth extremity point is correctly classified if it lies at less than threshold pixels from the
    corresponding extremity point of the prediction of the same class.
    Computes also the euclidian distance between each predicted extremity and its closest groundtruth extremity, when
    both the groundtruth and the prediction contain the element class.

    :param projected_lines: dictionary of detected lines classes as keys and associated predicted extremities as values
    :param groundtruth_lines: dictionary of annotated lines classes as keys and associated annotated points as values
    :param threshold: distance in pixels that distinguishes good matches from bad ones
    :return: confusion matrix, per class confusion matrix & per class localization errors
    """
    confusion_mat = np.zeros((2, 2), dtype=np.float32)
    per_class_confusion = {}
    errors_dict = {}
    detected_classes = set(detected_lines.keys())
    groundtruth_classes = set(groundtruth_lines.keys())

    if "Circle central" in groundtruth_classes:
        groundtruth_classes.remove("Circle central")
    if "Circle central" in detected_classes:
        detected_classes.remove("Circle central")

    false_positives_classes = detected_classes - groundtruth_classes
    for false_positive_class in false_positives_classes:
        false_positives = len(detected_lines[false_positive_class])
        confusion_mat[0, 1] += false_positives
        per_class_confusion[false_positive_class] = np.array([[0., false_positives], [0., 0.]])

    false_negatives_classes = groundtruth_classes - detected_classes
    for false_negatives_class in false_negatives_classes:
        false_negatives = len(groundtruth_lines[false_negatives_class])
        confusion_mat[1, 0] += false_negatives
        per_class_confusion[false_negatives_class] = np.array([[0., 0.], [false_negatives, 0.]])

    common_classes = detected_classes - false_positives_classes

    for detected_class in common_classes:
        # if detected_class == "Circle center":
        #     continue
        detected_points = detected_lines[detected_class]

        groundtruth_points = groundtruth_lines[detected_class]

        groundtruth_extremities = [groundtruth_points[0], groundtruth_points[-1]]
        predicted_extremities = [detected_points[0], detected_points[-1]]
        per_class_confusion[detected_class] = np.zeros((2, 2))

        dist1 = distance(groundtruth_extremities[0], predicted_extremities[0])
        dist1rev = distance(groundtruth_extremities[1], predicted_extremities[0])

        dist2 = distance(groundtruth_extremities[1], predicted_extremities[1])
        dist2rev = distance(groundtruth_extremities[0], predicted_extremities[1])
        if dist1rev <= dist1 and dist2rev <= dist2:
            # reverse order
            dist1 = dist1rev
            dist2 = dist2rev

        errors_dict[detected_class] = [dist1, dist2]

        if dist1 < threshold:
            confusion_mat[0, 0] += 1
            per_class_confusion[detected_class][0, 0] += 1
        else:
            # treat too far detections as false positives
            confusion_mat[0, 1] += 1
            per_class_confusion[detected_class][0, 1] += 1

        if dist2 < threshold:
            confusion_mat[0, 0] += 1
            per_class_confusion[detected_class][0, 0] += 1

        else:
            # treat too far detections as false positives
            confusion_mat[0, 1] += 1
            per_class_confusion[detected_class][0, 1] += 1

    return confusion_mat, per_class_confusion, errors_dict


def scale_points(points_dict, s_width, s_height):
    """
    Scale points by s_width and s_height factors
    :param points_dict: dictionary of annotations/predictions with normalized point values
    :param s_width: width scaling factor
    :param s_height: height scaling factor
    :return: dictionary with scaled points
    """
    line_dict = {}
    for line_class, points in points_dict.items():
        scaled_points = []
        for point in points:
            new_point = {'x': point['x'] * (s_width -1), 'y': point['y'] * (s_height-1)}
            scaled_points.append(new_point)
        if len(scaled_points):
            line_dict[line_class] = scaled_points
    
    return line_dict


# camera.py

def pan_tilt_roll_to_orientation(pan, tilt, roll):
    """
    Conversion from euler angles to orientation matrix.
    :param pan:
    :param tilt:
    :param roll:
    :return: orientation matrix
    """
    Rpan = np.array([
        [np.cos(pan), -np.sin(pan), 0],
        [np.sin(pan), np.cos(pan), 0],
        [0, 0, 1]])
    Rroll = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]])
    Rtilt = np.array([
        [1, 0, 0],
        [0, np.cos(tilt), -np.sin(tilt)],
        [0, np.sin(tilt), np.cos(tilt)]])
    rotMat = np.dot(Rpan, np.dot(Rtilt, Rroll))
    return rotMat


def rotation_matrix_to_pan_tilt_roll(rotation):
    """
    Decomposes the rotation matrix into pan, tilt and roll angles. There are two solutions, but as we know that cameramen
    try to minimize roll, we take the solution with the smallest roll.
    :param rotation: rotation matrix
    :return: pan, tilt and roll in radians
    """
    orientation = np.transpose(rotation)
    first_tilt = np.arccos(orientation[2, 2])
    second_tilt = - first_tilt

    sign_first_tilt = 1. if np.sin(first_tilt) > 0. else -1.
    sign_second_tilt = 1. if np.sin(second_tilt) > 0. else -1.

    first_pan = np.arctan2(sign_first_tilt * orientation[0, 2], sign_first_tilt * - orientation[1, 2])
    second_pan = np.arctan2(sign_second_tilt * orientation[0, 2], sign_second_tilt * - orientation[1, 2])
    first_roll = np.arctan2(sign_first_tilt * orientation[2, 0], sign_first_tilt * orientation[2, 1])
    second_roll = np.arctan2(sign_second_tilt * orientation[2, 0], sign_second_tilt * orientation[2, 1])

    # print(f"first solution {first_pan*180./np.pi}, {first_tilt*180./np.pi}, {first_roll*180./np.pi}")
    # print(f"second solution {second_pan*180./np.pi}, {second_tilt*180./np.pi}, {second_roll*180./np.pi}")
    if np.fabs(first_roll) < np.fabs(second_roll):
        return first_pan, first_tilt, first_roll
    return second_pan, second_tilt, second_roll


def unproject_image_point(homography, point2D):
    """
    Given the homography from the world plane of the pitch and the image and a point localized on the pitch plane in the
    image, returns the coordinates of the point in the 3D pitch plane.
    /!\ Only works for correspondences on the pitch (Z = 0).
    :param homography: the homography
    :param point2D: the image point whose relative coordinates on the world plane of the pitch are to be found
    :return: A 2D point on the world pitch plane in homogenous coordinates (X,Y,1) with X and Y being the world
    coordinates of the point.
    """
    hinv = np.linalg.inv(homography)
    pitchpoint = hinv @ point2D
    pitchpoint = pitchpoint / pitchpoint[2]
    return pitchpoint


class Camera:

    def __init__(self, iwidth=960, iheight=540):
        self.position = np.zeros(3)
        self.rotation = np.eye(3)
        self.calibration = np.eye(3)
        self.radial_distortion = np.zeros(6)
        self.thin_prism_disto = np.zeros(4)
        self.tangential_disto = np.zeros(2)
        self.image_width = iwidth
        self.image_height = iheight
        self.xfocal_length = 1
        self.yfocal_length = 1
        self.principal_point = (self.image_width / 2, self.image_height / 2)

    def solve_pnp(self, point_matches):
        """
        With a known calibration matrix, this method can be used in order to retrieve rotation and translation camera
        parameters.
        :param point_matches: A list of pairs of 3D-2D point matches .
        """
        target_pts = np.array([pt[0] for pt in point_matches])
        src_pts = np.array([pt[1] for pt in point_matches])
        _, rvec, t, inliers = cv.solvePnPRansac(target_pts, src_pts, self.calibration, None)
        self.rotation, _ = cv.Rodrigues(rvec)
        self.position = - np.transpose(self.rotation) @ t.flatten()

    def refine_camera(self, pointMatches):
        """
        Once that there is a minimal set of initial camera parameters (calibration, rotation and position roughly known),
        this method can be used to refine the solution using a non-linear optimization procedure.
        :param pointMatches:  A list of pairs of 3D-2D point matches .

        """
        rvec, _ = cv.Rodrigues(self.rotation)
        target_pts = np.array([pt[0] for pt in pointMatches])
        src_pts = np.array([pt[1] for pt in pointMatches])

        rvec, t = cv.solvePnPRefineLM(target_pts, src_pts, self.calibration, None, rvec, -self.rotation @ self.position,
                                      (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 20000, 0.00001))
        self.rotation, _ = cv.Rodrigues(rvec)
        self.position = - np.transpose(self.rotation) @ t

    def from_homography(self, homography):
        """
        This method initializes the essential camera parameters from the homography between the world plane of the pitch
        and the image. It is based on the extraction of the calibration matrix from the homography (Algorithm 8.2 of
        Multiple View Geometry in computer vision, p225), then using the relation between the camera parameters and the
        same homography, we extract rough rotation and position estimates (Example 8.1 of Multiple View Geometry in
        computer vision, p196).
        :param homography: The homography that captures the transformation between the 3D flat model of the soccer pitch
         and its image.
        """
        success, _ = self.estimate_calibration_matrix_from_plane_homography(homography)
        if not success:
            return False

        hprim = np.linalg.inv(self.calibration) @ homography
        lambda1 = 1 / np.linalg.norm(hprim[:, 0])
        lambda2 = 1 / np.linalg.norm(hprim[:, 1])
        lambda3 = np.sqrt(lambda1 * lambda2)

        r0 = hprim[:, 0] / lambda1
        r1 = hprim[:, 1] / lambda2
        r2 = np.cross(r0, r1)

        R = np.column_stack((r0, r1, r2))
        u, s, vh = np.linalg.svd(R)
        R = u @ vh
        if np.linalg.det(R) < 0:
            u[:, 2] *= -1
            R = u @ vh
        self.rotation = R
        t = hprim[:, 2] * lambda3
        self.position = - np.transpose(R) @ t
        return True

    def to_json_parameters(self):
        """
        Saves camera to a JSON serializable dictionary.
        :return: The dictionary
        """
        pan, tilt, roll = rotation_matrix_to_pan_tilt_roll(self.rotation)
        camera_dict = {
            "pan_degrees": pan * 180. / np.pi,
            "tilt_degrees": tilt * 180. / np.pi,
            "roll_degrees": roll * 180. / np.pi,
            "position_meters": self.position.tolist(),
            "x_focal_length": self.xfocal_length,
            "y_focal_length": self.yfocal_length,
            "principal_point": [self.principal_point[0], self.principal_point[1]],
            "radial_distortion": self.radial_distortion.tolist(),
            "tangential_distortion": self.tangential_disto.tolist(),
            "thin_prism_distortion": self.thin_prism_disto.tolist()

        }
        return camera_dict

    def from_json_parameters(self, calib_json_object):
        """
        Loads camera parameters from dictionary.
        :param calib_json_object: the dictionary containing camera parameters.
        """
        self.principal_point = calib_json_object["principal_point"]
        self.image_width = 2 * self.principal_point[0]
        self.image_height = 2 * self.principal_point[1]
        self.xfocal_length = calib_json_object["x_focal_length"]
        self.yfocal_length = calib_json_object["y_focal_length"]

        self.calibration = np.array([
            [self.xfocal_length, 0, self.principal_point[0]],
            [0, self.yfocal_length, self.principal_point[1]],
            [0, 0, 1]
        ], dtype='float')

        pan = calib_json_object['pan_degrees'] * np.pi / 180.
        tilt = calib_json_object['tilt_degrees'] * np.pi / 180.
        roll = calib_json_object['roll_degrees'] * np.pi / 180.

        self.rotation = np.array([
            [-np.sin(pan) * np.sin(roll) * np.cos(tilt) + np.cos(pan) * np.cos(roll),
             np.sin(pan) * np.cos(roll) + np.sin(roll) * np.cos(pan) * np.cos(tilt), np.sin(roll) * np.sin(tilt)],
            [-np.sin(pan) * np.cos(roll) * np.cos(tilt) - np.sin(roll) * np.cos(pan),
             -np.sin(pan) * np.sin(roll) + np.cos(pan) * np.cos(roll) * np.cos(tilt), np.sin(tilt) * np.cos(roll)],
            [np.sin(pan) * np.sin(tilt), -np.sin(tilt) * np.cos(pan), np.cos(tilt)]
        ], dtype='float')

        self.rotation = np.transpose(pan_tilt_roll_to_orientation(pan, tilt, roll))

        self.position = np.array(calib_json_object['position_meters'], dtype='float')

        self.radial_distortion = np.array(calib_json_object['radial_distortion'], dtype='float')
        self.tangential_disto = np.array(calib_json_object['tangential_distortion'], dtype='float')
        self.thin_prism_disto = np.array(calib_json_object['thin_prism_distortion'], dtype='float')

    def distort(self, point):
        """
        Given a point in the normalized image plane, apply distortion
        :param point: 2D point on the normalized image plane
        :return: 2D distorted point
        """
        numerator = 1
        denominator = 1
        radius = np.sqrt(point[0] * point[0] + point[1] * point[1])

        for i in range(3):
            k = self.radial_distortion[i]
            numerator += k * radius ** (2 * (i + 1))
            k2n = self.radial_distortion[i + 3]
            denominator += k2n * radius ** (2 * (i + 1))

        radial_distortion_factor = numerator / denominator
        xpp = point[0] * radial_distortion_factor + \
              2 * self.tangential_disto[0] * point[0] * point[1] + self.tangential_disto[1] * (
                      radius ** 2 + 2 * point[0] ** 2) + \
              self.thin_prism_disto[0] * radius ** 2 + self.thin_prism_disto[1] * radius ** 4
        ypp = point[1] * radial_distortion_factor + \
              2 * self.tangential_disto[1] * point[0] * point[1] + self.tangential_disto[0] * (
                      radius ** 2 + 2 * point[1] ** 2) + \
              self.thin_prism_disto[2] * radius ** 2 + self.thin_prism_disto[3] * radius ** 4
        return np.array([xpp, ypp], dtype=np.float32)

    def project_point(self, point3D, distort=True):
        """
        Uses current camera parameters to predict where a 3D point is seen by the camera.
        :param point3D: The 3D point in world coordinates.
        :param distort: optional parameter to allow projection without distortion.
        :return: The 2D coordinates of the imaged point
        """
        point = point3D - self.position
        rotated_point = self.rotation @ np.transpose(point)
        if rotated_point[2] <= 1e-3 :
            return np.zeros(3)
        rotated_point = rotated_point / rotated_point[2]
        if distort:
            distorted_point = self.distort(rotated_point)
        else:
            distorted_point = rotated_point
        x = distorted_point[0] * self.xfocal_length + self.principal_point[0]
        y = distorted_point[1] * self.yfocal_length + self.principal_point[1]
        return np.array([x, y, 1])

    def scale_resolution(self, factor):
        """
        Adapts the internal parameters for image resolution changes
        :param factor: scaling factor
        """
        self.xfocal_length = self.xfocal_length * factor
        self.yfocal_length = self.yfocal_length * factor
        self.image_width = self.image_width * factor
        self.image_height = self.image_height * factor

        self.principal_point = (self.image_width / 2, self.image_height / 2)

        self.calibration = np.array([
            [self.xfocal_length, 0, self.principal_point[0]],
            [0, self.yfocal_length, self.principal_point[1]],
            [0, 0, 1]
        ], dtype='float')

    def draw_corners(self, image, color=(0, 255, 0)):
        """
        Draw the corners of a standard soccer pitch in the image.
        :param image: cv image
        :param color
        :return: the image mat modified.
        """
        field = SoccerPitch()
        for pt3D in field.point_dict.values():
            projected = self.project_point(pt3D)
            if projected[2] == 0.:
                continue
            projected /= projected[2]
            if 0 < projected[0] < self.image_width and 0 < projected[1] < self.image_height:
                cv.circle(image, (int(projected[0]), int(projected[1])), 3, color, 2)
        return image

    def draw_pitch(self, image, color=(0, 255, 0)):
        """
        Draws all the lines of the pitch on the image.
        :param image
        :param color
        :return: modified image
        """
        field = SoccerPitch()

        polylines = field.sample_field_points()
        for line in polylines.values():
            prev_point = self.project_point(line[0])
            for point in line[1:]:
                projected = self.project_point(point)
                if projected[2] == 0.:
                    continue
                projected /= projected[2]
                if 0 < projected[0] < self.image_width and 0 < projected[1] < self.image_height:
                    cv.line(image, (int(prev_point[0]), int(prev_point[1])), (int(projected[0]), int(projected[1])),
                            color, 1)
                prev_point = projected
        return image

    def draw_colorful_pitch(self, image, palette):
        """
        Draws all the lines of the pitch on the image, each line color is specified by the palette argument.

        :param image:
        :param palette: dictionary associating line classes names with their BGR color.
        :return: modified image
        """
        field = SoccerPitch()

        polylines = field.sample_field_points()
        for key, line in polylines.items():
            if key not in palette.keys():
                print(f"Can't draw {key}")
                continue
            prev_point = self.project_point(line[0])
            for point in line[1:]:
                projected = self.project_point(point)
                if projected[2] == 0.:
                    continue
                projected /= projected[2]
                if 0 < projected[0] < self.image_width and 0 < projected[1] < self.image_height:
                    # BGR color
                    cv.line(image, (int(prev_point[0]), int(prev_point[1])), (int(projected[0]), int(projected[1])),
                            palette[key][::-1], 1)
                prev_point = projected
        return image

    def estimate_calibration_matrix_from_plane_homography(self, homography):
        """
        This method initializes the calibration matrix from the homography between the world plane of the pitch
        and the image. It is based on the extraction of the calibration matrix from the homography (Algorithm 8.2 of
        Multiple View Geometry in computer vision, p225). The extraction is sensitive to noise, which is why we keep the
        principal point in the middle of the image rather than using the one extracted by this method.
        :param homography: homography between the world plane of the pitch and the image
        """
        H = np.reshape(homography, (9,))
        A = np.zeros((5, 6))
        A[0, 1] = 1.
        A[1, 0] = 1.
        A[1, 2] = -1.
        A[2, 3] = 9.0 / 16.0
        A[2, 4] = -1.0
        A[3, 0] = H[0] * H[1]
        A[3, 1] = H[0] * H[4] + H[1] * H[3]
        A[3, 2] = H[3] * H[4]
        A[3, 3] = H[0] * H[7] + H[1] * H[6]
        A[3, 4] = H[3] * H[7] + H[4] * H[6]
        A[3, 5] = H[6] * H[7]
        A[4, 0] = H[0] * H[0] - H[1] * H[1]
        A[4, 1] = 2 * H[0] * H[3] - 2 * H[1] * H[4]
        A[4, 2] = H[3] * H[3] - H[4] * H[4]
        A[4, 3] = 2 * H[0] * H[6] - 2 * H[1] * H[7]
        A[4, 4] = 2 * H[3] * H[6] - 2 * H[4] * H[7]
        A[4, 5] = H[6] * H[6] - H[7] * H[7]

        u, s, vh = np.linalg.svd(A)
        w = vh[-1]
        W = np.zeros((3, 3))
        W[0, 0] = w[0] / w[5]
        W[0, 1] = w[1] / w[5]
        W[0, 2] = w[3] / w[5]
        W[1, 0] = w[1] / w[5]
        W[1, 1] = w[2] / w[5]
        W[1, 2] = w[4] / w[5]
        W[2, 0] = w[3] / w[5]
        W[2, 1] = w[4] / w[5]
        W[2, 2] = w[5] / w[5]

        try:
            Ktinv = np.linalg.cholesky(W)
        except np.linalg.LinAlgError:
            K = np.eye(3)
            return False, K

        K = np.linalg.inv(np.transpose(Ktinv))
        K /= K[2, 2]

        self.xfocal_length = K[0, 0]
        self.yfocal_length = K[1, 1]
        # the principal point estimated by this method is very noisy, better keep it in the center of the image
        self.principal_point = (self.image_width / 2, self.image_height / 2)
        # self.principal_point = (K[0,2], K[1,2])
        self.calibration = np.array([
            [self.xfocal_length, 0, self.principal_point[0]],
            [0, self.yfocal_length, self.principal_point[1]],
            [0, 0, 1]
        ], dtype='float')
        return True, K


# soccerpitch.py

class SoccerPitch:
    """Static class variables that are specified by the rules of the game """
    GOAL_LINE_TO_PENALTY_MARK = 11.0
    PENALTY_AREA_WIDTH = 40.32
    PENALTY_AREA_LENGTH = 16.5
    GOAL_AREA_WIDTH = 18.32
    GOAL_AREA_LENGTH = 5.5
    CENTER_CIRCLE_RADIUS = 9.15
    GOAL_HEIGHT = 2.44
    GOAL_LENGTH = 7.32

    lines_classes = [
        'Big rect. left bottom',
        'Big rect. left main',
        'Big rect. left top',
        'Big rect. right bottom',
        'Big rect. right main',
        'Big rect. right top',
        'Circle central',
        'Circle left',
        'Circle right',
        'Goal left crossbar',
        'Goal left post left ',
        'Goal left post right',
        'Goal right crossbar',
        'Goal right post left',
        'Goal right post right',
        'Goal unknown',
        'Line unknown',
        'Middle line',
        'Side line bottom',
        'Side line left',
        'Side line right',
        'Side line top',
        'Small rect. left bottom',
        'Small rect. left main',
        'Small rect. left top',
        'Small rect. right bottom',
        'Small rect. right main',
        'Small rect. right top'
    ]

    symetric_classes = {
        'Side line top': 'Side line bottom',
        'Side line bottom': 'Side line top',
        'Side line left': 'Side line right',
        'Middle line': 'Middle line',
        'Side line right': 'Side line left',
        'Big rect. left top': 'Big rect. right bottom',
        'Big rect. left bottom': 'Big rect. right top',
        'Big rect. left main': 'Big rect. right main',
        'Big rect. right top': 'Big rect. left bottom',
        'Big rect. right bottom': 'Big rect. left top',
        'Big rect. right main': 'Big rect. left main',
        'Small rect. left top': 'Small rect. right bottom',
        'Small rect. left bottom': 'Small rect. right top',
        'Small rect. left main': 'Small rect. right main',
        'Small rect. right top': 'Small rect. left bottom',
        'Small rect. right bottom': 'Small rect. left top',
        'Small rect. right main': 'Small rect. left main',
        'Circle left': 'Circle right',
        'Circle central': 'Circle central',
        'Circle right': 'Circle left',
        'Goal left crossbar': 'Goal right crossbar',
        'Goal left post left ': 'Goal right post left',
        'Goal left post right': 'Goal right post right',
        'Goal right crossbar': 'Goal left crossbar',
        'Goal right post left': 'Goal left post left ',
        'Goal right post right': 'Goal left post right',
        'Goal unknown': 'Goal unknown',
        'Line unknown': 'Line unknown'
    }

    # RGB values
    palette = {
        'Big rect. left bottom': (127, 0, 0),
        'Big rect. left main': (102, 102, 102),
        'Big rect. left top': (0, 0, 127),
        'Big rect. right bottom': (86, 32, 39),
        'Big rect. right main': (48, 77, 0),
        'Big rect. right top': (14, 97, 100),
        'Circle central': (0, 0, 255),
        'Circle left': (255, 127, 0),
        'Circle right': (0, 255, 255),
        'Goal left crossbar': (255, 255, 200),
        'Goal left post left ': (165, 255, 0),
        'Goal left post right': (155, 119, 45),
        'Goal right crossbar': (86, 32, 139),
        'Goal right post left': (196, 120, 153),
        'Goal right post right': (166, 36, 52),
        'Goal unknown': (0, 0, 0),
        'Line unknown': (0, 0, 0),
        'Middle line': (255, 255, 0),
        'Side line bottom': (255, 0, 255),
        'Side line left': (0, 255, 150),
        'Side line right': (0, 230, 0),
        'Side line top': (230, 0, 0),
        'Small rect. left bottom': (0, 150, 255),
        'Small rect. left main': (254, 173, 225),
        'Small rect. left top': (87, 72, 39),
        'Small rect. right bottom': (122, 0, 255),
        'Small rect. right main': (255, 255, 255),
        'Small rect. right top': (153, 23, 153)
    }

    def __init__(self, pitch_length=105., pitch_width=68.):
        """
        Initialize 3D coordinates of all elements of the soccer pitch.
        :param pitch_length: According to FIFA rules, length belong to [90,120] meters
        :param pitch_width: According to FIFA rules, length belong to [45,90] meters
        """
        self.PITCH_LENGTH = pitch_length
        self.PITCH_WIDTH = pitch_width

        self.center_mark = np.array([0, 0, 0], dtype='float')
        self.halfway_and_bottom_touch_line_mark = np.array([0, pitch_width / 2.0, 0], dtype='float')
        self.halfway_and_top_touch_line_mark = np.array([0, -pitch_width / 2.0, 0], dtype='float')
        self.halfway_line_and_center_circle_top_mark = np.array([0, -SoccerPitch.CENTER_CIRCLE_RADIUS, 0],
                                                                dtype='float')
        self.halfway_line_and_center_circle_bottom_mark = np.array([0, SoccerPitch.CENTER_CIRCLE_RADIUS, 0],
                                                                   dtype='float')
        self.bottom_right_corner = np.array([pitch_length / 2.0, pitch_width / 2.0, 0], dtype='float')
        self.bottom_left_corner = np.array([-pitch_length / 2.0, pitch_width / 2.0, 0], dtype='float')
        self.top_right_corner = np.array([pitch_length / 2.0, -pitch_width / 2.0, 0], dtype='float')
        self.top_left_corner = np.array([-pitch_length / 2.0, -34, 0], dtype='float')

        self.left_goal_bottom_left_post = np.array([-pitch_length / 2.0, SoccerPitch.GOAL_LENGTH / 2., 0.],
                                                   dtype='float')
        self.left_goal_top_left_post = np.array(
            [-pitch_length / 2.0, SoccerPitch.GOAL_LENGTH / 2., -SoccerPitch.GOAL_HEIGHT], dtype='float')
        self.left_goal_bottom_right_post = np.array([-pitch_length / 2.0, -SoccerPitch.GOAL_LENGTH / 2., 0.],
                                                    dtype='float')
        self.left_goal_top_right_post = np.array(
            [-pitch_length / 2.0, -SoccerPitch.GOAL_LENGTH / 2., -SoccerPitch.GOAL_HEIGHT], dtype='float')

        self.right_goal_bottom_left_post = np.array([pitch_length / 2.0, -SoccerPitch.GOAL_LENGTH / 2., 0.],
                                                    dtype='float')
        self.right_goal_top_left_post = np.array(
            [pitch_length / 2.0, -SoccerPitch.GOAL_LENGTH / 2., -SoccerPitch.GOAL_HEIGHT], dtype='float')
        self.right_goal_bottom_right_post = np.array([pitch_length / 2.0, SoccerPitch.GOAL_LENGTH / 2., 0.],
                                                     dtype='float')
        self.right_goal_top_right_post = np.array(
            [pitch_length / 2.0, SoccerPitch.GOAL_LENGTH / 2., -SoccerPitch.GOAL_HEIGHT], dtype='float')

        self.left_penalty_mark = np.array([-pitch_length / 2.0 + SoccerPitch.GOAL_LINE_TO_PENALTY_MARK, 0, 0],
                                          dtype='float')
        self.right_penalty_mark = np.array([pitch_length / 2.0 - SoccerPitch.GOAL_LINE_TO_PENALTY_MARK, 0, 0],
                                           dtype='float')

        self.left_penalty_area_top_right_corner = np.array(
            [-pitch_length / 2.0 + SoccerPitch.PENALTY_AREA_LENGTH, -SoccerPitch.PENALTY_AREA_WIDTH / 2.0, 0],
            dtype='float')
        self.left_penalty_area_top_left_corner = np.array(
            [-pitch_length / 2.0, -SoccerPitch.PENALTY_AREA_WIDTH / 2.0, 0],
            dtype='float')
        self.left_penalty_area_bottom_right_corner = np.array(
            [-pitch_length / 2.0 + SoccerPitch.PENALTY_AREA_LENGTH, SoccerPitch.PENALTY_AREA_WIDTH / 2.0, 0],
            dtype='float')
        self.left_penalty_area_bottom_left_corner = np.array(
            [-pitch_length / 2.0, SoccerPitch.PENALTY_AREA_WIDTH / 2.0, 0],
            dtype='float')
        self.right_penalty_area_top_right_corner = np.array(
            [pitch_length / 2.0, -SoccerPitch.PENALTY_AREA_WIDTH / 2.0, 0],
            dtype='float')
        self.right_penalty_area_top_left_corner = np.array(
            [pitch_length / 2.0 - SoccerPitch.PENALTY_AREA_LENGTH, -SoccerPitch.PENALTY_AREA_WIDTH / 2.0, 0],
            dtype='float')
        self.right_penalty_area_bottom_right_corner = np.array(
            [pitch_length / 2.0, SoccerPitch.PENALTY_AREA_WIDTH / 2.0, 0],
            dtype='float')
        self.right_penalty_area_bottom_left_corner = np.array(
            [pitch_length / 2.0 - SoccerPitch.PENALTY_AREA_LENGTH, SoccerPitch.PENALTY_AREA_WIDTH / 2.0, 0],
            dtype='float')

        self.left_goal_area_top_right_corner = np.array(
            [-pitch_length / 2.0 + SoccerPitch.GOAL_AREA_LENGTH, -SoccerPitch.GOAL_AREA_WIDTH / 2.0, 0], dtype='float')
        self.left_goal_area_top_left_corner = np.array([-pitch_length / 2.0, - SoccerPitch.GOAL_AREA_WIDTH / 2.0, 0],
                                                       dtype='float')
        self.left_goal_area_bottom_right_corner = np.array(
            [-pitch_length / 2.0 + SoccerPitch.GOAL_AREA_LENGTH, SoccerPitch.GOAL_AREA_WIDTH / 2.0, 0], dtype='float')
        self.left_goal_area_bottom_left_corner = np.array([-pitch_length / 2.0, SoccerPitch.GOAL_AREA_WIDTH / 2.0, 0],
                                                          dtype='float')
        self.right_goal_area_top_right_corner = np.array([pitch_length / 2.0, -SoccerPitch.GOAL_AREA_WIDTH / 2.0, 0],
                                                         dtype='float')
        self.right_goal_area_top_left_corner = np.array(
            [pitch_length / 2.0 - SoccerPitch.GOAL_AREA_LENGTH, -SoccerPitch.GOAL_AREA_WIDTH / 2.0, 0], dtype='float')
        self.right_goal_area_bottom_right_corner = np.array([pitch_length / 2.0, SoccerPitch.GOAL_AREA_WIDTH / 2.0, 0],
                                                            dtype='float')
        self.right_goal_area_bottom_left_corner = np.array(
            [pitch_length / 2.0 - SoccerPitch.GOAL_AREA_LENGTH, SoccerPitch.GOAL_AREA_WIDTH / 2.0, 0], dtype='float')

        x = -pitch_length / 2.0 + SoccerPitch.PENALTY_AREA_LENGTH;
        dx = SoccerPitch.PENALTY_AREA_LENGTH - SoccerPitch.GOAL_LINE_TO_PENALTY_MARK;
        y = -np.sqrt(SoccerPitch.CENTER_CIRCLE_RADIUS * SoccerPitch.CENTER_CIRCLE_RADIUS - dx * dx);
        self.top_left_16M_penalty_arc_mark = np.array([x, y, 0], dtype='float')

        x = pitch_length / 2.0 - SoccerPitch.PENALTY_AREA_LENGTH;
        dx = SoccerPitch.PENALTY_AREA_LENGTH - SoccerPitch.GOAL_LINE_TO_PENALTY_MARK;
        y = -np.sqrt(SoccerPitch.CENTER_CIRCLE_RADIUS * SoccerPitch.CENTER_CIRCLE_RADIUS - dx * dx);
        self.top_right_16M_penalty_arc_mark = np.array([x, y, 0], dtype='float')

        x = -pitch_length / 2.0 + SoccerPitch.PENALTY_AREA_LENGTH;
        dx = SoccerPitch.PENALTY_AREA_LENGTH - SoccerPitch.GOAL_LINE_TO_PENALTY_MARK;
        y = np.sqrt(SoccerPitch.CENTER_CIRCLE_RADIUS * SoccerPitch.CENTER_CIRCLE_RADIUS - dx * dx);
        self.bottom_left_16M_penalty_arc_mark = np.array([x, y, 0], dtype='float')

        x = pitch_length / 2.0 - SoccerPitch.PENALTY_AREA_LENGTH;
        dx = SoccerPitch.PENALTY_AREA_LENGTH - SoccerPitch.GOAL_LINE_TO_PENALTY_MARK;
        y = np.sqrt(SoccerPitch.CENTER_CIRCLE_RADIUS * SoccerPitch.CENTER_CIRCLE_RADIUS - dx * dx);
        self.bottom_right_16M_penalty_arc_mark = np.array([x, y, 0], dtype='float')

        # self.set_elevations(elevation)

        self.point_dict = {}
        self.point_dict["CENTER_MARK"] = self.center_mark
        self.point_dict["L_PENALTY_MARK"] = self.left_penalty_mark
        self.point_dict["R_PENALTY_MARK"] = self.right_penalty_mark
        self.point_dict["TL_PITCH_CORNER"] = self.top_left_corner
        self.point_dict["BL_PITCH_CORNER"] = self.bottom_left_corner
        self.point_dict["TR_PITCH_CORNER"] = self.top_right_corner
        self.point_dict["BR_PITCH_CORNER"] = self.bottom_right_corner
        self.point_dict["L_PENALTY_AREA_TL_CORNER"] = self.left_penalty_area_top_left_corner
        self.point_dict["L_PENALTY_AREA_TR_CORNER"] = self.left_penalty_area_top_right_corner
        self.point_dict["L_PENALTY_AREA_BL_CORNER"] = self.left_penalty_area_bottom_left_corner
        self.point_dict["L_PENALTY_AREA_BR_CORNER"] = self.left_penalty_area_bottom_right_corner

        self.point_dict["R_PENALTY_AREA_TL_CORNER"] = self.right_penalty_area_top_left_corner
        self.point_dict["R_PENALTY_AREA_TR_CORNER"] = self.right_penalty_area_top_right_corner
        self.point_dict["R_PENALTY_AREA_BL_CORNER"] = self.right_penalty_area_bottom_left_corner
        self.point_dict["R_PENALTY_AREA_BR_CORNER"] = self.right_penalty_area_bottom_right_corner

        self.point_dict["L_GOAL_AREA_TL_CORNER"] = self.left_goal_area_top_left_corner
        self.point_dict["L_GOAL_AREA_TR_CORNER"] = self.left_goal_area_top_right_corner
        self.point_dict["L_GOAL_AREA_BL_CORNER"] = self.left_goal_area_bottom_left_corner
        self.point_dict["L_GOAL_AREA_BR_CORNER"] = self.left_goal_area_bottom_right_corner

        self.point_dict["R_GOAL_AREA_TL_CORNER"] = self.right_goal_area_top_left_corner
        self.point_dict["R_GOAL_AREA_TR_CORNER"] = self.right_goal_area_top_right_corner
        self.point_dict["R_GOAL_AREA_BL_CORNER"] = self.right_goal_area_bottom_left_corner
        self.point_dict["R_GOAL_AREA_BR_CORNER"] = self.right_goal_area_bottom_right_corner

        self.point_dict["L_GOAL_TL_POST"] = self.left_goal_top_left_post
        self.point_dict["L_GOAL_TR_POST"] = self.left_goal_top_right_post
        self.point_dict["L_GOAL_BL_POST"] = self.left_goal_bottom_left_post
        self.point_dict["L_GOAL_BR_POST"] = self.left_goal_bottom_right_post

        self.point_dict["R_GOAL_TL_POST"] = self.right_goal_top_left_post
        self.point_dict["R_GOAL_TR_POST"] = self.right_goal_top_right_post
        self.point_dict["R_GOAL_BL_POST"] = self.right_goal_bottom_left_post
        self.point_dict["R_GOAL_BR_POST"] = self.right_goal_bottom_right_post

        self.point_dict["T_TOUCH_AND_HALFWAY_LINES_INTERSECTION"] = self.halfway_and_top_touch_line_mark
        self.point_dict["B_TOUCH_AND_HALFWAY_LINES_INTERSECTION"] = self.halfway_and_bottom_touch_line_mark
        self.point_dict["T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION"] = self.halfway_line_and_center_circle_top_mark
        self.point_dict[
            "B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION"] = self.halfway_line_and_center_circle_bottom_mark
        self.point_dict["TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION"] = self.top_left_16M_penalty_arc_mark
        self.point_dict["BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION"] = self.bottom_left_16M_penalty_arc_mark
        self.point_dict["TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION"] = self.top_right_16M_penalty_arc_mark
        self.point_dict["BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION"] = self.bottom_right_16M_penalty_arc_mark

        self.line_extremities = dict()
        self.line_extremities["Big rect. left bottom"] = (self.point_dict["L_PENALTY_AREA_BL_CORNER"],
                                                          self.point_dict["L_PENALTY_AREA_BR_CORNER"])
        self.line_extremities["Big rect. left top"] = (self.point_dict["L_PENALTY_AREA_TL_CORNER"],
                                                       self.point_dict["L_PENALTY_AREA_TR_CORNER"])
        self.line_extremities["Big rect. left main"] = (self.point_dict["L_PENALTY_AREA_TR_CORNER"],
                                                        self.point_dict["L_PENALTY_AREA_BR_CORNER"])
        self.line_extremities["Big rect. right bottom"] = (self.point_dict["R_PENALTY_AREA_BL_CORNER"],
                                                           self.point_dict["R_PENALTY_AREA_BR_CORNER"])
        self.line_extremities["Big rect. right top"] = (self.point_dict["R_PENALTY_AREA_TL_CORNER"],
                                                        self.point_dict["R_PENALTY_AREA_TR_CORNER"])
        self.line_extremities["Big rect. right main"] = (self.point_dict["R_PENALTY_AREA_TL_CORNER"],
                                                         self.point_dict["R_PENALTY_AREA_BL_CORNER"])

        self.line_extremities["Small rect. left bottom"] = (self.point_dict["L_GOAL_AREA_BL_CORNER"],
                                                            self.point_dict["L_GOAL_AREA_BR_CORNER"])
        self.line_extremities["Small rect. left top"] = (self.point_dict["L_GOAL_AREA_TL_CORNER"],
                                                         self.point_dict["L_GOAL_AREA_TR_CORNER"])
        self.line_extremities["Small rect. left main"] = (self.point_dict["L_GOAL_AREA_TR_CORNER"],
                                                          self.point_dict["L_GOAL_AREA_BR_CORNER"])
        self.line_extremities["Small rect. right bottom"] = (self.point_dict["R_GOAL_AREA_BL_CORNER"],
                                                             self.point_dict["R_GOAL_AREA_BR_CORNER"])
        self.line_extremities["Small rect. right top"] = (self.point_dict["R_GOAL_AREA_TL_CORNER"],
                                                          self.point_dict["R_GOAL_AREA_TR_CORNER"])
        self.line_extremities["Small rect. right main"] = (self.point_dict["R_GOAL_AREA_TL_CORNER"],
                                                           self.point_dict["R_GOAL_AREA_BL_CORNER"])

        self.line_extremities["Side line top"] = (self.point_dict["TL_PITCH_CORNER"],
                                                  self.point_dict["TR_PITCH_CORNER"])
        self.line_extremities["Side line bottom"] = (self.point_dict["BL_PITCH_CORNER"],
                                                     self.point_dict["BR_PITCH_CORNER"])
        self.line_extremities["Side line left"] = (self.point_dict["TL_PITCH_CORNER"],
                                                   self.point_dict["BL_PITCH_CORNER"])
        self.line_extremities["Side line right"] = (self.point_dict["TR_PITCH_CORNER"],
                                                    self.point_dict["BR_PITCH_CORNER"])
        self.line_extremities["Middle line"] = (self.point_dict["T_TOUCH_AND_HALFWAY_LINES_INTERSECTION"],
                                                self.point_dict["B_TOUCH_AND_HALFWAY_LINES_INTERSECTION"])

        self.line_extremities["Goal left crossbar"] = (self.point_dict["L_GOAL_TR_POST"],
                                                       self.point_dict["L_GOAL_TL_POST"])
        self.line_extremities["Goal left post left "] = (self.point_dict["L_GOAL_TL_POST"],
                                                         self.point_dict["L_GOAL_BL_POST"])
        self.line_extremities["Goal left post right"] = (self.point_dict["L_GOAL_TR_POST"],
                                                         self.point_dict["L_GOAL_BR_POST"])

        self.line_extremities["Goal right crossbar"] = (self.point_dict["R_GOAL_TL_POST"],
                                                        self.point_dict["R_GOAL_TR_POST"])
        self.line_extremities["Goal right post left"] = (self.point_dict["R_GOAL_TL_POST"],
                                                         self.point_dict["R_GOAL_BL_POST"])
        self.line_extremities["Goal right post right"] = (self.point_dict["R_GOAL_TR_POST"],
                                                          self.point_dict["R_GOAL_BR_POST"])
        self.line_extremities["Circle right"] = (self.point_dict["TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION"],
                                                 self.point_dict["BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION"])
        self.line_extremities["Circle left"] = (self.point_dict["TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION"],
                                                self.point_dict["BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION"])

        self.line_extremities_keys = dict()
        self.line_extremities_keys["Big rect. left bottom"] = ("L_PENALTY_AREA_BL_CORNER",
                                                               "L_PENALTY_AREA_BR_CORNER")
        self.line_extremities_keys["Big rect. left top"] = ("L_PENALTY_AREA_TL_CORNER",
                                                            "L_PENALTY_AREA_TR_CORNER")
        self.line_extremities_keys["Big rect. left main"] = ("L_PENALTY_AREA_TR_CORNER",
                                                             "L_PENALTY_AREA_BR_CORNER")
        self.line_extremities_keys["Big rect. right bottom"] = ("R_PENALTY_AREA_BL_CORNER",
                                                                "R_PENALTY_AREA_BR_CORNER")
        self.line_extremities_keys["Big rect. right top"] = ("R_PENALTY_AREA_TL_CORNER",
                                                             "R_PENALTY_AREA_TR_CORNER")
        self.line_extremities_keys["Big rect. right main"] = ("R_PENALTY_AREA_TL_CORNER",
                                                              "R_PENALTY_AREA_BL_CORNER")

        self.line_extremities_keys["Small rect. left bottom"] = ("L_GOAL_AREA_BL_CORNER",
                                                                 "L_GOAL_AREA_BR_CORNER")
        self.line_extremities_keys["Small rect. left top"] = ("L_GOAL_AREA_TL_CORNER",
                                                              "L_GOAL_AREA_TR_CORNER")
        self.line_extremities_keys["Small rect. left main"] = ("L_GOAL_AREA_TR_CORNER",
                                                               "L_GOAL_AREA_BR_CORNER")
        self.line_extremities_keys["Small rect. right bottom"] = ("R_GOAL_AREA_BL_CORNER",
                                                                  "R_GOAL_AREA_BR_CORNER")
        self.line_extremities_keys["Small rect. right top"] = ("R_GOAL_AREA_TL_CORNER",
                                                               "R_GOAL_AREA_TR_CORNER")
        self.line_extremities_keys["Small rect. right main"] = ("R_GOAL_AREA_TL_CORNER",
                                                                "R_GOAL_AREA_BL_CORNER")

        self.line_extremities_keys["Side line top"] = ("TL_PITCH_CORNER",
                                                       "TR_PITCH_CORNER")
        self.line_extremities_keys["Side line bottom"] = ("BL_PITCH_CORNER",
                                                          "BR_PITCH_CORNER")
        self.line_extremities_keys["Side line left"] = ("TL_PITCH_CORNER",
                                                        "BL_PITCH_CORNER")
        self.line_extremities_keys["Side line right"] = ("TR_PITCH_CORNER",
                                                         "BR_PITCH_CORNER")
        self.line_extremities_keys["Middle line"] = ("T_TOUCH_AND_HALFWAY_LINES_INTERSECTION",
                                                     "B_TOUCH_AND_HALFWAY_LINES_INTERSECTION")

        self.line_extremities_keys["Goal left crossbar"] = ("L_GOAL_TR_POST",
                                                            "L_GOAL_TL_POST")
        self.line_extremities_keys["Goal left post left "] = ("L_GOAL_TL_POST",
                                                              "L_GOAL_BL_POST")
        self.line_extremities_keys["Goal left post right"] = ("L_GOAL_TR_POST",
                                                              "L_GOAL_BR_POST")

        self.line_extremities_keys["Goal right crossbar"] = ("R_GOAL_TL_POST",
                                                             "R_GOAL_TR_POST")
        self.line_extremities_keys["Goal right post left"] = ("R_GOAL_TL_POST",
                                                              "R_GOAL_BL_POST")
        self.line_extremities_keys["Goal right post right"] = ("R_GOAL_TR_POST",
                                                               "R_GOAL_BR_POST")
        self.line_extremities_keys["Circle right"] = ("TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
                                                      "BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION")
        self.line_extremities_keys["Circle left"] = ("TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
                                                     "BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION")

    def points(self):
        return [
            self.center_mark,
            self.halfway_and_bottom_touch_line_mark,
            self.halfway_and_top_touch_line_mark,
            self.halfway_line_and_center_circle_top_mark,
            self.halfway_line_and_center_circle_bottom_mark,
            self.bottom_right_corner,
            self.bottom_left_corner,
            self.top_right_corner,
            self.top_left_corner,
            self.left_penalty_mark,
            self.right_penalty_mark,
            self.left_penalty_area_top_right_corner,
            self.left_penalty_area_top_left_corner,
            self.left_penalty_area_bottom_right_corner,
            self.left_penalty_area_bottom_left_corner,
            self.right_penalty_area_top_right_corner,
            self.right_penalty_area_top_left_corner,
            self.right_penalty_area_bottom_right_corner,
            self.right_penalty_area_bottom_left_corner,
            self.left_goal_area_top_right_corner,
            self.left_goal_area_top_left_corner,
            self.left_goal_area_bottom_right_corner,
            self.left_goal_area_bottom_left_corner,
            self.right_goal_area_top_right_corner,
            self.right_goal_area_top_left_corner,
            self.right_goal_area_bottom_right_corner,
            self.right_goal_area_bottom_left_corner,
            self.top_left_16M_penalty_arc_mark,
            self.top_right_16M_penalty_arc_mark,
            self.bottom_left_16M_penalty_arc_mark,
            self.bottom_right_16M_penalty_arc_mark,
            self.left_goal_top_left_post,
            self.left_goal_top_right_post,
            self.left_goal_bottom_left_post,
            self.left_goal_bottom_right_post,
            self.right_goal_top_left_post,
            self.right_goal_top_right_post,
            self.right_goal_bottom_left_post,
            self.right_goal_bottom_right_post
        ]

    def sample_field_points(self, dist=0.1, dist_circles=0.2):
        """
        Samples each pitch element every dist meters, returns a dictionary associating the class of the element with a list of points sampled along this element.
        :param dist: the distance in meters between each point sampled
        :param dist_circles: the distance in meters between each point sampled on circles
        :return:  a dictionary associating the class of the element with a list of points sampled along this element.
        """
        polylines = dict()
        center = self.point_dict["CENTER_MARK"]
        fromAngle = 0.
        toAngle = 2 * np.pi

        if toAngle < fromAngle:
            toAngle += 2 * np.pi
        x1 = center[0] + np.cos(fromAngle) * SoccerPitch.CENTER_CIRCLE_RADIUS
        y1 = center[1] + np.sin(fromAngle) * SoccerPitch.CENTER_CIRCLE_RADIUS
        z1 = 0.
        point = np.array((x1, y1, z1))
        polyline = [point]
        length = SoccerPitch.CENTER_CIRCLE_RADIUS * (toAngle - fromAngle)
        nb_pts = int(length / dist_circles)
        dangle = dist_circles / SoccerPitch.CENTER_CIRCLE_RADIUS
        for i in range(1, nb_pts):
            angle = fromAngle + i * dangle
            x = center[0] + np.cos(angle) * SoccerPitch.CENTER_CIRCLE_RADIUS
            y = center[1] + np.sin(angle) * SoccerPitch.CENTER_CIRCLE_RADIUS
            z = 0
            point = np.array((x, y, z))
            polyline.append(point)
        polylines["Circle central"] = polyline
        for key, line in self.line_extremities.items():

            if "Circle" in key:
                if key == "Circle right":
                    top = self.point_dict["TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION"]
                    bottom = self.point_dict["BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION"]
                    center = self.point_dict["R_PENALTY_MARK"]
                    toAngle = np.arctan2(top[1] - center[1],
                                         top[0] - center[0]) + 2 * np.pi
                    fromAngle = np.arctan2(bottom[1] - center[1],
                                           bottom[0] - center[0]) + 2 * np.pi
                elif key == "Circle left":
                    top = self.point_dict["TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION"]
                    bottom = self.point_dict["BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION"]
                    center = self.point_dict["L_PENALTY_MARK"]
                    fromAngle = np.arctan2(top[1] - center[1],
                                           top[0] - center[0]) + 2 * np.pi
                    toAngle = np.arctan2(bottom[1] - center[1],
                                         bottom[0] - center[0]) + 2 * np.pi
                if toAngle < fromAngle:
                    toAngle += 2 * np.pi
                x1 = center[0] + np.cos(fromAngle) * SoccerPitch.CENTER_CIRCLE_RADIUS
                y1 = center[1] + np.sin(fromAngle) * SoccerPitch.CENTER_CIRCLE_RADIUS
                z1 = 0.
                xn = center[0] + np.cos(toAngle) * SoccerPitch.CENTER_CIRCLE_RADIUS
                yn = center[1] + np.sin(toAngle) * SoccerPitch.CENTER_CIRCLE_RADIUS
                zn = 0.
                start = np.array((x1, y1, z1))
                end = np.array((xn, yn, zn))
                polyline = [start]
                length = SoccerPitch.CENTER_CIRCLE_RADIUS * (toAngle - fromAngle)
                nb_pts = int(length / dist_circles)
                dangle = dist_circles / SoccerPitch.CENTER_CIRCLE_RADIUS
                for i in range(1, nb_pts + 1):
                    angle = fromAngle + i * dangle
                    x = center[0] + np.cos(angle) * SoccerPitch.CENTER_CIRCLE_RADIUS
                    y = center[1] + np.sin(angle) * SoccerPitch.CENTER_CIRCLE_RADIUS
                    z = 0
                    point = np.array((x, y, z))
                    polyline.append(point)
                polyline.append(end)
                polylines[key] = polyline
            else:
                start = line[0]
                end = line[1]

                polyline = [start]

                total_dist = np.sqrt(np.sum(np.square(start - end)))
                nb_pts = int(total_dist / dist - 1)

                v = end - start
                v /= np.linalg.norm(v)
                prev_pt = start
                for i in range(nb_pts):
                    pt = prev_pt + dist * v
                    prev_pt = pt
                    polyline.append(pt)
                polyline.append(end)
                polylines[key] = polyline
        return polylines

    def get_2d_homogeneous_line(self, line_name):
        """
        For lines belonging to the pitch lawn plane returns its 2D homogenous equation coefficients
        :param line_name
        :return: an array containing the three coefficients of the line
        """
        # ensure line in football pitch plane
        if line_name in self.line_extremities.keys() and \
                "post" not in line_name and \
                "crossbar" not in line_name and "Circle" not in line_name:
            extremities = self.line_extremities[line_name]
            p1 = np.array([extremities[0][0], extremities[0][1], 1], dtype="float")
            p2 = np.array([extremities[1][0], extremities[1][1], 1], dtype="float")
            line = np.cross(p1, p2)

            return line
        return None
