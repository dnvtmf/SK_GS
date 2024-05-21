import math

import numpy as np
import trimesh


def sample_points_from_mesh(vertices: np.ndarray, triangles: np.ndarray, num_of_points: int):
    mesh = trimesh.Trimesh(vertices, triangles)
    points, index = mesh.sample(num_of_points, return_index=True)
    normals = mesh.face_normals[index]
    return points, normals


def sample_points_from_mesh2(vertices: np.ndarray, triangles: np.ndarray, num_of_points: int) -> np.ndarray:
    epsilon = 1e-6
    triangle_area_list = np.zeros([len(triangles)], np.float32)
    triangle_normal_list = np.zeros([len(triangles), 3], np.float32)
    for i in range(len(triangles)):
        # area = |u x v|/2 = |u||v|sin(uv)/2
        a, b, c = vertices[triangles[i, 1]] - vertices[triangles[i, 0]]
        x, y, z = vertices[triangles[i, 2]] - vertices[triangles[i, 0]]
        ti = b * z - c * y
        tj = c * x - a * z
        tk = a * y - b * x
        area2 = math.sqrt(ti * ti + tj * tj + tk * tk)
        if area2 < epsilon:
            triangle_area_list[i] = 0
            triangle_normal_list[i, 0] = 0
            triangle_normal_list[i, 1] = 0
            triangle_normal_list[i, 2] = 0
        else:
            triangle_area_list[i] = area2
            triangle_normal_list[i, 0] = ti / area2
            triangle_normal_list[i, 1] = tj / area2
            triangle_normal_list[i, 2] = tk / area2

    triangle_area_sum = np.sum(triangle_area_list)
    sample_prob_list = (num_of_points / triangle_area_sum) * triangle_area_list

    triangle_index_list = np.arange(len(triangles))

    point_normal_list = np.zeros([num_of_points, 6], np.float32)
    count = 0
    watchdog = 0

    while count < num_of_points:
        np.random.shuffle(triangle_index_list)
        watchdog += 1
        if watchdog > 100:
            print("infinite loop here!")
            return point_normal_list
        for i in range(len(triangle_index_list)):
            if count >= num_of_points:
                break
            dxb = triangle_index_list[i]
            prob = sample_prob_list[dxb]
            prob_i = int(prob)
            prob_f = prob - prob_i
            if np.random.random() < prob_f:
                prob_i += 1
            normal_direction = triangle_normal_list[dxb]
            u = vertices[triangles[dxb, 1]] - vertices[triangles[dxb, 0]]
            v = vertices[triangles[dxb, 2]] - vertices[triangles[dxb, 0]]
            base = vertices[triangles[dxb, 0]]
            for j in range(prob_i):
                # sample a point here:
                u_x = np.random.random()
                v_y = np.random.random()
                if u_x + v_y >= 1:
                    u_x = 1 - u_x
                    v_y = 1 - v_y
                ppp = u * u_x + v * v_y + base

                point_normal_list[count, :3] = ppp
                point_normal_list[count, 3:] = normal_direction
                count += 1
                if count >= num_of_points:
                    break

    return point_normal_list
