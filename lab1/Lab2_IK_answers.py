import numpy as np
from scipy.spatial.transform import Rotation as R
from answer import *

def FABR(meta_data, joint_positions, joint_orientations, target_pose):
    path_positions = []
    path_offsets = []
    path_orientations = []
    path_positions_old = []

    path, path_name, _, _ = meta_data.get_path_from_root_to_end()
    for joint in path:
        path_positions.append(joint_positions[joint])
        path_orientations.append(R.from_quat(joint_orientations[joint]))
    path_positions_old = path_positions.copy()
    length = []
    total_length = 0
    path_offsets.append(np.array([0., 0., 0.]))
    for i in range(len(path) - 1):#计算每个关节的局部offset
        offset = meta_data.joint_initial_position[path[i + 1]] - meta_data.joint_initial_position[path[i]]
        path_offsets.append(offset)
        length.append(np.linalg.norm(offset))
        total_length += np.linalg.norm(offset)
    begin_target_dis = target_pose - path_positions[0]

    if total_length <= np.linalg.norm(begin_target_dis):# cannot reach
        dir = begin_target_dis / np.linalg.norm(begin_target_dis)
        for i in range(len(path_positions)-1):
            path_positions[i+1] = path_positions[i] * dir * length[i]
    else:
        cnt = 0
        end_index = path_name.index(meta_data.end_joint)
        while (np.linalg.norm(path_positions[end_index] - target_pose) >= 1e-2 and cnt <= 10):#最大迭代次数10
            moveTo = target_pose
            start_pos = path_positions[0]
            #前向
            i = (len(path_positions) - 1)
            while (i > 0):
                path_positions[i] = moveTo
                dir = path_positions[i - 1] - path_positions[i]
                dir = dir / np.linalg.norm(dir)
                moveTo = path_positions[i] + dir * length[i-1]
                i -= 1
            #补上第一个位置
            path_positions[0] = moveTo
            moveTo = start_pos
            #后向
            i = 0
            while ( i < len(path_positions) - 1):
                path_positions[i] = moveTo
                dir = path_positions[i+1] - path_positions[i]
                dir = dir / np.linalg.norm(dir)
                moveTo = path_positions[i] + dir * length[i]
                i += 1
            path_positions[i] = moveTo
            cnt += 1
        print(cnt)
        #根据位置计算旋转
        path_rotations = []
        path_rotations.append(path_orientations[0])
        for j in range(len(path_orientations) - 1):
            path_rotations.append(R.inv(path_orientations[j]) * path_orientations[j + 1])#求出每个子关节相对于父关节的局部R

        for i in range(len(path_positions) - 1):
            new_dir = path_positions[i+1] - path_positions[i]
            new_dir = new_dir / np.linalg.norm(new_dir)
            old_dir = path_positions_old[i+1] - path_positions_old[i]
            old_dir = old_dir / np.linalg.norm(old_dir)
            # 计算轴角
            rotation_radius = np.arccos(np.clip(np.dot(old_dir,new_dir), -1, 1))
            temp_axis = np.cross(old_dir,new_dir)
            rotation_axis = temp_axis / np.linalg.norm(temp_axis)
            rotation_vector = R.from_rotvec(rotation_radius * rotation_axis)
            path_orientations[i] = rotation_vector * path_orientations[i]#旋转关节
            
            for j in range(i, len(path)-1):
                path_positions_old[j + 1] = path_positions_old[j] + path_orientations[j].apply(path_offsets[j + 1])#从current_index开始做accumulate
                if j + 1 < end_index:
                    path_orientations[j + 1] = path_orientations[j] * path_rotations[j + 1]#根据子关节旋转和父朝向计算子朝向
                else:
                    path_orientations[j + 1] = path_orientations[j]
    return path_positions, path_orientations

def CCD(meta_data, joint_positions, joint_orientations, target_pose):
    # 构建IK链
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    path_positions = []
    path_offsets = []
    path_orientations = []

    for joint in path:
        #计算每个关节的全局位置
        path_positions.append(joint_positions[joint])
    #Root have no offset
    path_offsets.append(np.array([0., 0., 0.]))
    for i in range(len(path) - 1):
        #计算每个关节的局部offset
        path_offsets.append(meta_data.joint_initial_position[path[i + 1]] - meta_data.joint_initial_position[path[i]])
    
    for i in range(len(path2) - 1):
        path_orientations.append(R.from_quat(joint_orientations[path2[i + 1]]))
    path_orientations.append(R.from_quat(joint_orientations[path2[-1]]))#加入root的朝向
    for i in range(len(path1) - 1):
        path_orientations.append(R.from_quat(joint_orientations[path1[~i]]))#反向遍历，这里相当于path顺序
    path_orientations.append(R.identity())
    #构建了起始姿势的关节位置 偏移和朝向

    # CCD 循环
    cnt = 0
    end_index = path_name.index(meta_data.end_joint)
    while (np.linalg.norm(joint_positions[path[end_index]] - target_pose) >= 1e-2 and cnt <= 10):#最大迭代次数10
        for i in range(end_index):
            current_index = end_index - i - 1#最后一个手部关节是不旋转的
            current_position = path_positions[current_index]
            end_position = path_positions[end_index]
            vector_current2end = end_position - current_position
            vector_current2target = target_pose - current_position
            current2end = vector_current2end / np.linalg.norm(vector_current2end)
            current2target = vector_current2target / np.linalg.norm(vector_current2target)

            # 计算轴角
            rotation_radius = np.arccos(np.clip(np.dot(current2end, current2target), -1, 1))
            temp_axis = np.cross(current2end, current2target)
            rotation_axis = temp_axis / np.linalg.norm(temp_axis)
            if current_index == 0:#root不要旋转
                rotation_radius = 0
            rotation_vector = R.from_rotvec(rotation_radius * rotation_axis)
            
            # 计算方位与位置
            path_orientations[current_index] = rotation_vector * path_orientations[current_index]#旋转关节
            path_rotations = []
            path_rotations.append(path_orientations[0])
            for j in range(len(path_orientations) - 1):
                path_rotations.append(R.inv(path_orientations[j]) * path_orientations[j + 1])#求出每个子关节相对于父关节的局部R
            for j in range(current_index, end_index):
                path_positions[j + 1] = path_positions[j] + path_orientations[j].apply(path_offsets[j + 1])#从current_index开始做accumulate
                if j + 1 < end_index:
                    path_orientations[j + 1] = path_orientations[j] * path_rotations[j + 1]#根据子关节旋转和父朝向计算子朝向
                else:
                    path_orientations[j + 1] = path_orientations[j]
        cnt += 1
    print(cnt)
    return path_positions, path_orientations

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    #path_positions, path_orientations = FABR(meta_data, joint_positions, joint_orientations, target_pose)
    #path_positions, path_orientations = CCD(meta_data, joint_positions, joint_orientations, target_pose)
    path_positions, path_orientations = dampedGaussNewtonMethod(meta_data, joint_positions, joint_orientations, target_pose)
    
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()#path1 endeffect到root前一个 path2 starteffect 到 root

    # 计算 path_joints 的旋转
    joint_rotations = R.identity(len(meta_data.joint_name))
    for i in range(len(meta_data.joint_parent)):
        if meta_data.joint_parent[i] == -1:
            joint_rotations[i] = R.from_quat(joint_orientations[i])
        else:
            joint_rotations[i] = R.inv(R.from_quat(joint_orientations[meta_data.joint_parent[i]])) * R.from_quat(joint_orientations[i])
    
    # 更新path_joints 的 朝向和位置 
    # 注意如果IK路径穿过根节点，path2的旋转要从子关节赋给父关节
    if 0 in path:
        for i in range(len(path2) - 1):
            joint_orientations[path2[i + 1]] = path_orientations[i].as_quat()
        joint_orientations[path2[-1]] = path_orientations[len(path2) - 1].as_quat()
        for i in range(len(path1) - 1):
            joint_orientations[path1[~i]] = path_orientations[i + len(path2)].as_quat()
    else:
        for i in range(len(path)):
            joint_orientations[path[i]] = path_orientations[i].as_quat()
            
    for i in range(len(path)):
        joint_positions[path[i]] = path_positions[i]
    # 其余 joints 的 forward_kinematics
    for i in range(len(meta_data.joint_parent)):
        if meta_data.joint_parent[i] == -1:
            continue
        if meta_data.joint_name[i] not in path_name:
            joint_positions[i] = joint_positions[meta_data.joint_parent[i]] + \
                R.from_quat(joint_orientations[meta_data.joint_parent[i]]).apply(meta_data.joint_initial_position[i] - \
                meta_data.joint_initial_position[meta_data.joint_parent[i]])
            joint_orientations[i] = (R.from_quat(joint_orientations[meta_data.joint_parent[i]]) * joint_rotations[i]).as_quat()
    
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移
    以及目标高度target_height
    """
    for i in range(len(meta_data.joint_parent)):
        if meta_data.joint_parent[i] == -1:
            root_idx = i
            break
    root_pos = joint_positions[root_idx]
    root_orientations = joint_orientations[root_idx]
    target_offsetsXZ = np.array([relative_x, 0., relative_z])
    target_pose = root_pos + R.from_quat(root_orientations).apply(target_offsetsXZ)
    target_pose[1,] = target_height
    joint_positions, joint_orientations = part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose)
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations