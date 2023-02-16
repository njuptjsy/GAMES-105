import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    with open(bvh_file_path, 'r') as f:
        channels = []
        joints = []
        joint_parents = []
        joint_offsets = []
        end_sites = []

        parent_stack = [None]
        for line in f:
            if 'ROOT' in line or 'JOINT' in line:
                joints.append(line.split()[-1])
                joint_parents.append(parent_stack[-1])
                channels.append('')
                joint_offsets.append([0, 0, 0])

            elif 'End Site' in line:
                end_sites.append(len(joints))
                joints.append(parent_stack[-1] + '_end')
                joint_parents.append(parent_stack[-1])
                channels.append('')
                joint_offsets.append([0, 0, 0])

            elif '{' in line:
                parent_stack.append(joints[-1])

            elif '}' in line:
                parent_stack.pop()

            elif 'OFFSET' in line:
                joint_offsets[-1] = np.array([float(x) for x in line.split()[-3:]]).reshape(1,3)

            elif 'CHANNELS' in line:
                trans_order = []
                rot_order = []
                for token in line.split():
                    if 'position' in token:
                        trans_order.append(token[0])

                    if 'rotation' in token:
                        rot_order.append(token[0])

                channels[-1] = ''.join(trans_order)+ ''.join(rot_order)

            elif 'Frame Time:' in line:
                break
        
    joint_parents = [-1]+ [joints.index(i) for i in joint_parents[1:]]
    channels = [len(i) for i in channels]
    return joints, joint_parents, joint_offsets


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    
    # 把motion_data里的数据分配到joint_position和joint_rotation里
    motion_data_frame = motion_data[frame_id]
    joint_positions = np.zeros((len(joint_name), 3))
    joint_orientations = np.zeros((len(joint_name), 4))
    joint_orientations[:,3] = 1.0 # 四元数的w分量默认为1

    cur_channel = 0
    next_channe = 0
    for i in range(len(joint_name)):
        if '_end' in joint_name[i]:
            joint_positions[i,:] = joint_offset[i].reshape(1,3)
            continue  
        elif 'Root' in joint_name[i]:
            joint_positions[i, :] = motion_data_frame[cur_channel:cur_channel+3]
            rotation = motion_data_frame[cur_channel+3:cur_channel+6] 
            next_channe = 6
        else:
            joint_positions[i,:] = joint_offset[i].reshape(1,3)
            rotation = motion_data_frame[cur_channel:cur_channel+3]
            next_channe = 3
        joint_orientations[i, :] = R.from_euler('XYZ', rotation,degrees=True).as_quat()
        cur_channel += next_channe
    
    # 从局部坐标到全局
    joint_translation = np.zeros_like(joint_positions)
    joint_orientation = np.zeros_like(joint_orientations)
    joint_orientation[:,3] = 1.0 # 四元数的w分量默认为1
        
    # 一个小hack是root joint的parent是-1, 对应最后一个关节
    # 计算根节点时最后一个关节还未被计算，刚好是0偏移和单位朝向
        
    for i in range(len(joint_name)):
        pi = joint_parent[i]
        parent_orientation = R.from_quat(joint_orientation[pi,:]) 
        joint_translation[i, :] = joint_translation[pi, :] + parent_orientation.apply(joint_positions[i, :])
        joint_orientation[i, :] = (parent_orientation * R.from_quat(joint_orientations[i, :])).as_quat()
    return joint_translation, joint_orientation

def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    
    joint_nameT, joint_parentA, joint_offsetA = part1_calculate_T_pose(T_pose_bvh_path)
    joint_nameA, joint_parentT, joint_offsetT = part1_calculate_T_pose(A_pose_bvh_path)
    motion_dataA = load_motion_data(A_pose_bvh_path)
    motion_data = np.zeros_like(motion_dataA)
    
    cur_channelT = 0
    for i in range(len(joint_nameT)):
        if '_end' in joint_nameT[i]:#channel = 0 
            continue
        cur_channelA = 0
        for j in range(len(joint_nameA)):
            if joint_nameT[i] == joint_nameA[j]:
                if 'Root' in joint_nameA[j]:
                    motion_data[:, cur_channelT:cur_channelT+6] = motion_dataA[:, cur_channelA:cur_channelA+6]
                    cur_channelT += 6
                    cur_channelA += 6
                    break
                else:
                    motion_data[:, cur_channelT:cur_channelT+3] = motion_dataA[:, cur_channelA:cur_channelA+3]
                    if 'lShoulder' in joint_nameA[j]:
                        motion_data[:, cur_channelT+2] += -45
                    elif 'rShoulder' in joint_nameA[j]:
                        motion_data[:, cur_channelT+2] += 45
                    cur_channelT += 3
                    cur_channelA += 3
                    break
            else:
                if '_end' in joint_nameA[j]:#channel = 0 
                    continue
                elif 'Root' in joint_nameA[j]:
                    cur_channelA += 6
                    continue
                else:
                    cur_channelA += 3
                    continue
    return motion_data
