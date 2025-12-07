#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# 使用 ROS2 Humble 自带的消息类型
typestore = get_typestore(Stores.ROS2_HUMBLE)


def read_joint_imu_bag(bag_path: str, joint_num: int = 12):
    """
    从 rosbag2 中读取 12 个关节的电机数据以及 IMU 数据

    Args:
        bag_path (str): rosbag2 目录路径（包含 metadata.yaml）
        joint_num (int): 关节数量，默认 12

    Returns:
        dict: {
          "joint": {
             "time": (N_joint,),
             "names": list[str],
             "position": (N_joint, joint_num),
             "velocity": (N_joint, joint_num),
             "effort": (N_joint, joint_num),
          },
          "imu": {
             "time": (N_imu,),
             "orientation": (N_imu, 4),
             "angular_velocity": (N_imu, 3),
             "linear_acceleration": (N_imu, 3),
          }
        }
    """
    bag_path = Path(bag_path)

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        # 选出两个话题
        joint_conns = [c for c in reader.connections if c.topic == "/joint_states"]
        imu_conns = [
            c for c in reader.connections
            if c.topic == "/imu_state_broadcaster/imu"
        ]

        if not joint_conns:
            raise RuntimeError("在 bag 中没有找到 /joint_states 话题")
        if not imu_conns:
            raise RuntimeError("在 bag 中没有找到 /imu_state_broadcaster/imu 话题")

        joint_conn = joint_conns[0]
        imu_conn = imu_conns[0]

        # ----------------- 先读取关节数据 -----------------
        joint_len = joint_conn.msgcount
        print(f"Number of /joint_states messages: {joint_len}")

        joint_time = np.zeros(joint_len, dtype=np.float64)
        joint_pos = np.zeros((joint_len, joint_num), dtype=np.float64)
        joint_vel = np.zeros((joint_len, joint_num), dtype=np.float64)
        joint_eff = np.zeros((joint_len, joint_num), dtype=np.float64)
        joint_names = None
        start_time_joint = None

        for i, (_, timestamp, rawdata) in enumerate(reader.messages([joint_conn])):
            msg = reader.deserialize(rawdata, joint_conn.msgtype)

            if i == 0:
                # name 即为 12 个关节的名字
                joint_names = list(msg.name)
                if len(joint_names) != joint_num:
                    raise ValueError(
                        f"期望 {joint_num} 个关节，但 JointState 中有 {len(joint_names)} 个"
                    )
                start_time_joint = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            joint_time[i] = t - start_time_joint

            # position / velocity / effort 都是长度为 joint_num 的数组
            if len(msg.position) >= joint_num:
                joint_pos[i, :] = msg.position[:joint_num]
            if len(msg.velocity) >= joint_num:
                joint_vel[i, :] = msg.velocity[:joint_num]
            if len(msg.effort) >= joint_num:
                joint_eff[i, :] = msg.effort[:joint_num]

        # ----------------- 再读取 IMU 数据 -----------------
        imu_len = imu_conn.msgcount
        print(f"Number of /imu_state_broadcaster/imu messages: {imu_len}")

        imu_time = np.zeros(imu_len, dtype=np.float64)
        imu_orientation = np.zeros((imu_len, 4), dtype=np.float64)       # x, y, z, w
        imu_ang_vel = np.zeros((imu_len, 3), dtype=np.float64)           # wx, wy, wz
        imu_lin_acc = np.zeros((imu_len, 3), dtype=np.float64)           # ax, ay, az
        start_time_imu = None

        for i, (_, timestamp, rawdata) in enumerate(reader.messages([imu_conn])):
            msg = reader.deserialize(rawdata, imu_conn.msgtype)

            if i == 0:
                start_time_imu = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            imu_time[i] = t - start_time_imu

            q = msg.orientation
            imu_orientation[i, :] = [q.x, q.y, q.z, q.w]

            av = msg.angular_velocity
            imu_ang_vel[i, :] = [av.x, av.y, av.z]

            la = msg.linear_acceleration
            imu_lin_acc[i, :] = [la.x, la.y, la.z]

    data = {
        "joint": {
            "time": joint_time,
            "names": joint_names,
            "position": joint_pos,
            "velocity": joint_vel,
            "effort": joint_eff,
        },
        "imu": {
            "time": imu_time,
            "orientation": imu_orientation,
            "angular_velocity": imu_ang_vel,
            "linear_acceleration": imu_lin_acc,
        },
    }
    return data


if __name__ == "__main__":
    bag_dir = "/home/mjy/下载/walk_data_1"  
    data = read_joint_imu_bag(bag_dir, joint_num=12)
    
    print("关节名称:", data["joint"]["names"])

    print("\n--- Joint Data (First 5) ---")
    print("Time:\n", data["joint"]["time"][:5])
    print("Position:\n", data["joint"]["position"][:5])
    print("Velocity:\n", data["joint"]["velocity"][:5])
    print("Effort:\n", data["joint"]["effort"][:5])

    print("\n--- IMU Data (First 5) ---")
    print("Time:\n", data["imu"]["time"][:5])
    print("Orientation:\n", data["imu"]["orientation"][:5])