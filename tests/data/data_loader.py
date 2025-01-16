import numpy as np
import open3d as o3d
import time
import cv2, os, sys
import json, pickle, re

class EpisodeLoader:
    def __init__(self, path="GraspCube_H1-2_Inspire_20241115", i_episode=0):
        scriptdir = os.path.split(os.path.realpath(sys.argv[0]))[0]
        self.path = os.path.join(scriptdir, path)
        #with open(os.path.join(self.path, 'config.json'), 'r') as f:
        #    config = json.load(f)
        with open(os.path.join(self.path, 'episode_{}.pkl'.format(i_episode)), 'rb') as f:
            self.data = pickle.load(f)
        
        fns = os.listdir(self.path)
        for f in fns:
            if f.startswith('episode_{}.observation.state'.format(i_episode)):
                state_path = os.path.join(self.path, f)
                with open(state_path, 'rb') as file:
                    k = re.search(r'\.(.+)\.', f).group(1)
                    self.data['observation.state'] = pickle.load(file)
            elif f.startswith('episode_{}.observation.image.camera'.format(i_episode)):
                rgb_path = os.path.join(self.path, f)
                rgb_frames = self.load_rgb_video(rgb_path)
                k = re.search(r'\.(.+)\.', f).group(1)
                self.data[k] = rgb_frames
            elif f.startswith('episode_{}.observation.depth.camera'.format(i_episode)):
                depth_path = os.path.join(self.path, f)
                depth_frames = self.load_depth_video(depth_path)
                k = re.search(r'\.(.+)\.', f).group(1)
                self.data[k] = depth_frames
            elif f.startswith('episode_{}.observation.point_cloud'.format(i_episode)):
                continue
        
        self.T = len(self.data['timestep'])
        
    def get_data(self, t):
        ret = {}
        for k in self.data:
            if k!='score' and k!='human_data':
                #print(k)
                ret[k] = self.data[k][t]
        return ret

    def __len__(self):
        return self.T

    def load_rgb_video(self, input_path):
        # 打开视频文件
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件：{input_path}")
    
        frames = []  # 用于存储 RGB 帧

        while True:
            ret, frame = cap.read()  # 逐帧读取视频
            if not ret:
                break  # 当没有更多帧时退出循环
        
            # OpenCV 默认读取 BGR 图像，将其转换为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
        cap.release()  # 释放视频资源
        return frames

    # mp4的数值范围是0-255，原始深度范围是0-1500mm
    def load_depth_video(self, input_path, max_depth=1500):
        # 打开视频文件
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件：{input_path}")
    
        depth_frames = []  # 存储深度帧序列

        while True:
            ret, frame = cap.read()  # 读取一帧
            if not ret:
                break
        
           # 检查是否是灰度图
            if len(frame.shape) == 3 and frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        
            # 将 0-255 的值映射回原始深度值
            frame_depth = (frame.astype(np.float32) * max_depth / 255).astype(np.float32)
            depth_frames.append(frame_depth)
    
        cap.release()  # 释放资源
        return depth_frames


    def get_pcd_from_rgbd(self, color_image, depth_image, pinhole_camera_intrinsic):
        #print(depth_image.shape, color_image.shape)
        depth = o3d.geometry.Image(depth_image)
        color = o3d.geometry.Image(color_image)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
        return pcd

    def visualize_point_cloud_sequence(self, interval=0.1):
        # 创建一个可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)
    
        pcd = o3d.geometry.PointCloud()  # 创建一个空的点云
        vis.add_geometry(pcd)

        # 设置观察方向
        view_control = vis.get_view_control()
        view_control.set_lookat([-0.1, -0.1, 0.3])  # 观察点
        view_control.set_front([0, 0, -1])  # 前视方向Z
        view_control.set_up([0, -1, 0])  # 上方向设置-Y
        view_control.set_zoom(1)  # 设置缩放比例以适应场景
    
        from tqdm import trange
        for t in trange(100000):
            i = t % len(self.rgb_frames)
            new_pcd = self.get_pcd_from_rgbd(self.rgb_frames[i], self.depth_frames[i], self.pinhole_camera_intrinsic)
            pcd.points = new_pcd.points
            pcd.colors = new_pcd.colors
            vis.update_geometry(pcd)  # 更新点云
            vis.poll_events()         # 处理事件
            vis.update_renderer()     # 更新渲染器
            time.sleep(interval)      # 控制帧间隔
    
        vis.destroy_window()  # 销毁窗口
    

if __name__=='__main__':
    episode = EpisodeLoader(path="PolePickPlace_H1-2_Inspire_20241117", i_episode=40)
    #d = (episode.get_data(0))
    #for k in d:
    #    print(k, d[k].shape if hasattr(d[k], 'shape') else (d[k]))
    for k in episode.data:
        if k!='score' and k!='human_data':
            print(k, len(episode.data[k]), episode.data[k][0].shape)
    #print(episode.data['observation.last_action'] - episode.data['action'])
    print(np.mean(np.abs(episode.data['action'][0:len(episode.data['observation.state'])][0:] \
                         - episode.data['observation.state'][:]) , axis=0))
    #print(episode.data['observation.state'])
    #print(episode.data['action'])