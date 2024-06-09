import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon

class EvoluteFrame:
    def __init__(self,
                 x=None, y=None, w=None, h=None, theta=None):
        self.x = x if x is not None else np.random.uniform(0, 10)
        self.y = y if y is not None else np.random.uniform(0, 10)
        self.w = w if w is not None else np.random.uniform(0, 10)
        self.h = h if h is not None else np.random.uniform(0, 10)
        self.theta = theta if theta is not None else np.random.uniform(0, 2 * np.pi)
        self.position = self.__get_position()
        self.path = Path(self.position)

        self.sigma_x = 0.5
        self.sigma_y = 0.5
        self.sigma_w = 0.5
        self.sigma_h = 0.5
        self.sigma_theta = 0.5

    def __get_position(self):
        '''
        self.x, self.y is the center,
        self.w is the width, self.h is the height,
        and the theta is the rotation.
        return the 4 points for the rectangle
        '''
        wc = self.w*np.cos(self.theta)/2
        ws = self.w*np.sin(self.theta)/2
        hc = self.h*np.cos(self.theta)/2
        hs = self.h*np.sin(self.theta)/2

        x1 = self.x - wc - hs
        y1 = self.y + ws - hc

        x2 = self.x - wc + hs
        y2 = self.y + ws + hc

        x3 = self.x + wc + hs
        y3 = self.y - ws + hc

        x4 = self.x + wc - hs
        y4 = self.y - ws - hc

        return np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    def update_position(self):
        self.position = self.__get_position()
        self.path = Path(self.position)
    def get_position(self):
        return self.position
    def is_in_rect(self, point):
        return self.path.contains_point(point)

    def calculate_score(self, points, flag):
        '''
        if the points are in the rectangle, return 1,
        else return the 1/d where d is the distance from the point to the center.
        '''
        position = self.get_position()
        score = 0
        for i in range(len(points)):
            if self.is_in_rect(points[i]):
                if flag[i] == 0:
                    flag[i] = 1
                    score += 1/np.linalg.norm(points[i] - [self.x, self.y])
            else:
                flag[i] = 0
            # else:
            #     score +=
        return score

    # 重写打印函数，打印的时候打印x,y,w,h,theta
    def __str__(self):
        return f'x: {self.x}, y: {self.y}, w: {self.w}, h: {self.h}, theta: {self.theta}'

    def generate_offsprint(self):
        x_star = np.random.normal(self.x, self.sigma_x)
        y_star = np.random.normal(self.y, self.sigma_y)
        w_star = np.random.normal(self.w, self.sigma_w)
        h_star = np.random.normal(self.h, self.sigma_h)
        theta_star = np.random.normal(self.theta, self.sigma_theta)
        return EvoluteFrame(x_star, y_star, w_star, h_star, theta_star)

class Boxes:
    def __init__(self, num_clusers, points, flags, boxes = None):
        self.boxes = boxes if boxes is not None else [EvoluteFrame() for _ in range(num_clusers)]
        self.score = self.calculate_score(points, flags)
        # for b in self.boxes:
        #     print(b)


    def calculate_score(self, points, flags):
        self.score = np.sum([box.calculate_score(points, flags) for box in self.boxes])
        return self.score

    def generate_offspring(self, num_cluster, points, flags):
        new_boxes = []
        for b in self.boxes:
            b = b.generate_offsprint()
            new_boxes.append(b)
        return Boxes(num_cluster, points, flags, new_boxes)




# 第1组数据：（2.5，7.5）为圆心生成半径为2.0的分布的圆内的随机点，生成33个
theta2 = np.random.uniform(0, 2 * np.pi, 33)
r2 = 2.0 * np.sqrt(np.random.uniform(0, 1, 33))
x2 = 2.5 + r2 * np.cos(theta2)
y2 = 7.5 + r2 * np.sin(theta2)
group2 = np.vstack((x2, y2)).T

# 第2组数据：（7.5，2.5）为中心生成半径为2.0的圆，里面分布随机点33个
theta3 = np.random.uniform(0, 2 * np.pi, 33)
r3 = 2.0 * np.sqrt(np.random.uniform(0, 1, 33))
x3 = 7.5 + r3 * np.cos(theta3)
y3 = 2.5 + r3 * np.sin(theta3)
group3 = np.vstack((x3, y3)).T

# 合并所有数据点
points = np.vstack((group2, group3))
flag = np.zeros(66)

# 进化聚类算法（修改后的版本）
def evolving_clusters(points, num_clusters, num_generations=500):
    total_number = 100
    parent_number = 50
    hyperboxes = [Boxes(num_clusters, points, flag.copy()) for _ in range(parent_number)]
    # 对hyperboxes排序。分数最高的排在前面。
    hyperboxes.sort(key=lambda x: x.score, reverse=True)
    best_fitness = hyperboxes[0].score
    plt.figure()
    # 可视化结果
    plt.scatter(points[:, 0], points[:, 1], c='blue', marker='x')
    for hyperbox in hyperboxes[0].boxes:
        polygon = Polygon(hyperbox.get_position(), closed=True, fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(polygon)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Evolving Clusters with Rotatable Hyperboxes init')
    plt.show()

    for generation in range(num_generations):

        for i in range(total_number):
            hyperboxes.append(hyperboxes[i].generate_offspring(num_clusters, points, flag.copy()))
        hyperboxes.sort(key=lambda x: x.score, reverse=True)
        hyperboxes = hyperboxes[:parent_number]


    return hyperboxes[0].boxes


# 其余代码保持不变


# 变异超矩形
def mutate_hyperbox(hyperbox):
    x_min, y_min, x_max, y_max = hyperbox
    x_min += np.random.uniform(-0.5, 0.5)
    y_min += np.random.uniform(-0.5, 0.5)
    x_max += np.random.uniform(-0.5, 0.5)
    y_max += np.random.uniform(-0.5, 0.5)
    return [x_min, y_min, x_max, y_max]

# 使用进化聚类算法
num_clusters = 2
hyperboxes = evolving_clusters(points, num_clusters)

# 可视化结果
plt.scatter(points[:, 0], points[:, 1], c='blue', marker='x')
plt.figure()
# 可视化结果
plt.scatter(points[:, 0], points[:, 1], c='blue', marker='x')
for hyperbox in hyperboxes:
    polygon = Polygon(hyperbox.get_position(), closed=True, fill=False, edgecolor='red', linewidth=2)
    plt.gca().add_patch(polygon)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Evolving Clusters with Rotatable Hyperboxes Final')
plt.show()
