from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import copy


class Mobility(object):
    def __init__(self, side_length=500, velocity=5, num_clients_per_zone=10, num_zones=4, location=None):
        self.side_length = side_length
        self.velocity = velocity
        self.num_clients_per_zone = num_clients_per_zone
        self.num_zones = num_zones
        self.bs_locations = np.array([[-0.5 * side_length, 0.5 * side_length], [0.5 * side_length, 0.5 * side_length],
                                      [0.5 * side_length, -0.5 * side_length],
                                      [-0.5 * side_length, -0.5 * side_length]])
        if location is None:
            self.client_locations = self.initialize_client_locations()
        else:
            self.client_locations = location
        self.previous_location = copy.deepcopy(self.client_locations)
        self.dist, self.zone = self.get_distance_and_zone()
        self.previous_zone = copy.deepcopy(self.zone)
        self.zone_client_id = self.get_zone_client_id()
        self.channel_gain = self.get_channel_gain()
        self.cross = (np.array(self.zone) == np.array(self.previous_zone))
        self.time=0
    def initialize_client_locations(self):
        client_locations = np.zeros((2, self.num_clients_per_zone * self.num_zones))

        for i in range(self.num_zones):
            x = np.random.uniform(self.bs_locations[i, 0] - 0.5 * self.side_length,
                                  self.bs_locations[i, 0] + 0.5 * self.side_length, self.num_clients_per_zone)
            y = np.random.uniform(self.bs_locations[i, 1] - 0.5 * self.side_length,
                                  self.bs_locations[i, 1] + 0.5 * self.side_length, self.num_clients_per_zone)
            client_locations[:, i * self.num_clients_per_zone: (i + 1) * self.num_clients_per_zone] = np.array([x, y])

        return client_locations

    def move(self, time):
        self.time=time
        self.previous_location = copy.deepcopy(self.client_locations)
        self.previous_zone = copy.deepcopy(self.zone)
        temp = np.random.random(self.num_clients_per_zone * self.num_zones) * np.pi * 2
        self.client_locations[0] = self.client_locations[0] + np.cos(temp) * self.velocity * time
        self.client_locations[1] = self.client_locations[1] + np.sin(temp) * self.velocity * time
        self.check_boundary()
        self.get_distance_and_zone()
        self.get_channel_gain()
        self.get_zone_client_id()
        self.cross = (np.array(self.zone) == np.array(self.previous_zone))

    def check_boundary(self):
        self.client_locations[0][np.where(self.client_locations[0] < -self.side_length)] = -2 * self.side_length - \
                                                                                           self.client_locations[0][
                                                                                               np.where(
                                                                                                   self.client_locations[
                                                                                                       0] < -self.side_length)]
        self.client_locations[1][np.where(self.client_locations[1] < -self.side_length)] = -2 * self.side_length - \
                                                                                           self.client_locations[1][
                                                                                               np.where(
                                                                                                   self.client_locations[
                                                                                                       1] < -self.side_length)]
        self.client_locations[0][np.where(self.client_locations[0] > self.side_length)] = 2 * self.side_length - \
                                                                                          self.client_locations[0][
                                                                                              np.where(
                                                                                                  self.client_locations[
                                                                                                      0] > self.side_length)]
        self.client_locations[1][np.where(self.client_locations[1] > self.side_length)] = 2 * self.side_length - \
                                                                                          self.client_locations[1][
                                                                                              np.where(
                                                                                                  self.client_locations[
                                                                                                      1] > self.side_length)]

    def get_distance_and_zone(self):
        dis = np.zeros([self.num_zones, self.num_clients_per_zone * self.num_zones])
        for i in range(self.num_zones):
            for j in range(self.num_clients_per_zone * self.num_zones):
                dis[i, j] = np.sqrt((self.client_locations[0, j] - self.bs_locations[i, 0]) ** 2 + (
                        self.client_locations[1, j] - self.bs_locations[i, 1]) ** 2)
        self.dist = np.min(dis, axis=0)
        self.zone = np.argmin(dis, axis=0)
        return self.dist, self.zone

    def get_zone_client_id(self):
        zone_client_id = [[] for _ in range(self.num_zones)]
        for i, z in enumerate(self.zone):
            zone_client_id[z].append(i)
        self.zone_client_id = zone_client_id
        return self.zone_client_id

    def get_channel_gain(self):
        distance, _ = self.get_distance_and_zone()
        h = np.random.rayleigh(1, self.num_clients_per_zone * self.num_zones)
        h = np.power(h, 2)
        # loss_db=np.log10(self.distance)*37.6+128.1
        loss = np.power(1000 / distance, 3.76) * 1.549e-13
        #修改了噪声和路径损耗
        self.channel_gain = h * loss
        return self.channel_gain

    def cal_cross_time(self):  # 这里要注意默认是一秒时间间隔
        cross = (np.sign(self.client_locations) == np.sign(self.previous_location))
        #print(cross)
        cross_list = []
        time_list = []
        for i in range(self.num_clients_per_zone * self.num_zones):
            if cross[0, i] == 0 & cross[1, i] == 0:
                cross_list.append(i)
                time1 = abs(self.previous_location[1, i]) / (
                        abs(self.previous_location[1, i]) + abs(self.client_locations[1, i]))
                time2 = abs(self.previous_location[0, i]) / (
                        abs(self.previous_location[0, i]) + abs(self.client_locations[0, i]))
                time_list.append(min(time1, time2))
                continue
            if cross[0, i] == 0:
                cross_list.append(i)
                time = abs(self.previous_location[0, i]) / (
                        abs(self.previous_location[0, i]) + abs(self.client_locations[0, i]))
                time_list.append(time)
            if cross[1, i] == 0:
                cross_list.append(i)
                time = abs(self.previous_location[1, i]) / (
                        abs(self.previous_location[1, i]) + abs(self.client_locations[1, i]))
                time_list.append(time)
        time_list=[j*self.time for j in time_list]
        return cross_list, time_list

    def plot_locations(self):
        plt.figure(figsize=(8, 8))

        # Plot base stations
        plt.scatter(self.bs_locations[:, 0], self.bs_locations[:, 1], marker='s', color='red', label='Base Station')

        # Plot client locations
        plt.scatter(self.client_locations[0], self.client_locations[1], marker='o', color='blue', label='Client')

        # Set plot limits and labels
        plt.xlim(-1.5 * self.side_length, 1.5 * self.side_length)
        plt.ylim(-1.5 * self.side_length, 1.5 * self.side_length)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        # Add legend
        plt.legend()

        # Show the plot
        plt.show()


if __name__ == '__main__':
    from fedavg import load_pkl, save_pkl

    v = 20
    t = 1
    round=2000
    Mob_ = load_pkl('data/mob_{}_{}_{}.pkl'.format(100, 1000,5))
    Mob = Mobility(500, v, 10, 4, Mob_[0].client_locations)
    Mob_list = []
    Mob_list.append(copy.deepcopy(Mob))
    for i in range(round):
        Mob.move(t)
        Mob_list.append(copy.deepcopy(Mob))
    save_pkl(Mob_list, 'data', 'mob_{}_{}_{}.pkl'.format(v, round,t))


