import numpy as np
import copy


class scheduler(object):
    def __init__(self, args, datasize, tmax, mob, Qn):
        self.args = args
        self.datasize = datasize
        self.tmax = tmax
        self.mob = mob
        self.Qn = Qn

    def random(self):
        # 随机选择,最优能量
        idxs = []
        for i in range(self.args.num_edges):
            sample_size = min(self.args.channel_num, len(self.mob.zone_client_id[i]))
            idxs = idxs + list(np.random.choice(self.mob.zone_client_id[i], sample_size, replace=False))
        energy = [(self.cal_min_rt(idx) + self.Qn[idx]) / self.args.lyp_v for idx in idxs]
        idxs = [idx for idx, e in zip(idxs, energy) if not np.isinf(e)]
        energy = [e for e in energy if not np.isinf(e)]
        return idxs, energy
    def rs(self):
        idxs = []
        final_idx=[]
        t_rs=[]
        for i in range(self.args.num_edges):
            sample_size = min(self.args.channel_num, len(self.mob.zone_client_id[i]))
            idxs = idxs + list(np.random.choice(self.mob.zone_client_id[i], sample_size, replace=False))
        for number in idxs:
            t_left = max(self.args.upload_size / (self.args.bandwidth * np.log2(
                self.args.power * self.mob.channel_gain[number] / self.args.noise + 1)),
                         self.tmax[number] - self.args.num_local_epoch * self.args.gamma * self.datasize[
                             number] / self.args.f_min, 0)
            t_right = self.tmax[number] - self.args.num_local_epoch * self.args.gamma * self.datasize[
                number] / self.args.f_max
            if t_left<t_right:
                final_idx.append(number)
                t=np.random.random()*(t_right-t_left)+t_left
                t_rs.append(t)
        energy = [(self.rt(t, idx) + self.Qn[idx]) / self.args.lyp_v for idx,t in zip(final_idx,t_rs)]
        return final_idx, energy


    def max_power_and_frequency(self):
        # 随机选择，最大功率和频率
        idxs = []
        energy = []
        for i in range(self.args.num_edges):
            idxs = idxs + list(np.random.choice(self.mob.zone_client_id[i], self.args.channel_num, replace=False))
        for idx in idxs:
            e = self.args.alpha * self.args.num_local_epoch * self.args.gamma * self.datasize[
                idx] * self.args.f_max ** 2
            e = e + self.args.power * self.args.upload_size / (self.args.bandwidth * np.log2(
                1 + self.args.power * self.mob.channel_gain[idx] / self.args.noise))
            energy.append(e)
        idxs = [idx for idx, e in zip(idxs, energy) if not np.isinf(e)]
        energy = [e for e in energy if not np.isinf(e)]
        return idxs, energy

    def policy(self):
        min_rt = []
        idxs = []
        for i in range(self.args.num_clients):
            min_rt.append(self.cal_min_rt(i))
        idx_rt = list(enumerate(min_rt))

        for i in range(self.args.num_edges):
            sorted_list = sorted([idx_rt[index] for index in self.mob.zone_client_id[i]], key=lambda x: x[1])
            min_indices = [idx for idx, _ in sorted_list[:self.args.channel_num]]
            idxs = idxs + min_indices
        energy = [(min_rt[idx] + self.Qn[idx]) / self.args.lyp_v for idx in idxs]
        # Check if any energy value is inf and remove corresponding idxs and energy values
        idxs = [idx for idx, e in zip(idxs, energy) if not np.isinf(e)]
        energy = [e for e in energy if not np.isinf(e)]
        return idxs, energy
    def policy_add_rho(self,rho):
        self.Qn=[self.Qn[i]+rho[i] for i in range(self.args.num_clients)]
        idxs, energy = self.policy()
        return idxs, energy
    def constant_participation(self):
        return self.policy()



    def cal_min_rt(self, number):
        t_left = max(self.args.upload_size / (self.args.bandwidth * np.log2(
            self.args.power * self.mob.channel_gain[number] / self.args.noise + 1)),
                     self.tmax[number] - self.args.num_local_epoch * self.args.gamma * self.datasize[
                         number] / self.args.f_min, 0)
        t_right = self.tmax[number] - self.args.num_local_epoch * self.args.gamma * self.datasize[
            number] / self.args.f_max
        if t_right < t_left:
            print('false,right<left', t_left, t_right, self.tmax[number])
            return float('inf')
        temp = self.grad_rt(t_left, number) * self.grad_rt(t_right, number)
        if temp > 0:
            print('side point')
            return min(self.rt(t_left, number), self.rt(t_right, number))
        else:
            t = self.binary_search(t_left, t_right, number)
            print('binary search')
            return self.rt(t, number)

    def energy_driven(self):
        print('energy driven')
        idxs = []
        energy = []
        channel_gain = copy.deepcopy(np.array(self.mob.channel_gain))
        for i in range(self.args.num_edges):
            sorted_list = sorted([idx for idx in range(self.args.num_clients) if idx in self.mob.zone_client_id[i]],
                                 key=lambda x: channel_gain[x],reverse=True)
            min_indices = [idx for idx in sorted_list[:self.args.channel_num]]
            e = [(self.cal_min_rt(idx) + self.Qn[idx]) / self.args.lyp_v for idx in min_indices]
            idxs = idxs + min_indices
            energy = energy + e
        idxs = [idx for idx, e in zip(idxs, energy) if not np.isinf(e)]
        energy = [e for e in energy if not np.isinf(e)]
        return idxs, energy

    def loss_driven(self, loss):
        idxs = []
        energy = []
        for i in range(self.args.num_edges):
            sorted_list = sorted([idx for idx in range(self.args.num_clients) if idx in self.mob.zone_client_id[i]],
                                    key=lambda x: loss[x],reverse=True)
            min_indices = [idx for idx in sorted_list[:self.args.channel_num]]
            e = [(self.cal_min_rt(idx) + self.Qn[idx]) / self.args.lyp_v for idx in min_indices]
            idxs = idxs + min_indices
            energy = energy + e
        idxs = [idx for idx, e in zip(idxs, energy) if not np.isinf(e)]
        energy = [e for e in energy if not np.isinf(e)]
        return idxs, energy
    def binary_search(self, t_left, t_right, number, tol=1e-6, max_iter=1000):
        iter_count = 0
        while abs(t_right - t_left) > tol and iter_count < max_iter:
            t_mid = (t_left + t_right) / 2
            grad_mid = self.grad_rt(t_mid, number)
            if grad_mid == 0:
                return t_mid
            elif grad_mid > 0:
                t_right = t_mid
            else:
                t_left = t_mid
            iter_count += 1

        # 如果达到最大迭代次数或者收敛到所需精度，则返回当前的 t 值
        return (t_left + t_right) / 2

    '''
    def rt(self, t, number):
        return (self.args.alpha * self.args.num_local_epoch ** 3 * self.args.gamma ** 3 * self.datasize[number] ** 3 / (
                    t - self.tmax[number]) ** 2 + self.args.noise * t * (
                            2 ** (self.args.upload_size / (self.args.bandwidth * t)) - 1) / (
                self.mob.channel_gain[number]) - self.Qn[number])

    import torch
    '''

    def rt(self, t, number):
        # 提取参数
        alpha = self.args.alpha
        E = self.args.num_local_epoch
        gamma = self.args.gamma
        D = self.datasize[number]
        tmax = self.tmax[number]
        noise = self.args.noise
        U = self.args.upload_size
        B = self.args.bandwidth
        G = self.mob.channel_gain[number]
        Qn = self.Qn[number]
        V = self.args.lyp_v
        # 第一项
        first_term = alpha * E ** 3 * gamma ** 3 * D ** 3 / (t - tmax) ** 2

        # 第二项
        exp_term = 2 ** (U / (B * t))
        second_term = noise * t * (exp_term - 1) / G

        # 总体
        result = V * first_term + V * second_term - Qn
        print('first_term:', first_term)
        print('second_term', second_term)

        return result

    def grad_rt(self, t, number):
        E = self.args.num_local_epoch
        D = self.datasize[number]
        alpha = self.args.alpha
        gamma = self.args.gamma
        tmax = self.tmax[number]
        noise = self.args.noise
        U = self.args.upload_size
        B = self.args.bandwidth
        G = self.mob.channel_gain[number]
        Qn = self.Qn[number]

        # 第一项对 t 的导数
        first_term_grad = -2 * alpha * E ** 3 * gamma ** 3 * D ** 3 / (t - tmax) ** 3

        # 第二项对 t 的导数

        exp_u = 2 ** (U / (B * t))
        second_term_grad = noise * (exp_u - 1) / G - np.log(2) * noise * U * exp_u / (B * G * t)

        # 最终导数
        grad = first_term_grad + second_term_grad

        return grad
