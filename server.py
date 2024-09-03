
import copy
import torch

class Server():

    def __init__(self,init_model):
        self.receiver_buffer = {}
        self.shared_state_dict = copy.deepcopy(init_model.state_dict())
        self.id_registration = []
        self.sample_registration = {}


    def refresh_cloudserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None



    def receive_from_edge(self, edge_id, eshared_state_dict,total_sample):
        self.receiver_buffer[edge_id] = eshared_state_dict
        self.sample_registration[edge_id]=total_sample
        return None

    def aggregate(self):
        received_dict = [value for key,value in self.receiver_buffer.items() if self.sample_registration[key]>0]
        sample_num = [snum for snum in self.sample_registration.values() if snum>0]

        #received_dict = [dict for dict in self.receiver_buffer.values()]
        #sample_num = [snum for snum in self.sample_registration.values()]
        self.shared_state_dict = average_weights(w=received_dict,
                                                 s_num=sample_num)
        return None

    def send_to_edge(self, edge):
        edge.receive_from_cloudserver(copy.deepcopy(self.shared_state_dict))
        return None

def average_weights(w, s_num):
    
    total_sample_num = sum(s_num)
    temp_sample_num = s_num[0]
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():  #the nn layer loop
        for i in range(1, len(w)):   #the client loop
            w_avg[k] += torch.mul(w[i][k], s_num[i]/temp_sample_num)
        w_avg[k] = torch.mul(w_avg[k], temp_sample_num/total_sample_num)
    return w_avg