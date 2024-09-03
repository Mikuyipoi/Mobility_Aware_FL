
import copy
import torch

class Edge():

    def __init__(self, id):
        """
        id: edge id
        cids: ids of the clients under this edge
        receiver_buffer: buffer for the received updates from selected clients
        shared_state_dict: state dict for shared network
        id_registration: participated clients in this round of traning
        sample_registration: number of samples of the participated clients in this round of training
        all_trainsample_num: the training samples for all the clients under this edge
        shared_state_dict: the dictionary of the shared state dict
        clock: record the time after each aggregation
        :param id: Index of the edge
        :param cids: Indexes of all the clients under this edge
        :param shared_layers: Structure of the shared layers
        :return:
        """
        self.id = id
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        #self.id_registration = []
        self.sample_registration = {}
        self.moved_id=[]
        self.total_sample = 0


    def refresh_edgeserver(self):
        self.receiver_buffer.clear()
        #del self.id_registration[:]
        self.sample_registration.clear()
        self.total_sample=0




    def receive_from_client(self, client_id, cshared_state_dict,num_sample):
        self.receiver_buffer[client_id] = cshared_state_dict
        self.sample_registration[client_id]=num_sample
        self.total_sample+=num_sample

    def aggregate(self):
        """
        Using the old aggregation funciton
        :param args:
        :return:
        """
        if self.total_sample == 0:
            return None
        received_dict = [dict for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        self.shared_state_dict = average_weights(w = received_dict,
                                                 s_num= sample_num)

    def send_to_client(self, client):
        client.receive_from_edgeserver(copy.deepcopy(self.shared_state_dict))
        return None

    def send_to_cloudserver(self, cloud):
        cloud.receive_from_edge(edge_id=self.id,
                                eshared_state_dict= copy.deepcopy(self.shared_state_dict),
                                total_sample=copy.deepcopy(self.total_sample)
                                    )
        return None

    def receive_from_cloudserver(self, shared_state_dict):
        self.shared_state_dict = shared_state_dict
        return None

def average_weights(w, s_num):
    #copy the first client's weights
    total_sample_num = sum(s_num)
    temp_sample_num = s_num[0]
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():  #the nn layer loop
        for i in range(1, len(w)):   #the client loop
            w_avg[k] += torch.mul(w[i][k], s_num[i]/temp_sample_num)
        w_avg[k] = torch.mul(w_avg[k], temp_sample_num/total_sample_num)
    return w_avg