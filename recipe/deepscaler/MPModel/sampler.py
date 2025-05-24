

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import copy
class MP_BatchSampler(object):
    def __init__(self, args,risk_learner_trainer, gamma_0, gamma_1, rejection_sampling_ratio=0.0):
        self.risklearner_trainer = risk_learner_trainer
        self.args = args
        self.gamma_0 = gamma_0
        self.gamma_1 = gamma_1
        # self.warmup = args.warmup
        self.current_epoch = 0
        self.rejection_sampling_ratio = rejection_sampling_ratio
        

    def identifier_preprocess(self, tasks):
        if isinstance(tasks, dict):
            identifier_list = []
            for key in tasks.keys():
                identifier_list.append(torch.tensor(tasks[key]).float().to(device))
            candidate_identifier = torch.cat(identifier_list, dim=-1)
        else:
            candidate_identifier = torch.tensor(tasks).float().to(device)
        return candidate_identifier
    
    def get_acquisition_score(self, tasks, real_batch_size=None):
        tasks = self.identifier_preprocess(tasks)
        acquisition_score, acquisition_mean, acquisition_std = self.risklearner_trainer.acquisition_function(tasks, self.gamma_0, self.gamma_1)
        return acquisition_score, acquisition_mean, acquisition_std

    def sample_tasks(self, tasks, real_batch_size):
        if isinstance(tasks, dict):
            for key in tasks.keys():
                candidate_num = tasks[key].shape[0]
                break
        else:
            candidate_num = tasks.shape[0]
        acquisition_score, acquisition_mean, acquisition_std = self.get_acquisition_score(tasks) # candidate tasks 15 * loss 1
        acquisition_score = acquisition_score.squeeze(1) # candidate tasks 15
        real_acquisition_score = acquisition_score.clone()
        sorted_acquisition_score, sorted_index = torch.sort(acquisition_score, descending=True)
        acquisition_score[sorted_index[:int(((len(acquisition_score)-real_batch_size)*self.rejection_sampling_ratio+0.5)//1)]] = -float('inf')
        if not self.args.no_add_random:
            selected_values, selected_index = torch.topk(acquisition_score, k=real_batch_size//2)
        else:
            selected_values, selected_index = torch.topk(acquisition_score, k=real_batch_size)
        mask = ~torch.isin(torch.arange(0, candidate_num), selected_index.cpu())
        unselected_index = torch.arange(0, candidate_num)[mask]
        index=torch.cat((selected_index.cpu(),unselected_index),dim=0)[:real_batch_size][torch.randperm(real_batch_size)] # num_tasks 10
        index = index.cpu()
        return index, real_acquisition_score[index]
    
    def train(self, tasks, y):
        tasks = self.identifier_preprocess(tasks)
        y = y.to(device)
        loss, recon_loss, kl_loss = self.risklearner_trainer.train(tasks, y)
        return loss, recon_loss, kl_loss
    
class Diverse_MP_BatchSampler(MP_BatchSampler):
    def __init__(self, args,risk_learner_trainer, gamma_0, gamma_1, gamma_2):
        self.gamma_2 = gamma_2
        super(Diverse_MP_BatchSampler, self).__init__(args,risk_learner_trainer, gamma_0, gamma_1)

    def get_acquisition_score(self, tasks, real_batch_size=None, diversified=False):
        tasks = self.identifier_preprocess(tasks)
        if real_batch_size is None:
            if isinstance(tasks, dict):
                for key in tasks.keys():
                    candidate_num = tasks[key].shape[0]
                    break
            else:
                candidate_num = tasks.shape[0]
            real_batch_size = candidate_num
        if diversified:
            best_batch_id, diversified_score, combine_local_diverse_score, combine_local_acquisition_score, acquisition_score = self.risklearner_trainer.acquisition_function(tasks,  self.gamma_0, self.gamma_1, self.gamma_2, real_batch_size=real_batch_size)
            return best_batch_id, diversified_score, combine_local_diverse_score, combine_local_acquisition_score, acquisition_score
        else:
            acquisition_score, acquisition_mean, acquisition_std = self.risklearner_trainer.acquisition_function(tasks, self.gamma_0, self.gamma_1, self.gamma_2, pure_acquisition=True, real_batch_size=real_batch_size)
            return acquisition_score, acquisition_mean, acquisition_std


    def sample_tasks(self, tasks, real_batch_size):
        best_batch_id, diversified_score, combine_local_diverse_score, combine_local_acquisition_score, acquisition_score = self.get_acquisition_score(tasks, real_batch_size=real_batch_size, diversified=True) # candidate tasks 15 * loss 1

        return best_batch_id
