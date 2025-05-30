
# from wordllama import WordLlama
import torch

from .new_trainer_risklearner import RiskLearnerTrainer
from .risklearner import RiskLearner
from .sampler import MP_BatchSampler

# from sentence_transformers import SentenceTransformer


class TS4LLM():
    def __init__(self, args, tokenizer, device='cuda'):
        self.args = args#device, mpts_proxy_lr, mpts_kl_weight
        self.framework = self.args.tasksampler.framework #0: uniform; 1: gdro; 2: cvar; 3: mpts
        self.ts_ratio = self.args.tasksampler.ts_ratio
        self.real_batch_size = self.args.data.train_batch_size
        self.gamma_0 = self.args.tasksampler.gamma_0
        self.gamma_1 = self.args.tasksampler.gamma_1
        self.gamma_2 = self.args.tasksampler.gamma_2
        self.tokenizer = tokenizer
        self.embed_dim = 256#384
        self.device = device
        from wordllama import WordLlama
        self.wl = WordLlama.load(cache_dir='~/deepscaler/hfmodels/wordllama', dim=self.embed_dim,disable_download=True)
        # self.embed_model = SentenceTransformer("/home/quyun/deepscaler/hfmodels/sentence-transformers/all-MiniLM-L6-v2")

        if self.args.tasksampler.framework == 3:
            risklearner = RiskLearner(self.embed_dim, 1, 64, 64, 512).to(device)
            risklearner_optimizer = torch.optim.Adam(risklearner.parameters(), lr=self.args.tasksampler.mpts_proxy_lr)
            self.risklearner_trainer = RiskLearnerTrainer(device, risklearner, risklearner_optimizer, kl_weight=self.args.tasksampler.mpts_kl_weight)
            self.sampler = MP_BatchSampler(self.args.tasksampler, self.risklearner_trainer, self.gamma_0, self.gamma_1, rejection_sampling_ratio=self.args.tasksampler.rejection_sampling_ratio)
    def inputids_to_embeddings(self, input_ids):
        real_sentences = []
        for input_id in input_ids:
            sentence = self.tokenizer.decode(input_id)
            assert '<｜User｜>' in sentence
            # real_sentences.append(sentence.split('<｜User｜>')[1].split('<｜Assistant｜>')[0])
            real_sentences.append(sentence.split('<｜User｜>')[1].split("Let's think step by step and output the final answer")[0])
        prompt_embeddings = self.wl.embed(real_sentences)
        # prompt_embeddings = self.embed_model.encode(real_sentences)
        return prompt_embeddings
    
    def sample_batch(self, batch_candidates_dict):
        if self.framework == 0:
            return self.sample_batch_uniform(batch_candidates_dict), None
        elif self.framework == 3:
            sampled_index, sampled_acquisition_score = self.sampler.sample_tasks(self.inputids_to_embeddings(batch_candidates_dict['input_ids']), self.real_batch_size)
            batch_candidates_dict = {k: v[sampled_index] for k, v in batch_candidates_dict.items()}
            return batch_candidates_dict, sampled_acquisition_score
        else:
            raise NotImplementedError

    def sample_batch_uniform(self, batch_candidates_dict):
        uniform_index = torch.randperm(len(batch_candidates_dict['input_ids']))[:self.real_batch_size]
        batch_candidates_dict = {k: v[uniform_index] for k, v in batch_candidates_dict.items()}
        return batch_candidates_dict
    
    def train(self, batch_candidates_dict, y):
        if self.framework == 3:
            loss, recon_loss, kl_loss = self.sampler.train(self.inputids_to_embeddings(batch_candidates_dict['input_ids']), -y)
            return loss, recon_loss, kl_loss
        else:
            return None, None, None
    def save(self, save_path):
        if self.args.tasksampler.framework == 3:
            #保存risklearner模型和optimizer
            torch.save({
                'model_state_dict': self.risklearner_trainer.risklearner.state_dict(),
                'optimizer_state_dict': self.risklearner_trainer.optimizer.state_dict(),
            }, save_path + '/risklearner.pth')
    def load(self, load_path):
        try:
            if self.args.tasksampler.framework == 3:
                #加载risklearner模型和optimizer
                checkpoint = torch.load(load_path + '/risklearner.pth')
                self.risklearner_trainer.risklearner.load_state_dict(checkpoint['model_state_dict'])
                self.risklearner_trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.risklearner_trainer.risklearner.to(self.device)
        except:
            pass



import os

import numpy as np


class PosteriorSampler:
    def __init__(self, args, total_num_samples, prior_alpha=1.0, prior_beta=1.0, init=False, init_dir=f"{os.environ['HOME']}/deepscaler/outputs/init_eval_train_1/index_score.json"):
        """
        :param num_samples: 总样本数 N
        :param prior_alpha, prior_beta: Beta先验参数
        :param target_mean, target_std: 目标成功率分布的参数
        :param sample_std: 控制 softmax 采样 sharpness
        """
        self.args = args
        self.real_batch_size = self.args.data.train_batch_size
        self.num_samples = total_num_samples
        # self.alpha = np.ones(total_num_samples) * prior_alpha
        self.alpha = {}
        self.beta = {}
        # self.beta = np.ones(total_num_samples) * prior_beta
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.target_mean = args.tasksampler.bandit_target_mean
        self.target_std = args.tasksampler.bandit_target_std
        self.sample_std = args.tasksampler.bandit_sample_std
        self.lower_bound = args.tasksampler.bandit_lower_bound
        self.upper_bound = args.tasksampler.bandit_upper_bound
        self.upper_bound_decay_steps = args.tasksampler.bandit_upper_decay_steps
        self.upper_bound_decay_lower = args.tasksampler.bandit_upper_decay_lower
        self.sampling_strategy = args.tasksampler.bandit_sample_strategy
        if init:
            self.initialize_from_json(init_dir)

    def sample_batch(self, batch_candidates_dict):
        """
        :param candidate_indices: 候选样本索引数组，大小为 m
        :param batch_size: 最终选出的样本数量
        :return: 被选中的样本索引数组，大小为 batch_size
        """
        candidate_indices = batch_candidates_dict['index']
        m = len(candidate_indices)
        assert self.real_batch_size <= m, "batch_size must be <= number of candidates"

        # Step 1: 从目标成功率分布中采样 mu_t
        target_mu = np.random.normal(loc=self.target_mean, scale=self.target_std)

        # Step 2: 从候选的 posterior 中采样 r_i
        local_alpha = []
        local_beta = []
        for index in candidate_indices:
            index = str(index)
            if index not in self.alpha.keys():
                self.alpha[index] = self.prior_alpha
            local_alpha.append(self.alpha[index])
            if index not in self.beta.keys():
                self.beta[index] = self.prior_beta
            local_beta.append(self.beta[index])
        sampled_r = np.random.beta(np.array(local_alpha), np.array(local_beta))

        # Step 3: 计算 softmax 权重（越接近 mu_t 权重越大）
        if self.sampling_strategy == 'uniform':
            sampled_index = np.random.choice(m, size=self.real_batch_size, replace=False)
        elif self.sampling_strategy == 'softmax':
            # 计算 softmax 权重
            distances = (sampled_r - target_mu) ** 2
            weights = np.exp(-distances / (2 * self.sample_std ** 2))
            probs = weights / weights.sum()
            sampled_index = np.random.choice(m, size=self.real_batch_size, p=probs, replace=False)
        elif self.sampling_strategy == 'topk':
            distances = (sampled_r - target_mu) ** 2
            sampled_index = np.argsort(distances)[:self.real_batch_size]
        elif self.sampling_strategy == 'threshold':
            in_range_mask = (sampled_r >= self.lower_bound) & (sampled_r <= self.upper_bound)
            in_range_indices = np.where(in_range_mask)[0]
            if len(in_range_indices) >= self.real_batch_size:
                # 范围内样本足够，随机选择
                np.random.shuffle(in_range_indices)
                sampled_index = in_range_indices[:self.real_batch_size]
            else:
                # 范围内样本不足，计算到范围边界的距离
                scores = np.zeros_like(sampled_r)
                too_low = sampled_r < self.lower_bound
                too_high = sampled_r > self.upper_bound
                scores[too_low] = self.lower_bound - sampled_r[too_low]  # 距离下界的距离
                scores[too_high] = sampled_r[too_high] - self.upper_bound  # 距离上界的距离
                
                # 选择评分最低的样本（在范围内或最接近范围的）
                sampled_index = np.argsort(scores)[:self.real_batch_size]
        elif self.sampling_strategy == 'median':
            # 计算所有候选样本的后验中位数
            median_r = np.median(sampled_r)
            distances = np.abs(sampled_r - median_r)
            sampled_index = np.argsort(distances)[:self.real_batch_size]
    
        batch_candidates_dict = {k: v[sampled_index] for k, v in batch_candidates_dict.items()}
        # Step 4: 更新上界
        if self.upper_bound_decay_steps > 0:
            decay_factor = (self.upper_bound - self.upper_bound_decay_lower) / self.upper_bound_decay_steps
            self.upper_bound = max(self.upper_bound - decay_factor, self.upper_bound_decay_lower)
        return batch_candidates_dict, torch.tensor(sampled_r[sampled_index])#.to('cuda')

    def train(self, batch_candidates_dict, y):
        """
        :param indices: 被训练过的样本索引数组
        :param successes: 成功率数组（float ∈ [0,1]）
        """
        indices = batch_candidates_dict['index']
        for idx, s in zip(indices, y):
            idx = str(idx)
            self.alpha[idx] += s * self.args.actor_rollout_ref.rollout.n
            self.beta[idx] += (1 - s)  * self.args.actor_rollout_ref.rollout.n
        return None, None, None
    
    
    def initialize_from_json(self, json_path = None):
        """
        从json文件中加载先验观测，更新alpha和beta
        """
        import json
        if json_path is None:
            json_path = f"{os.environ['HOME']}/deepscaler/outputs/init_eval_train_1/index_score.json"
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for key, results in data.items():
            idx = key
            successes = sum(results)
            failures = len(results) - successes
            self.alpha[idx] = successes * 3 + self.prior_alpha
            self.beta[idx] = failures * 3 + self.prior_beta

    def save(self, save_path):
        """
        保存alpha和beta到json文件
        """
        import json
        import os
        data = {}
        for index in self.alpha.keys():
            data[index] = [int(self.alpha[index]), int(self.beta[index])]
        with open(os.path.join(save_path, 'index_score.json'), 'w') as f:
            json.dump(data, f)

    def load(self, load_path):
        """
        从json文件中加载alpha和beta
        """
        try:
            import json
            import os
            with open(os.path.join(load_path, 'index_score.json'), 'r') as f:
                data = json.load(f)
            
            for key, results in data.items():
                # idx = int(key)
                idx = key
                self.alpha[idx] = results[0]
                self.beta[idx] = results[1]
        except:
            pass




class HistorySampler:
    def __init__(self, total_num_samples):
        """
        :param num_samples: 总样本数 N
        :param prior_alpha, prior_beta: Beta先验参数
        :param target_mean, target_std: 目标成功率分布的参数
        :param sample_std: 控制 softmax 采样 sharpness
        """
        self.num_samples = total_num_samples
        # self.scores = np.zeros(total_num_samples)
        self.scores = {}

    def sample_batch(self, batch_candidates_dict):
        return batch_candidates_dict, None

    def train(self, batch_candidates_dict, y):
        """
        :param indices: 被训练过的样本索引数组
        :param successes: 成功率数组（float ∈ [0,1]）
        """
        indices = batch_candidates_dict['index']
        for idx, s in zip(indices, y):
            self.scores[idx] = s
        return None, None, None
    
    def get_informative_indices(self):
        """
        返回所有 score < 1 的样本原始索引列表
        """
        # np.where 返回的是元组，第一个元素即是满足条件的索引数组
        low_idxs = []
        for key in self.scores.keys():
            if self.scores[key] < 1:
                low_idxs.append(key)
        return low_idxs
        # low_idxs = np.where(self.scores < 1)[0]
        # return low_idxs.tolist()

    def save(self, save_path):
        """
        保存 scores 到 json 文件
        """
        import json
        import os
        data = {}
        for index in self.scores.keys():
            data[index] = float(self.scores[index])
        with open(os.path.join(save_path, 'index_score.json'), 'w') as f:
            json.dump(data, f)
    def load(self, load_path):
        """
        从 json 文件加载 scores
        """
        try:
            import json
            import os
            with open(os.path.join(load_path, 'index_score.json'), 'r') as f:
                data = json.load(f)
            
            for key, score in data.items():
                idx = key
                self.scores[idx] = score
        except:
            pass
    