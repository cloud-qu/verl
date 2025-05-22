
# from .risklearner import RiskLearner
# from .new_trainer_risklearner import RiskLearnerTrainer
# # from wordllama import WordLlama
# import torch
# from .sampler import MP_BatchSampler
# # from sentence_transformers import SentenceTransformer


# class TS4LLM():
#     def __init__(self, args, tokenizer, device='cuda'):
#         self.args = args#device, mpts_proxy_lr, mpts_kl_weight
#         self.framework = self.args.tasksampler.framework #0: uniform; 1: gdro; 2: cvar; 3: mpts
#         self.ts_ratio = self.args.tasksampler.ts_ratio
#         self.real_batch_size = self.args.data.train_batch_size
#         self.gamma_0 = self.args.tasksampler.gamma_0
#         self.gamma_1 = self.args.tasksampler.gamma_1
#         self.gamma_2 = self.args.tasksampler.gamma_2
#         self.tokenizer = tokenizer
#         self.embed_dim = 256#384
#         self.device = device
#         from wordllama import WordLlama
#         self.wl = WordLlama.load(cache_dir='~/deepscaler/hfmodels/wordllama', dim=self.embed_dim,disable_download=True)
#         # self.embed_model = SentenceTransformer("/home/quyun/deepscaler/hfmodels/sentence-transformers/all-MiniLM-L6-v2")

#         if self.args.tasksampler.framework == 3:
#             risklearner = RiskLearner(self.embed_dim, 1, 64, 64, 512).to(device)
#             risklearner_optimizer = torch.optim.Adam(risklearner.parameters(), lr=self.args.tasksampler.mpts_proxy_lr)
#             self.risklearner_trainer = RiskLearnerTrainer(device, risklearner, risklearner_optimizer, kl_weight=self.args.tasksampler.mpts_kl_weight)
#             self.sampler = MP_BatchSampler(self.args.tasksampler, self.risklearner_trainer, self.gamma_0, self.gamma_1, rejection_sampling_ratio=self.args.tasksampler.rejection_sampling_ratio)
#     def inputids_to_embeddings(self, input_ids):
#         real_sentences = []
#         for input_id in input_ids:
#             sentence = self.tokenizer.decode(input_id)
#             assert '<｜User｜>' in sentence
#             # real_sentences.append(sentence.split('<｜User｜>')[1].split('<｜Assistant｜>')[0])
#             real_sentences.append(sentence.split('<｜User｜>')[1].split("Let's think step by step and output the final answer")[0])
#         prompt_embeddings = self.wl.embed(real_sentences)
#         # prompt_embeddings = self.embed_model.encode(real_sentences)
#         return prompt_embeddings
    
#     def sample_batch(self, batch_candidates_dict):
#         if self.framework == 0:
#             return self.sample_batch_uniform(batch_candidates_dict), None
#         elif self.framework == 3:
#             sampled_index, sampled_acquisition_score = self.sampler.sample_tasks(self.inputids_to_embeddings(batch_candidates_dict['input_ids']), self.real_batch_size)
#             batch_candidates_dict = {k: v[sampled_index] for k, v in batch_candidates_dict.items()}
#             return batch_candidates_dict, sampled_acquisition_score
#         else:
#             raise NotImplementedError

#     def sample_batch_uniform(self, batch_candidates_dict):
#         uniform_index = torch.randperm(len(batch_candidates_dict['input_ids']))[:self.real_batch_size]
#         batch_candidates_dict = {k: v[uniform_index] for k, v in batch_candidates_dict.items()}
#         return batch_candidates_dict
    
#     def train(self, batch_candidates_dict, y):
#         if self.framework == 3:
#             loss, recon_loss, kl_loss = self.sampler.train(self.inputids_to_embeddings(batch_candidates_dict['input_ids']), -y)
#             return loss, recon_loss, kl_loss
#         else:
#             return None, None, None


import numpy as np

class PosteriorSampler:
    def __init__(self, args, total_num_samples=40315, prior_alpha=1.0, prior_beta=1.0, 
                 target_mean=0.5, target_std=0.1, sample_std=0.05):
        """
        :param num_samples: 总样本数 N
        :param prior_alpha, prior_beta: Beta先验参数
        :param target_mean, target_std: 目标成功率分布的参数
        :param sample_std: 控制 softmax 采样 sharpness
        """
        self.args = args
        self.real_batch_size = self.args.data.train_batch_size
        self.num_samples = total_num_samples
        self.alpha = np.ones(total_num_samples) * prior_alpha
        self.beta = np.ones(total_num_samples) * prior_beta
        self.init_success_rate = np.zeros(total_num_samples)
        self.target_mean = target_mean
        self.target_std = target_std
        self.sample_std = sample_std
        self.initialize_from_json()

    def sample_batch(self, batch_candidates_dict):
        """
        :param candidate_indices: 候选样本索引数组，大小为 m
        :param batch_size: 最终选出的样本数量
        :return: 被选中的样本索引数组，大小为 batch_size
        """
        candidate_indices = batch_candidates_dict['index'].astype('int')
        m = len(candidate_indices)
        assert self.real_batch_size <= m, "batch_size must be <= number of candidates"

        # Step 1: 从目标成功率分布中采样 mu_t
        target_mu = np.random.normal(loc=self.target_mean, scale=self.target_std)

        # Step 2: 从候选的 posterior 中采样 r_i
        sampled_r = np.random.beta(self.alpha[candidate_indices], self.beta[candidate_indices])

        # Step 3: 计算 softmax 权重（越接近 mu_t 权重越大）
        distances = (sampled_r - target_mu) ** 2
        weights = np.exp(-distances / (2 * self.sample_std ** 2))
        probs = weights / weights.sum()

        # Step 4: 根据概率从候选中采样 batch_size 个
        # sampled_index = np.random.choice(m, size=self.real_batch_size, p=probs, replace=False)
        sampled_index = np.argsort(distances)[:self.real_batch_size]
        batch_candidates_dict = {k: v[sampled_index] for k, v in batch_candidates_dict.items()}
        return batch_candidates_dict, torch.tensor(sampled_r[sampled_index]).to('cuda'), sampled_index

    def train(self, batch_candidates_dict, y):
        """
        :param indices: 被训练过的样本索引数组
        :param successes: 成功率数组（float ∈ [0,1]）
        """
        indices = batch_candidates_dict['index'].astype('int')
        for idx, s in zip(indices, y):
            self.alpha[idx] += s * self.args.actor_rollout_ref.rollout.n
            self.beta[idx] += (1 - s)  * self.args.actor_rollout_ref.rollout.n
        return None, None, None
    
    def initialize_from_json(self, json_path = None):
        """
        从json文件中加载先验观测，更新alpha和beta
        """
        import json
        import os
        if json_path is None:
            json_path = f"{os.environ['HOME']}/deepscaler/outputs/init_eval_train_1/index_score.json"
    
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for key, results in data.items():
            idx = int(key)
            successes = sum(results)
            failures = len(results) - successes
            self.alpha[idx] += successes * 3
            self.beta[idx] += failures * 3
            self.init_success_rate[idx] = successes / (successes + failures)



if __name__ == "__main__":
    import json
    import numpy as np
    from tqdm import tqdm
    import torch

    from types import SimpleNamespace

    # 创建嵌套参数结构
    class NestedNamespace(SimpleNamespace):
        def __init__(self, dictionary=None, **kwargs):
            if dictionary is None:
                dictionary = {}
            for key, value in {**dictionary, **kwargs}.items():
                if isinstance(value, dict):
                    setattr(self, key, NestedNamespace(value))
                else:
                    setattr(self, key, value)

    # 初始化参数
    args = NestedNamespace({
        'data': {
            'train_batch_size': 40315
        },
        'actor_rollout_ref': {
            'rollout': {
                'n': 8
            }
        }
    })
    total_num_samples = 40315  # Example total number of samples
    sampler = PosteriorSampler(args, total_num_samples)

    # Example batch_candidates_dict
    total_data = {'index': np.arange(total_num_samples)}

    # Sample a batch
    sampled_batch, sampled_acquisition_score, sampled_index= sampler.sample_batch(total_data)
    
    corr = np.corrcoef(sampled_acquisition_score.squeeze().cpu().detach().numpy(), (sampler.init_success_rate[sampled_index]))[0,1]
    print(f"Correlation coefficient: {corr}")


