import argparse
import os

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Independent, Normal




from LCRL.env import DummyVectorEnv
# from LCRL.exploration import GaussianNoise
# from LCRL.utils.net.common import Net
# from LCRL.utils.net.continuous import Actor, Critic
# import LCRL.reach_rl_gym_envs as reach_rl_gym_envs

# from MARL.data import Collector, VectorReplayBuffer
# from MARL.env import DummyVectorEnv
# from MARL.policy.MARL_base import MARL_BasePolicy
# from MARL.trainer import OnpolicyTrainer
# from MARL.utils import TensorboardLogger

from MARL.policy.gym_marl_policy.ippo import IPPOPolicy
from MARL.utils.net.common import ActorCritic, Net
from MARL.utils.net.continuous import ActorProb, Critic
from MARL.data import Batch

# from gymnasium.vector import SyncVectorEnv
from copy import deepcopy
# from gymnasium.vector.utils import concatenate

def get_args_ppo() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="ra_ppo_droneracing_Game-v6")
    parser.add_argument("--reward-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=40000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.95) #9)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--total-episodes", type=int, default=30)
    parser.add_argument("--step-per-epoch", type=int, default=40000)
    parser.add_argument("--episode-per-collect", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--critic-net", type=int, nargs="*", default=[512]*3)
    parser.add_argument("--actor-net", type=int, nargs="*", default=[512]*3)
    parser.add_argument("--training-num", type=int, default=64)
    parser.add_argument("--test-num", type=int, default=100)
    # parser.add_argument("--logdir", type=str, default="icra/log_MARL_fix_bug")
    parser.add_argument("--logdir", type=str, default="log_final1")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument('--continue-training-logdir', type=str, default=None)
    parser.add_argument('--continue-training-epoch', type=int, default=None)
    parser.add_argument('--behavior-loss-weight', type=float, default=0.1)
    parser.add_argument('--behavior-loss-weight-decay', type=float, default=1)
    parser.add_argument('--regularization', type=bool, default=False) # if true, then expert_policy = 0
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    # ppo special
    parser.add_argument('--repeat-per-collect', type=int, default=10)
    parser.add_argument('--vf-coef', type=float, default=0.25)
    parser.add_argument('--ent-coef', type=float, default=0.004)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=1)
    parser.add_argument('--norm-adv', type=int, default=1)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument("--save-interval", type=int, default=4)
    parser.add_argument('--kwargs', type=str, default='{}')
    return parser.parse_known_args()[0]

def get_env_and_policy_ppo(args, epoch_id):
    env = gym.make(args.task)
    print(f"Task: {args.task}")
    args.max_action = env.action_space.high[0]
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    train_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.training_num)])

    test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    args.action_shape_per_player = env.nu

    actor_net = Net(args.state_shape, hidden_sizes=args.actor_net, device=args.device)
    actor_list = [
        ActorProb(
            actor_net, args.action_shape_per_player, max_action=args.max_action, device=args.device
        ).to(args.device) for i in range(env.num_players)
    ]
    critic_net = Net(
        args.state_shape,
        hidden_sizes=args.critic_net,
        device=args.device
    )
    critic_list = [
        Critic(critic_net, device=args.device).to(args.device) for i in range(env.num_players)
    ]
    actor_critic_list = [ActorCritic(actor_list[i], critic_list[i]) for i in range(env.num_players)]
    # orthogonal initialization
    for i in range(env.num_players):
        for m in actor_critic_list[i].modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)
    optim_list = [
        torch.optim.Adam(actor_critic_list[i].parameters(), lr=args.lr) for i in range(env.num_players)
    ]
    # replace DiagGuassian with Independent(Normal) which is equivalent
    # pass *logits to be consistent with policy.forward
    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = IPPOPolicy(
        actor_list,
        critic_list,
        optim_list,
        env.num_players,
        env.nu,
        dist,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        gae_lambda=args.gae_lambda,
        action_space=env.action_space,
        device=args.device,
        deterministic_eval=True,
        pure_policy_regulation = args.regularization,
        env = env,
        batch_size = args.batch_size,
    )
    # # collector
    # train_collector = Collector(
    #         policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs))
    #     )
    # test_collector = Collector(policy, test_envs)

    # log_path = os.path.join(args.logdir, args.task, 
    #                         'ippo_training_num_{}_buffer_size_{}_c_{}_{}_a_{}_{}_gamma_{}_behavior_loss_{}_{}_L2_reg_{}'.format(
    #     args.training_num, 
    #     args.buffer_size,
    #     args.critic_net[0],
    #     len(args.critic_net),
    #     args.actor_net[0],
    #     len(args.actor_net),
    #     args.gamma,
    #     args.behavior_loss_weight,
    #     args.behavior_loss_weight_decay,
    #     args.regularization
    #     )
    # )
    # log_path = log_path+'/lr_{}_batch_{}_step_per_epoch_{}_kwargs_{}_seed_{}'.format(
    #     args.lr, 
    #     args.batch_size,
    #     args.step_per_epoch,
    #     args.kwargs,
    #     args.seed
    # )
    # print(f"Loading policy from: {log_path}")
    log_path = "pretrained_neural_networks/ppo_droneracing"
    if os.path.exists(log_path):
        file_name = "/epoch_id_{}/policy.pth".format(epoch_id)
        policy.eval()
        policy.load_state_dict(torch.load(log_path+file_name))
        print(f"Policy loaded!")

    else:
        print(f"Policy path does not exist!") 

    return env, policy


def find_a_ppo(state, policy):
    tmp_obs = np.array(state).reshape(1,-1)
    tmp_batch = Batch(obs = tmp_obs, info = Batch())
    tmp = policy(tmp_batch, model = "actor_old").act
    act = policy.map_action(tmp).cpu().detach().numpy().flatten()
    return act