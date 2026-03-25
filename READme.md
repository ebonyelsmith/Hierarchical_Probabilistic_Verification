# From Global to Local: Hierarchical Probabilistic Verification for Reachability Learning
### [Paper](https://arxiv.org/)<br>

Ebonye Smith, Sampada Deglurkar, Jingqi Li, Gechen Qu, Claire Tomlin<br>
University of California, Berkeley<br> 
University of Texas at Austin

This repository contains the implementation of a **Hierarchical Safe, Target-Reaching Framework** that combines global probabilistic verification with online local refinement for a drone racing case study. We leverage scenario optimization theory to provide probabilistic safety guarantees for globally and locally verified sets.

## Setup

In a conda environment, please follow the instructions to install the [Lipschitz Continuous Reachability Learning](https://github.com/jamesjingqili/Lipschitz_Continuous_Reachability_Learning) repo. This repos provides the necessary gym environment for drone racing case study and the learned reachability value function with associated learned policy.

Also, refer to environment.yml for additional installation of dependencies in your conda environment.

## Run experiments
To run a Monte Carlo experiment that uses our method compared with baselines, use the command
```
python run_all_controllers.py     --value-path experiment_script/pretrained_neural_networks/ra_droneracing_Game-v6/ddpg_reach_avoid_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_1_tau_0.005_training_num_8_buffer_size_40000_c_net_512_4_a1_512_4_a2_512_4_gamma_0.95/noise_0.1_actor_lr_0.0001_critic_lr_0.001_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/epoch_id_100/policy.pth  --save-figure experiment_script/data/pipeline.png     --save-gif experiment_script/data/pipeline.gif
```

To simply run an experiment with our framework alone, use command
```
python controllers/local_verif_switch_updated_scen.py     --value-path experiment_script/pretrained_neural_networks/ra_droneracing_Game-v6/ddpg_reach_avoid_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_1_tau_0.005_training_num_8_buffer_size_40000_c_net_512_4_a1_512_4_a2_512_4_gamma_0.95/noise_0.1_actor_lr_0.0001_critic_lr_0.001_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/epoch_id_100/policy.pth  --save-figure experiment_script/data/pipeline.png     --save-gif experiment_script/data/pipeline.gif
```

## Evaluation
We provide a dedicated Jupyter Notebook to evaluate Monte Carlo results and produce figures and gifs.
```
jupyter notebook experiment_script/evaluate_monte_carlo_updated.ipynb
```

## Citation

<!--
## Citation
If you find our work useful in your research, please cite:
```
@inproceedings{bansal2020deepreach,
    author = {Bansal, Somil
              and Tomlin, Claire},
    title = {{DeepReach}: A Deep Learning Approach to High-Dimensional Reachability},
    booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
    year={2021}
}
```
--->

<!-- 
## Contact
If you have any questions, please feel free to email the authors.
-->