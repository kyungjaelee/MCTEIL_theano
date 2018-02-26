import argparse
import json
import h5py
import numpy as np
import yaml
import os, os.path, shutil

from policyopt import util

def load_trained_policy_and_mdp(env_name, policy_state_str):
    import gym
    import policyopt
    from policyopt import nn, rl
    from environments import rlgymenv

    # Load the saved state
    policy_file, policy_key = util.split_h5_name(policy_state_str)
    print 'Loading policy parameters from %s in %s' % (policy_key, policy_file)
    with h5py.File(policy_file, 'r') as f:
        train_args = json.loads(f.attrs['args'])

    # Initialize the MDP
    print 'Loading environment', env_name
    mdp = rlgymenv.RLGymMDP(env_name)
    print 'MDP observation space, action space sizes: %d, %d\n' % (mdp.obs_space.dim, mdp.action_space.storage_size)

    # Initialize the policy
    nn.reset_global_scope()
    enable_obsnorm = bool(train_args['enable_obsnorm']) if 'enable_obsnorm' in train_args else train_args['obsnorm_mode'] != 'none'
    if isinstance(mdp.action_space, policyopt.ContinuousSpace):
        policy_cfg = rl.GaussianPolicyConfig(
            hidden_spec=train_args['policy_hidden_spec'],
            min_stdev=0.,
            init_logstdev=0.,
            enable_obsnorm=enable_obsnorm)
        policy = rl.GaussianPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GaussianPolicy')
    else:
        policy_cfg = rl.GibbsPolicyConfig(
            hidden_spec=train_args['policy_hidden_spec'],
            enable_obsnorm=enable_obsnorm)
        policy = rl.GibbsPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GibbsPolicy')

    # Load the policy parameters
    policy.load_h5(policy_file, policy_key)

    return mdp, policy, train_args

def gen_taskname2outfile(spec, assert_not_exists=False):
    '''
    Generate dataset filenames for each task. Phase 0 (sampling) writes to these files,
    phase 1 (training) reads from them.
    '''
    taskname2outfile = {}
    trajdir = os.path.join(spec['options']['storagedir'], spec['options']['traj_subdir'])
    util.mkdir_p(trajdir)
    for task in spec['tasks']:
        assert task['name'] not in taskname2outfile
        fname = os.path.join(trajdir, 'trajs_{}.h5'.format(task['name']))
        # if assert_not_exists:
        #     assert not os.path.exists(fname), 'Traj destination {} already exists'.format(fname)
        taskname2outfile[task['name']] = fname
    return taskname2outfile



def exec_saved_policy(env_name, policystr, num_trajs, deterministic, max_traj_len=None):
    import policyopt
    from policyopt import SimConfig, rl, util, nn, tqdm
    from environments import rlgymenv
    import gym

    # Load MDP and policy
    mdp, policy, _ = load_trained_policy_and_mdp(env_name, policystr)
    max_traj_len = min(mdp.env_spec.timestep_limit, max_traj_len) if max_traj_len is not None else mdp.env_spec.timestep_limit

    print 'Sampling {} trajs (max len {}) from policy {} in {}'.format(num_trajs, max_traj_len, policystr, env_name)

    # Sample trajs
    trajbatch = mdp.sim_mp(
        policy_fn=lambda obs_B_Do: policy.sample_actions(obs_B_Do, deterministic),
        obsfeat_fn=lambda obs:obs,
        cfg=policyopt.SimConfig(
            min_num_trajs=num_trajs,
            min_total_sa=-1,
            batch_size=None,
            max_traj_len=max_traj_len))

    return trajbatch, policy, mdp

#with open('./pipelines/im_classic_pipeline_kj.yaml', 'r') as f:
#with open('./pipelines/im_classic_pipeline_sparse.yaml', 'r') as f:
# with open('./pipelines/im_test_pipeline.yaml','r') as f:
with open('./pipelines/im_ga_mixture3_reacher.yaml','r') as f:
    spec = yaml.load(f)

util.header('=== Phase 0: Sampling trajs from expert policies ===')

num_trajs = spec['training']['full_dataset_num_trajs']
util.header('Sampling {} trajectories'.format(num_trajs))

# Make filenames and check if they're valid first
taskname2outfile = gen_taskname2outfile(spec, assert_not_exists=True)

# Sample trajs for each task
for task in spec['tasks']:
    # Execute the policy
    trajbatch, policy, _ = exec_saved_policy(
        task['env'], task['policy'], num_trajs,
        deterministic=spec['training']['deterministic_expert'],
        max_traj_len=None)

    # Quick evaluation
    returns = trajbatch.r.padded(fill=0.).sum(axis=1)
    avgr = trajbatch.r.stacked.mean()
    lengths = np.array([len(traj) for traj in trajbatch])
    ent = policy._compute_actiondist_entropy(trajbatch.adist.stacked).mean()
    print 'ret: {} +/- {}'.format(returns.mean(), returns.std())
    print 'avgr: {}'.format(avgr)
    print 'len: {} +/- {}'.format(lengths.mean(), lengths.std())
    print 'ent: {}'.format(ent)

    # Save the trajs to a file
    with h5py.File(taskname2outfile[task['name']], 'w') as f:
        def write(dsetname, a):
            f.create_dataset(dsetname, data=a, compression='gzip', compression_opts=9)
        # Right-padded trajectory data
        write('obs_B_T_Do', trajbatch.obs.padded(fill=0.))
        write('a_B_T_Da', trajbatch.a.padded(fill=0.))
        write('r_B_T', trajbatch.r.padded(fill=0.))
        # Trajectory lengths
        write('len_B', np.array([len(traj) for traj in trajbatch], dtype=np.int32))
        # # Also save args to this script
        # argstr = json.dumps(vars(args), separators=(',', ':'), indent=2)
        # f.attrs['args'] = argstr
    util.header('Wrote {}'.format(taskname2outfile[task['name']]))