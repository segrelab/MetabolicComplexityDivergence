#!/usr/bin/env python
# coding: utf-8

# # Modeling divergence with increasing metabolic complexity
# 
# Want to see what is required to induce divergence/complexity signal. To do so, we can create a very basic model and then incrementally add complexity. The parameters to include are:
# 1. C and D matrix
#     1. Structure: noise or trophic
#     2. Distribution: uniform or weighted (specialist are rarer)
# 2. Resources: mimic experiment. 
#     1. Categories: "cellulose", "cellobiose", "glucose", and "citrate" [trophic]. Different "complex" metabolites (cellulose and another one)?
#     1. Number: fixed
#     1. Energy desnity uniform or weighted by metabolite
#     2. Leakage: uniform or weighted by metabolite. Also leak everything or only externally degraded?
#     2. Supply rate: fixed
#     3. Initial supply: fixed
# 3. Consumers:
#     1. Number: 10, 100, or 1000
#     2. Consumption rates: uniform or lognormal (not implemented yet)
#     1. Growth rate: uniform or weighted by specialization
#     2. Initial distribution: uniform or weighted by specialization

from community_simulator.usertools import *
from community_simulator import Community
from ms_tools import crm, transform

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

from skbio.diversity.alpha import shannon
from skbio.stats.composition import clr

from multiprocessing.pool import Pool
from functools import partial

"Functions"
def TestWell(N0, R0, params, well_name, dynamics, **TestWell_kwargs):
    comm = Community((N0, R0), dynamics, params)
    t, n, r = comm.TestWell(show_plots=False, **TestWell_kwargs)
    n_df_wide = pd.DataFrame(n, t).rename_axis(index='time', columns='species')
    
    # Filter out empty species
    n_df_wide = n_df_wide.loc[:, N0.astype(bool)]
    
    # Stack
    n_df = n_df_wide.stack().reset_index(name='abundance')
    
    n_df['Sample'] = well_name

    r_df = pd.DataFrame(r, t).rename_axis(index='time', columns='resource').stack().reset_index(name='abundance')
    r_df['Sample'] = well_name
    return n_df, r_df

if __name__ == '__main__':
    "Parameterization"
    "Resource parameters"
    np.random.seed(123)
    # Resource class sizes
    n_resources = 200
    resource_class_distribution = np.array([.4, .3, .2, .1])
    resource_class_sizes = (n_resources * resource_class_distribution).astype(int)
    n_resource_classes = len(resource_class_sizes)

    # D matrix
    self_renew = .2
    exchange = 1 - self_renew
    resource_transition_matrix = np.ones((n_resource_classes, n_resource_classes)) * 1e-4
    for i in range(n_resource_classes - 1):
        resource_transition_matrix[i, i] = self_renew
        resource_transition_matrix[i, i + 1] = exchange
    # resource_transition_matrix[-1, :] = 1 / n_resource_classes
    resource_transition_matrix[-1, -1] = 1 / n_resource_classes

    d_sparsity = .8

    D_trophic = crm.TrophicResourceMatrix(resource_class_sizes, resource_transition_matrix, sparsity=d_sparsity)

    # Construct noisy D matrix
    resource_uniform_transitions = np.ones(n_resources) / n_resources / d_sparsity
    D_noisy_data = np.random.dirichlet(resource_uniform_transitions, size=n_resources)
    D_noisy = pd.DataFrame(D_noisy_data, D_trophic.index, D_trophic.columns).T

    leakage = 0.7

    "Consumer parameters"
    world_size = 1000
    community_size = 200

    # Trophic consumer matrix
    consumer_preferences = np.eye(n_resource_classes).astype(bool)
    consumer_preferences[-1, :] = True
    consumer_preferences[:, -1] = True

    # weighted_class_distributions = np.array([.1, .3, .5, .1]) May 6: Class size should correlate, not anti, with complexity
    weighted_class_distributions = np.array([.5, .3, .1, .1])
    weighted_class_sizes = (world_size * weighted_class_distributions).astype(int)

    # c_sparsity = .8
    # C_trophic = crm.TrophicConsumerMatrix(weighted_class_sizes, consumer_preferences, resource_class_sizes, sparsity=c_sparsity)
    c_n = 35
    C_trophic = crm.TrophicConsumerMatrix(weighted_class_sizes, consumer_preferences, resource_class_sizes, n=c_n)

    # Noisy consumer matrix
    # c_noisy_data = np.random.rand(world_size, n_resources) > c_sparsity
    c_noisy_data = np.zeros((world_size, n_resources))
    for s in c_noisy_data:
        c_noisy_idx = np.random.choice(range(n_resources), c_n, replace=False)
        s[c_noisy_idx] = 1
    C_noisy = pd.DataFrame(c_noisy_data, C_trophic.index, C_trophic.columns)

    "Initial conditions"
    food_amount = 1000

    # Mimic conditions from study: all single and then all mixed 
    single_conditions = [f'T{i}' for i in range(n_resource_classes)]
    mixed_conditions = ['T3+T2', 'T3+T2+T1', 'T3+T2+T1+T0']
    all_conditions = single_conditions + mixed_conditions
    n_conditions = len(all_conditions)
    condition_components = {x: x.split('+') for x in all_conditions}

    # Create initial resource abundance that will be used for all communities with a given set of C and D matrices
    r0_init = pd.Series(np.zeros(n_resources), D_trophic.index)
    R0_data = []


    # component_fraction = .4
    # initial_components = {resource_type: np.random.choice(r0_init.loc[resource_type].index, int(component_fraction * r0_init.loc[resource_type].index.size)) for resource_type in single_conditions}

    # Construct each condition by indicting which resources are present in the condition, choose a fraction of resources within that condition,
    ## and then rescale so the total "mass" (food amount) is evenly distributed across all present resources. 
    # for condition_name, components in condition_components.items():
    #     condition_data = r0_init.copy()
    #     for component in components:
    #         condition_data.loc[component, initial_components[component]] = 1
    #     R0_data.append(condition_data.rename(condition_name))


    # Construct each condition by indicting which resources are present in the condition, choose a fraction of resources within that condition,
    ## and then rescale so the total "mass" (food amount) is evenly distributed across all present resources. 
    component_size = 20
    initial_components = {condition: np.random.choice(r0_init.loc[components].index, component_size, replace=False) for condition, components in condition_components.items()}
    for condition, resources in initial_components.items():
        condition_data = r0_init.copy()
        condition_data.loc[resources] = 1
        R0_data.append(condition_data.rename(condition))

    R0s = pd.concat(R0_data, axis=1)
    # Renormalize so each condition has same total amount of food equally distributed accross components
    R0s *= food_amount / R0s.sum()

    "Initial communities"
    n_communities = 6

    # Create data for all conditions
    all_md_data = []
    all_N0s_data = []
    all_R0s_data = []
    all_params = []
    samples = []

    i = 0

    for community in range(n_communities):
        # Sample present species for this community
        community_N0 = np.zeros(world_size) # Add to larger world to all for concatenation into one table
        pres_species = np.random.choice(range(world_size), community_size, replace=False)
        community_N0[pres_species] = 1
        for condition in all_conditions:
            R0 = R0s[condition]

            for D_type, D in zip(['trophic', 'noisy'], [D_trophic, D_noisy]):

                for c_type, c in zip(['trophic', 'noisy'], [C_trophic, C_noisy]):
                    # Metadata
                    sample_number = f'S{i}'
                    samples.append(sample_number)
                    md_data = {'Sample': sample_number, 
                               'community_size': community_size,
                               'community': community,
                               'D_type': D_type,
                               'c_type': c_type,
                               'condition': condition
                              }
                    all_md_data.append(md_data)
                    i += 1

                    ## Community simulator parameters
                    params = {'c': c, 'D': D, 'R0': R0, 'm': 1, 'g': 1, 'w': 1, 'tau': 1, 'r': 1, 'l': leakage}
                    all_params.append(params)
                    # Initial conidtions

                    all_N0s_data.append(pd.Series(community_N0, name=sample_number))
                    all_R0s_data.append(R0.rename(sample_number))

                    # Create dataframes
    md = pd.DataFrame(all_md_data).set_index('Sample')
    md['scheme'] = md[['community_size', 'D_type', 'c_type']].astype(str).apply(lambda x: '_'.join(x), 1)

    N0s_df = pd.concat(all_N0s_data, axis=1)
    R0s_df = pd.concat(all_R0s_data, axis=1)

    "Dynamics"
    assumptions = {'response': 'type I', 'regulation': 'independent', 'supply': 'external'}
    def dNdt(N,R,params):
        return MakeConsumerDynamics(assumptions)(N,R,params)
    def dRdt(N,R,params):
        return MakeResourceDynamics(assumptions)(N,R,params)
    init_state = (N0s_df, R0s_df)
    dynamics = [dNdt, dRdt]

    "Run simulation"
    pool = Pool()
    jobs = [pool.apply_async(TestWell, (N0, R0, params, sample, dynamics), {'T': 500, 'ns': 500, 'compress_species': True}) for N0, R0, params, sample in zip(all_N0s_data, all_R0s_data, all_params, samples)]
    results = [job.get() for job in jobs]
    pool.close()
    n_df = pd.concat([r[0] for r in results])
    r_df = pd.concat([r[1] for r in results])



    n_df.to_csv('/projectnb/cometsfba/msilver/env_complexity_experiment/data/simulation/simulation_results_n_05-08_allT3.csv', index=False)
    r_df.to_csv('/projectnb/cometsfba/msilver/env_complexity_experiment/data/simulation/simulation_results_r_05-08_allT3.csv', index=False)