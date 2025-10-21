# Code on evaluating models against adversarial attacks
# eval against val/test set

import hydra
from copy import deepcopy
import torch
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
from itertools import cycle
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from omegaconf import OmegaConf
import wandb
# logging
from src.utils import set_seed, load_model_explicit
from src.eval import robustness_evaluation
from src.utils import set_seed, get_run_hash, save_omega_config
from train import load_datamodule
from collections import defaultdict
import gc

def wandb_setup(cfg):
    dict_args = deepcopy(dict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)))
    if cfg.wandb.enabled:
        # Initialize Wandb logging
        logger = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.run_name,
            dir=cfg.wandb.save_dir,
            tags=[cfg.wandb.tag],
            config=dict_args,
            mode="online",
        )
    else:
        logger = False
    return logger

def plot_stats2(pivoted_df, attack, cfg, metric: str = "ACC"):
    # colors
    # Refer to https://plotly.com/python/discrete-color/
    palette = cycle(px.colors.qualitative.Vivid)
    # px.colors.qualitative.Light24
    #palette = cycle(px.colors.sequential.Magma)
                    
    epsilons = pivoted_df.columns
    i=0
    fig = go.Figure()
    for model in pivoted_df.index:
        adv_acc_values = pivoted_df.iloc[i,:].round(2).to_list()
        fig.add_trace(go.Bar(x=epsilons, y=adv_acc_values, name=model,
                            marker_color=next(palette),
                        hovertemplate="Model Description=%s<br>epsilon=%%{x}<br>Robust Test Accuracy=%%{y}<extra></extra>"% model))
        i=i+1
    title = cfg.general_attack_type + ": " + attack
    fig.update_layout(title=title, legend=dict(yanchor="top", y=8, xanchor="left", x=0),
                      width=1600, height=600)
    fig.update_xaxes(title_text="epsilon/severity",  type='category')
    fig.update_yaxes(title_text="Robust Test Accuracy")

    # fig.show()
    fig_path = "./assets/vis/tmp_plot.png"
    # makedirs if not exists
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.write_image(fig_path)
    # wandb.log({ f"{cfg.general_attack_type}/{attack}-Robust-{metric}" : wandb.Image(fig_path) })
    

def is_corruption_attack(attack_cfg):
    """Check if the attack is a corruption-based attack"""
    return attack_cfg.attack.attack_type == "corruption"

       
def run_attacks(cfg, datamodule) -> None:
    torch.set_float32_matmul_precision('medium')
    results_dict = defaultdict(list)
    
    # Check if we have any attacks configured
    if not hasattr(cfg, 'Attacks') or not cfg.Attacks:
        print("WARNING: No attacks configured. Skipping attack evaluation.")
        return
    
    # setup logging for the model
    cfg.run_hash = get_run_hash(cfg)
    cfg.run_name = f"{cfg.dataset.name}-{cfg.general_attack_type}-{cfg.file_name.replace('.pth','')}-n_seeds-{len(cfg.seeds)}-{cfg.run_hash}"
    save_omega_config(cfg, save_path=Path(cfg.result_file_path) / cfg.dataset.name / (cfg.run_name + ".yaml"))
    logger = wandb_setup(cfg)
    
    for model_idx, m in enumerate(cfg.Models):
        print("MODEL: ", m.model)
        print("MODEL DESC: ", m.model.model_desc)
        print("MODEL PATH: ", m.model.model_path)
        
        model_start_idx = len(results_dict["models"])
        
        # load model
        pl_module, pl_cfg = load_model_explicit(m.model.model_path, datamodule, return_cfg=True)
        pl_module.eval()
        if pl_cfg.training_type.lower() == "split_train":
            try:
                if m.model.no_betas:
                    print("Not using beta weights for adversarial robustness eval")
                    pl_module.model.feature_hook_fn = lambda x:x
            except:
                m.model.no_betas = False
                
        # Separate adversarial and corruption attacks
        adversarial_attacks = [attack for attack in cfg.Attacks if not is_corruption_attack(attack)]
        corruption_attacks = [attack for attack in cfg.Attacks if is_corruption_attack(attack)]
        
        print(f"Found {len(adversarial_attacks)} adversarial attacks and {len(corruption_attacks)} corruption attacks")
        
        # Run adversarial attacks with epsilons
        if adversarial_attacks and hasattr(cfg, 'epsilons') and cfg.epsilons:
            print("Running adversarial attacks...")
            for e in cfg.epsilons:
                eps_i = e/cfg.attacks_denominator
                print("EPSILON: ", eps_i)
                # string as fraction for logging
                eps_as_frac_str = f"{e}/{cfg.attacks_denominator}"
                
                for attack_cfg in adversarial_attacks:
                    val_rob_accs_i = []
                    test_rob_accs_i = []
                    for seed in cfg.seeds:
                        attack_cfg_copy = deepcopy(attack_cfg)
                        set_seed(seed)
                        print("ATTACK: ", attack_cfg_copy.attack.desc, "with seed ", seed, " and epsilon ", eps_as_frac_str)
                        if attack_cfg_copy.attack.attack_type == "autoattack":
                            attack_cfg_copy.attack.seed = seed
                            attack_cfg_copy.attack.n_classes = cfg.dataset.n_classes

                        attack_cfg_copy.attack={**attack_cfg_copy.attack,**cfg.attack_domain,'epsilon':eps_i}
                        if attack_cfg_copy.attack.attack_type == "pgd" and attack_cfg_copy.attack.pgd_alpha is None:
                            attack_cfg_copy.attack.pgd_alpha = eps_i / 2
                        merged_dict = {**attack_cfg_copy, 'dataset':cfg.dataset}

                        val_robust_accuracy = robustness_evaluation(pl_module, datamodule.val_dataloader, merged_dict)
                        test_robust_accuracy = robustness_evaluation(pl_module, datamodule.test_dataloader, merged_dict)
                        
                        val_rob_accs_i.append(val_robust_accuracy)
                        test_rob_accs_i.append(test_robust_accuracy)
                        print(f"VAL-ACC: {val_robust_accuracy:.2f} | TEST-ACC: {test_robust_accuracy:.2f}")    

                    results_dict["val_rob_accs"].append(np.mean(val_rob_accs_i))
                    results_dict["test_rob_accs"].append(np.mean(test_rob_accs_i))
                    results_dict["val_rob_accs_std"].append(np.std(val_rob_accs_i))
                    results_dict["test_rob_accs_std"].append(np.std(test_rob_accs_i))
                    results_dict["models"].append(m.model.model_path)
                    results_dict["models_desc"].append(m.model.model_desc)
                    results_dict["epsilons"].append(eps_i)
                    results_dict["eps_fracs"].append(eps_as_frac_str)
                    results_dict["attacks"].append(attack_cfg.attack.desc)
        elif adversarial_attacks:
            print("WARNING: Adversarial attacks configured but no epsilons specified. Skipping adversarial attacks.")
        
        # Run corruption attacks with severities
        if corruption_attacks and hasattr(cfg, 'corruption_severities') and cfg.corruption_severities:
            print("Running corruption attacks...")
            for severity in cfg.corruption_severities:
                print("SEVERITY: ", severity)
                severity_str = str(severity)
                
                for attack_cfg in corruption_attacks:
                    val_rob_accs_i = []
                    test_rob_accs_i = []
                    for seed in cfg.seeds:
                        attack_cfg_copy = deepcopy(attack_cfg)
                        set_seed(seed)
                        print("CORRUPTION: ", attack_cfg_copy.attack.desc, "with seed ", seed, " and severity ", severity)

                        # Set corruption-specific parameters
                        attack_cfg_copy.attack = {**attack_cfg_copy.attack, **cfg.attack_domain, 'severity': severity}
                        merged_dict = {**attack_cfg_copy, 'dataset': cfg.dataset}

                        val_robust_accuracy = robustness_evaluation(pl_module, datamodule.val_dataloader, merged_dict)
                        test_robust_accuracy = robustness_evaluation(pl_module, datamodule.test_dataloader, merged_dict)
                        
                        val_rob_accs_i.append(val_robust_accuracy)
                        test_rob_accs_i.append(test_robust_accuracy)
                        print(f"VAL-ACC: {val_robust_accuracy:.2f} | TEST-ACC: {test_robust_accuracy:.2f}")    

                    results_dict["val_rob_accs"].append(np.mean(val_rob_accs_i))
                    results_dict["test_rob_accs"].append(np.mean(test_rob_accs_i))
                    results_dict["val_rob_accs_std"].append(np.std(val_rob_accs_i))
                    results_dict["test_rob_accs_std"].append(np.std(test_rob_accs_i))
                    results_dict["models"].append(m.model.model_path)
                    results_dict["models_desc"].append(m.model.model_desc)
                    results_dict["epsilons"].append(severity)  # Store severity in epsilon field for consistency
                    results_dict["eps_fracs"].append(severity_str)  # Store severity as string
                    results_dict["attacks"].append(attack_cfg.attack.desc)
        elif corruption_attacks:
            print("WARNING: Corruption attacks configured but no corruption_severities specified. Skipping corruption attacks.")
        
        # Check if any attacks were actually run
        if len(results_dict["models"]) == model_start_idx:
            print(f"WARNING: No attacks were run for model {m.model.model_desc}. Check configuration.")
        else:
            # Save results for this specific model
            save_model_results(results_dict, model_start_idx, m.model.model_desc, cfg)
        
        #free up cuda memory
        del pl_module
        del pl_cfg
        gc.collect()        
        torch.cuda.empty_cache()
        
    # Save final aggregated results only if we have results
    if results_dict["models"]:
        save_final_results(results_dict, cfg)
    else:
        print("WARNING: No results to save. No attacks were successfully run.")

    if logger:
        logger.finish()


def save_model_results(results_dict, model_start_idx, model_desc, cfg):
    """Save results for a single model"""
    # Extract results for this model only
    model_results = {}
    for key, values in results_dict.items():
        model_results[key] = values[model_start_idx:]
    
    # Check if we have any results for this model
    if not model_results.get("models"):
        print(f"WARNING: No results to save for model {model_desc}")
        return
    
    # Create mean+- std string for this model
    test_acc_str = [f"{mean:.2f}±{std:.3f}" for mean, std in zip(model_results["test_rob_accs"], model_results["test_rob_accs_std"])]
    val_acc_str = [f"{mean:.2f}±{std:.3f}" for mean, std in zip(model_results["val_rob_accs"], model_results["val_rob_accs_std"])]
    
    # Create results dict for this model
    model_results_dict = {
        'Model': model_results["models"], 
        'Model Description': model_results["models_desc"],
        'attack': model_results["attacks"],
        'epsilon': model_results["epsilons"],
        'epsilon_frac': model_results["eps_fracs"],
        'VAL_Robust_ACC': model_results["val_rob_accs"],
        'TEST_Robust_ACC': model_results["test_rob_accs"],
        'VAL_Robust_ACC_STD': model_results["val_rob_accs_std"],
        'TEST_Robust_ACC_STD': model_results["test_rob_accs_std"],
        'TEST_Robust_ACC_STRING': test_acc_str,
        'VAL_Robust_ACC_STRING': val_acc_str,
    }
    
    df_model_results = pd.DataFrame(model_results_dict)
    
    # Create safe filename from model description
    safe_model_desc = "".join(c for c in model_desc if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_model_desc = safe_model_desc.replace(' ', '_')
    
    # Save individual model results
    model_filename = f"{cfg.general_attack_type}_{cfg.file_name}_{safe_model_desc}_{cfg.run_hash}.pandas_pth"
    torch.save(df_model_results, Path(cfg.result_file_path) / cfg.dataset.name / model_filename)
    
    # Create and save pivot tables for this model
    model_pivtables = {}
    for attack in df_model_results.attack.unique():
        print(f"Creating pivot table for model {model_desc}, attack: {attack}")
        at_df = df_model_results.loc[df_model_results['attack'] == attack]
        pivoted_test = at_df.pivot(index='Model Description', columns='epsilon_frac', values='TEST_Robust_ACC_STRING')
        pivoted_val = at_df.pivot(index='Model Description', columns='epsilon_frac', values='VAL_Robust_ACC_STRING')
        
        model_pivtables[attack+"-Test"] = pivoted_test
        model_pivtables[attack+"-Val"] = pivoted_val
    
    # Save pivot tables for this model
    pivot_filename = f"{cfg.general_attack_type}_{cfg.file_name}_{safe_model_desc}_{cfg.run_hash}.pth"
    torch.save(model_pivtables, Path(cfg.result_file_path) / cfg.dataset.name / pivot_filename)
    
    print(f"Saved results for model: {model_desc}")


def save_final_results(results_dict, cfg):
    """Save final aggregated results"""
    # Check if we have any results to save
    if not results_dict.get("models"):
        print("WARNING: No results to save in final results")
        return
    
    # create mean+- std string
    test_acc_str = [f"{mean:.2f}±{std:.3f}" for mean, std in zip(results_dict["test_rob_accs"], results_dict["test_rob_accs_std"])]
    val_acc_str = [f"{mean:.2f}±{std:.3f}" for mean, std in zip(results_dict["val_rob_accs"], results_dict["val_rob_accs_std"])]
    
    # create results dict
    final_results_dict = {'Model': results_dict["models"], 
                    'Model Description': results_dict["models_desc"],
                    'attack': results_dict["attacks"],
                    'epsilon': results_dict["epsilons"],
                    'epsilon_frac': results_dict["eps_fracs"],
                    'VAL_Robust_ACC': results_dict["val_rob_accs"],
                    'TEST_Robust_ACC': results_dict["test_rob_accs"],
                    'VAL_Robust_ACC_STD': results_dict["val_rob_accs_std"],
                    'TEST_Robust_ACC_STD': results_dict["test_rob_accs_std"],
                    'TEST_Robust_ACC_STRING': test_acc_str,
                    'VAL_Robust_ACC_STRING': val_acc_str,
                    }  
    df_results = pd.DataFrame(final_results_dict)
    torch.save(df_results, Path(cfg.result_file_path) / cfg.dataset.name / (cfg.general_attack_type + "_" + cfg.file_name + "_" + cfg.run_hash + ".pandas_pth") )
    
    # print(df_results)

    pivtables = {}
    for attack in df_results.attack.unique():
        print("For attack: ", attack)
        at_df = df_results.loc[df_results['attack'] == attack]
        pivoted_test = at_df.pivot(index='Model Description', columns='epsilon_frac', values='TEST_Robust_ACC_STRING')
        pivoted_val = at_df.pivot(index='Model Description', columns='epsilon_frac', values='VAL_Robust_ACC_STRING')
        
        pivtables[attack+"-Test"] = pivoted_test
        pivtables[attack+"-Val"] = pivoted_val
        
        plot_stats2(pivoted_test, attack, cfg, metric="TEST-ACC")
        plot_stats2(pivoted_val, attack, cfg, metric="VAL-ACC")
        
    torch.save(pivtables, Path(cfg.result_file_path) / cfg.dataset.name / (cfg.general_attack_type + "_" + cfg.file_name + "_" + cfg.run_hash + ".pth") )


@hydra.main(version_base=None, config_path="configs", config_name="attack_config")
def main(cfg: DictConfig) -> None:
    datamodule = load_datamodule(cfg, setup=True)
    
    # Determine which attack types to run based on configured epsilons/severities
    run_partial = hasattr(cfg, 'partial_attack_epsilons') and cfg.partial_attack_epsilons
    run_block = hasattr(cfg, 'block_attack_epsilons') and cfg.block_attack_epsilons  
    run_full = hasattr(cfg, 'full_attack_epsilons') and cfg.full_attack_epsilons
    run_corruption = hasattr(cfg, 'corruption_severities') and cfg.corruption_severities
    
    print(f"Attack types to run: Partial={run_partial}, Block={run_block}, Full={run_full}, Corruption={run_corruption}")
    
    # Run Block Attacks (if configured)
    if run_block:
        print("Running Block Attacks...")
        cfg_copy = deepcopy(cfg)
        cfg_copy.epsilons = cfg.block_attack_epsilons
        cfg_copy.general_attack_type = "BlockAttack"
        cfg_copy.attack_domain.attack_protected = True
        cfg_copy.attack_domain.attack_right = False  
        cfg_copy.attack_domain.attack_box = True
        run_attacks(cfg_copy, datamodule)
    
    # Run Partial Attacks (if configured)  
    if run_partial:
        print("Running Partial Attacks...")
        cfg_copy = deepcopy(cfg)
        cfg_copy.epsilons = cfg.partial_attack_epsilons
        cfg_copy.general_attack_type = "PartialAttack"
        cfg_copy.attack_domain.attack_protected = True
        cfg_copy.attack_domain.attack_right = False
        cfg_copy.attack_domain.attack_box = False
        run_attacks(cfg_copy, datamodule)
    
    # Run Full Attacks (if configured)
    if run_full:
        print("Running Full Attacks...")
        cfg_copy = deepcopy(cfg)
        cfg_copy.epsilons = cfg.full_attack_epsilons  
        cfg_copy.general_attack_type = "FullAttack"
        cfg_copy.attack_domain.attack_protected = False
        cfg_copy.attack_domain.attack_right = False
        cfg_copy.attack_domain.attack_box = False
        run_attacks(cfg_copy, datamodule)
    
    # Run Corruption Attacks (if configured)
    if run_corruption:
        print("Running Corruption Attacks...")
        cfg_copy = deepcopy(cfg)
        # Use corruption_severities instead of epsilons for corruption attacks
        # The run_attacks function will use corruption_severities directly
        
        # Check if we should attack full image or protected region for corruptions
        attack_full_image = getattr(cfg, 'corruption_attack_full_image', True)  # Default to full image
        
        if attack_full_image:
            cfg_copy.general_attack_type = "FullCorruption"
            cfg_copy.attack_domain.attack_protected = False
            cfg_copy.attack_domain.attack_right = False
            cfg_copy.attack_domain.attack_box = False
        else:
            cfg_copy.general_attack_type = "PartialCorruption" 
            cfg_copy.attack_domain.attack_protected = True
            cfg_copy.attack_domain.attack_right = False
            cfg_copy.attack_domain.attack_box = False
            
        run_attacks(cfg_copy, datamodule)
    
    # Print warning if no attacks are configured
    if not any([run_partial, run_block, run_full, run_corruption]):
        print("WARNING: No attacks configured to run. Please specify at least one of:")
        print("  - partial_attack_epsilons")  
        print("  - block_attack_epsilons")
        print("  - full_attack_epsilons")
        print("  - corruption_severities")

if __name__ == "__main__":
    main()