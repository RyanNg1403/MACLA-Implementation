#!/usr/bin/env python3
# MACLA_Generalized.py
# Domain-agnostic version - works for ALFWorld, WebShop, IC-SQL

import os
import sys
import random
import logging
import warnings
import argparse

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from macla import (
    LLMMACLAAgent,
    ALFWorldLikeLoader,
    WebShopLoader,
    TravelPlannerLoader,
    SQLLoader,
    build_agent_and_learn,
    run_evaluation,
    pretty_print_metrics,
)


def main():
    # ========================================
    # CONFIGURATION: Choose ONE option below
    # ========================================

    if len(sys.argv) == 1:
        # OPTION 1: ALFWorld only (3 separate files)
        # sys.argv.extend([
        #     "--dataset", "alfworld",
        #     "--train_file", r"C:\ALFWord_data\train_reactified.txt",
        #     "--valid_seen_file", r"C:\ALFWord_data\valid_seen_reactified.txt",
        #     "--valid_unseen_file", r"C:\ALFWord_data\valid_unseen_reactified.txt",
        #     "--no_llm"
        # ])

        # OPTION 2: WebShop only (1 file, auto-split)
        # sys.argv.extend([
        #     "--dataset", "webshop",
        #     "--webshop_file", r"C:\Webshop_data\webshop_sft.txt",
        #     "--no_llm"
        # ])

        # OPTION 3: TravelPlanner only (2 files, auto-split)
        sys.argv.extend([
            "--dataset", "travelplanner",
            "--travelplanner_test_file", r"C:\Travel Planner\travelplanner_test_synthetic_trajectories.txt",
            "--travelplanner_val_file", r"C:\Travel Planner\travelplanner_validation_synthetic_trajectories.txt",
            # "--no_llm"
        ])

        # OPTION 4: SQL only (1 file, auto-split)
        # sys.argv.extend([
        #     "--dataset", "sql",
        #     "--sql_file", r"C:\intercode_sql\intercode_sql.txt",
        #     "--no_llm"
        # ])

    parser = argparse.ArgumentParser(description="Domain-Agnostic MACLA Agent")
    parser.add_argument("--dataset", type=str, default="alfworld",
                       choices=["alfworld", "webshop", "travelplanner", "sql", "all"],
                       help="Dataset to use: alfworld, webshop, travelplanner, sql, or all")

    # ALFWorld specific
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--valid_seen_file", type=str, default=None)
    parser.add_argument("--valid_unseen_file", type=str, default=None)

    # WebShop specific
    parser.add_argument("--webshop_file", type=str, default=None,
                       help="WebShop data file (will be split automatically)")

    # TravelPlanner specific
    parser.add_argument("--travelplanner_test_file", type=str, default=None,
                       help="TravelPlanner test file")
    parser.add_argument("--travelplanner_val_file", type=str, default=None,
                       help="TravelPlanner validation file")

    # SQL specific
    parser.add_argument("--sql_file", type=str, default=None,
                       help="SQL data file (will be split automatically)")

    parser.add_argument("--llm_model", type=str, default=os.environ.get("MACLA_LLM_MODEL", "llama2"))
    parser.add_argument("--no_llm", action="store_true")
    parser.add_argument("--ablation", action="store_true",
                       help="Run ablation studies")

    args = parser.parse_args()

    # Initialize loaders
    alfworld_loader = ALFWorldLikeLoader()
    webshop_loader = WebShopLoader()
    travelplanner_loader = TravelPlannerLoader()
    sql_loader = SQLLoader()

    train_traj = []
    valid_seen = []
    valid_unseen = []

    # ========================================
    # PHASE 1: Load Data Based on Dataset Type
    # ========================================
    logger.info("="*60)
    logger.info("PHASE 1: Loading Training Data")
    logger.info(f"Dataset: {args.dataset.upper()}")
    logger.info("="*60)

    if args.dataset == "alfworld":
        if not args.train_file:
            logger.error("--train_file required for ALFWorld dataset")
            return

        train_traj = alfworld_loader.load_files([args.train_file], include_trajectory_paths=True)

        if args.valid_seen_file:
            valid_seen = alfworld_loader.load_files([args.valid_seen_file], include_trajectory_paths=True)
        if args.valid_unseen_file:
            valid_unseen = alfworld_loader.load_files([args.valid_unseen_file], include_trajectory_paths=True)

    elif args.dataset == "webshop":
        if not args.webshop_file:
            logger.error("--webshop_file required for WebShop dataset")
            return

        splits = webshop_loader.load_and_split_webshop(
            args.webshop_file,
            train_ratio=0.7,
            val_seen_ratio=0.15,
            val_unseen_ratio=0.15
        )
        train_traj = splits["train"]
        valid_seen = splits["val_seen"]
        valid_unseen = splits["val_unseen"]

    elif args.dataset == "travelplanner":
        logger.info("Loading TravelPlanner dataset")

        all_data = []

        if args.travelplanner_test_file:
            test_data = travelplanner_loader._parse_travelplanner_file(
                args.travelplanner_test_file
            )
            all_data.extend(test_data)
            logger.info(f"Loaded {len(test_data)} from test file")

        if args.travelplanner_val_file:
            val_data = travelplanner_loader._parse_travelplanner_file(
                args.travelplanner_val_file
            )
            all_data.extend(val_data)
            logger.info(f"Loaded {len(val_data)} from validation file")

        logger.info(f"=== TRAVELPLANNER DATA CHECK ===")
        logger.info(f"Total trajectories loaded: {len(all_data)}")
        if all_data:
            logger.info(f"Sample task: {all_data[0].get('task', 'NO TASK')[:100]}")
            logger.info(f"Sample actions: {all_data[0].get('actions', [])[:3]}")
        else:
            logger.error("⚠️ NO TRAVELPLANNER DATA LOADED - Files may be empty or parse failed!")

        random.seed(42)
        random.shuffle(all_data)
        n_train = int(len(all_data) * 0.7)
        n_val = int(len(all_data) * 0.15)

        train_traj = all_data[:n_train]
        valid_seen = all_data[n_train:n_train + n_val]
        valid_unseen = all_data[n_train + n_val:]
        logger.info(f"Split: {len(train_traj)} train, {len(valid_seen)} val1, {len(valid_unseen)} val2")

    elif args.dataset == "sql":
        if not args.sql_file:
            logger.error("--sql_file required for SQL dataset")
            return

        splits = sql_loader.load_and_split_sql(
            args.sql_file,
            train_ratio=0.7,
            val_seen_ratio=0.15,
            val_unseen_ratio=0.15,
            use_gold_as_success=True
        )
        train_traj = splits["train"]
        valid_seen = splits["val_seen"]
        valid_unseen = splits["val_unseen"]

        logger.info(f"\nSQL Dataset Analysis:")
        logger.info(f"  Tables extracted: {len(set(t for traj in train_traj for t in traj.get('sql_tables', [])))}")
        logger.info(f"  Operations found: {len(set(op for traj in train_traj for op in traj.get('sql_operations', [])))}")

    elif args.dataset == "all":
        logger.info("Loading ALL datasets: ALFWorld + WebShop + TravelPlanner + SQL")

        if args.train_file:
            alfworld_train = alfworld_loader.load_files([args.train_file], include_trajectory_paths=True)
            train_traj.extend(alfworld_train)
            logger.info(f"Added {len(alfworld_train)} ALFWorld training trajectories")

        if args.webshop_file:
            webshop_splits = webshop_loader.load_and_split_webshop(args.webshop_file)
            train_traj.extend(webshop_splits["train"])
            valid_seen.extend(webshop_splits["val_seen"])
            valid_unseen.extend(webshop_splits["val_unseen"])
            logger.info(f"Added {len(webshop_splits['train'])} WebShop training trajectories")

        if args.travelplanner_test_file:
            all_tp_data = []
            test_data = travelplanner_loader._parse_travelplanner_file(args.travelplanner_test_file)
            all_tp_data.extend(test_data)
            if args.travelplanner_val_file:
                val_data = travelplanner_loader._parse_travelplanner_file(args.travelplanner_val_file)
                all_tp_data.extend(val_data)

            random.seed(42)
            random.shuffle(all_tp_data)
            n_train = int(len(all_tp_data) * 0.7)
            n_val = int(len(all_tp_data) * 0.15)
            train_traj.extend(all_tp_data[:n_train])
            valid_seen.extend(all_tp_data[n_train:n_train + n_val])
            valid_unseen.extend(all_tp_data[n_train + n_val:])
            logger.info(f"Added {n_train} TravelPlanner training trajectories")

        if args.sql_file:
            sql_splits = sql_loader.load_and_split_sql(args.sql_file)
            train_traj.extend(sql_splits["train"])
            valid_seen.extend(sql_splits["val_seen"])
            valid_unseen.extend(sql_splits["val_unseen"])
            logger.info(f"Added {len(sql_splits['train'])} SQL training trajectories")

        if args.valid_seen_file:
            alfworld_seen = alfworld_loader.load_files([args.valid_seen_file], include_trajectory_paths=True)
            valid_seen.extend(alfworld_seen)
            logger.info(f"Added {len(alfworld_seen)} ALFWorld seen validation trajectories")

        if args.valid_unseen_file:
            alfworld_unseen = alfworld_loader.load_files([args.valid_unseen_file], include_trajectory_paths=True)
            valid_unseen.extend(alfworld_unseen)
            logger.info(f"Added {len(alfworld_unseen)} ALFWorld unseen validation trajectories")

    if not train_traj:
        logger.error("No training trajectories loaded. Exiting.")
        return

    logger.info(f"Total training trajectories: {len(train_traj)}")
    if valid_seen:
        logger.info(f"Total validation (seen) trajectories: {len(valid_seen)}")
    if valid_unseen:
        logger.info(f"Total validation (unseen) trajectories: {len(valid_unseen)}")

    # ========================================
    # DATA SANITY CHECK
    # ========================================
    logger.info("\n=== DATA SANITY CHECK ===")
    success_count = sum(1 for t in train_traj if t.get('success', False))
    logger.info(f"Training: {success_count}/{len(train_traj)} successful ({success_count/len(train_traj)*100:.1f}%)")

    if valid_seen:
        success_count = sum(1 for t in valid_seen if t.get('success', False))
        logger.info(f"Val Seen: {success_count}/{len(valid_seen)} successful ({success_count/len(valid_seen)*100:.1f}%)")

    if valid_unseen:
        success_count = sum(1 for t in valid_unseen if t.get('success', False))
        logger.info(f"Val Unseen: {success_count}/{len(valid_unseen)} successful ({success_count/len(valid_unseen)*100:.1f}%)")

    logger.info("\nSample trajectory inspection:")
    for i, traj in enumerate(train_traj[:3]):
        logger.info(f"  Train {i}: ID={traj.get('id')[:30]}, success={traj.get('success')}, "
                    f"actions={len(traj.get('actions', []))}, task={traj.get('task', '')[:50]}...")

    # ========================================
    # PHASE 2: Train Domain-Agnostic Agent
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: Training Domain-Agnostic Agent")
    logger.info("="*60)

    agent = build_agent_and_learn(train_traj, llm_model=args.llm_model, use_llm=(not args.no_llm))

    logger.info("\nTraining completed.")
    train_stats = agent.get_statistics()
    logger.info(f"Procedural memory size: {train_stats['procedural_memory_size']}")
    logger.info(f"Meta-procedural memory size: {train_stats['meta_procedural_size']}")

    # ========================================
    # PHASE 2.5: Ablation Studies (Optional)
    # ========================================
    if args.ablation:
        logger.info("\n" + "="*60)
        logger.info("PHASE 2.5: Running Ablation Studies")
        logger.info("="*60)

        ablation_configs = [
            {"name": "Full System",
             "bayesian": True, "contrastive": True, "meta": True, "ontology": True},
            {"name": "No Bayesian Selection",
             "bayesian": False, "contrastive": True, "meta": True, "ontology": True},
            {"name": "No Contrastive Learning",
             "bayesian": True, "contrastive": False, "meta": True, "ontology": True},
            {"name": "No Meta-Procedures",
             "bayesian": True, "contrastive": True, "meta": False, "ontology": True},
            {"name": "No Ontology",
             "bayesian": True, "contrastive": True, "meta": True, "ontology": False},
            {"name": "Minimal (No Bayesian, No Contrastive)",
             "bayesian": False, "contrastive": False, "meta": True, "ontology": True},
        ]

        ablation_results = {}

        for config in ablation_configs:
            logger.info(f"\n--- Testing: {config['name']} ---")

            ablation_agent = LLMMACLAAgent(
                N_a=2000, N_p=500, N_m=100,
                llm_model=args.llm_model,
                use_llm=(not args.no_llm)
            )

            if hasattr(ablation_agent, 'configure_ablation'):
                ablation_agent.configure_ablation(
                    use_bayesian=config["bayesian"],
                    use_contrastive=config["contrastive"],
                    use_meta=config["meta"],
                    use_ontology=config["ontology"]
                )

            ablation_agent.bayesian_selector.build_ontology(train_traj)
            if hasattr(ablation_agent, 'learn_from_trajectories_ablation'):
                ablation_agent.learn_from_trajectories_ablation(
                    train_traj,
                    use_contrastive=config["contrastive"],
                    use_meta=config["meta"]
                )
            else:
                ablation_agent.learn_from_trajectories(train_traj)

            if valid_unseen:
                metrics = run_evaluation(ablation_agent, valid_unseen)
                ablation_results[config['name']] = {
                    "f1": metrics.f1_score,
                    "accuracy": metrics.accuracy,
                    "reward": metrics.avg_reward
                }
                logger.info(f"F1: {metrics.f1_score:.3f}, Acc: {metrics.accuracy:.3f}, Reward: {metrics.avg_reward:.3f}")

        logger.info("\n" + "="*60)
        logger.info("ABLATION STUDY RESULTS")
        logger.info("="*60)
        logger.info(f"{'Configuration':<40} {'F1':>8} {'Accuracy':>10} {'Reward':>8}")
        logger.info("-" * 70)
        for name, results in ablation_results.items():
            logger.info(f"{name:<40} {results['f1']:>8.3f} {results['accuracy']:>10.3f} {results['reward']:>8.3f}")
        logger.info("="*60 + "\n")

    # ========================================
    # PHASE 3: Evaluate on SEEN Validation
    # ========================================
    metrics_seen = None
    if valid_seen:
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: Evaluating on SEEN Validation Set")
        logger.info("="*60)

        logger.info(f"Validation (seen) trajectories: {len(valid_seen)}")
        metrics_seen = run_evaluation(agent, valid_seen)

        logger.info("\n--- SEEN Validation Results ---")
        pretty_print_metrics(metrics_seen)

    # ========================================
    # PHASE 4: Evaluate on UNSEEN Validation
    # ========================================
    metrics_unseen = None
    if valid_unseen:
        logger.info("\n" + "="*60)
        logger.info("PHASE 4: Evaluating on UNSEEN Validation Set")
        logger.info("="*60)

        logger.info(f"Validation (unseen) trajectories: {len(valid_unseen)}")
        metrics_unseen = run_evaluation(agent, valid_unseen)

        logger.info("\n--- UNSEEN Validation Results ---")
        pretty_print_metrics(metrics_unseen)

    # ========================================
    # PHASE 5: Generalization Analysis
    # ========================================
    if metrics_seen and metrics_unseen:
        logger.info("\n" + "="*60)
        logger.info("PHASE 5: Generalization Analysis")
        logger.info("="*60)

        gap_accuracy = metrics_seen.accuracy - metrics_unseen.accuracy
        gap_f1 = metrics_seen.f1_score - metrics_unseen.f1_score

        logger.info(f"\nSeen Accuracy:     {metrics_seen.accuracy:.3f}")
        logger.info(f"Unseen Accuracy:   {metrics_unseen.accuracy:.3f}")
        logger.info(f"Accuracy Gap:      {gap_accuracy:+.3f}")

        logger.info(f"\nSeen F1:           {metrics_seen.f1_score:.3f}")
        logger.info(f"Unseen F1:         {metrics_unseen.f1_score:.3f}")
        logger.info(f"F1 Gap:            {gap_f1:+.3f}")

        if abs(gap_accuracy) < 0.05:
            logger.info("\n✓ Excellent generalization (gap < 0.05)")
        elif abs(gap_accuracy) < 0.10:
            logger.info("\n✓ Good generalization (gap < 0.10)")
        elif abs(gap_accuracy) < 0.20:
            logger.info("\n⚠ Moderate generalization (gap < 0.20)")
        else:
            logger.info("\n✗ Poor generalization (gap >= 0.20) - possible overfitting")

    # ========================================
    # PHASE 6: Final Summary
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)

    final_stats = agent.get_statistics()
    logger.info(f"\nAgent Statistics:")
    logger.info(f"  Dataset(s):             {args.dataset.upper()}")
    logger.info(f"  Procedural memory:      {final_stats['procedural_memory_size']}/{agent.memory_system.N_p}")
    logger.info(f"  Meta-procedural memory: {final_stats['meta_procedural_size']}/{agent.memory_system.N_m}")
    logger.info(f"  Total executions:       {final_stats['total_executions']}")
    logger.info(f"  Successful executions:  {final_stats['successful_executions']}")

    if metrics_seen:
        logger.info(f"\nSeen validation F1:     {metrics_seen.f1_score:.3f}")
    if metrics_unseen:
        logger.info(f"Unseen validation F1:   {metrics_unseen.f1_score:.3f}")

    logger.info("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
