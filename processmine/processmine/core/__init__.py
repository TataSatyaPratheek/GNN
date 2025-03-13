# processmine/core/__init__.py
"""
Core functionality for ProcessMine including training, evaluation, experiment management, and execution.
"""
from processmine.core.training import (
    train_model,
    evaluate_model,
    create_optimizer,
    create_lr_scheduler,
    compute_class_weights
)

from processmine.core.experiment import (
    ExperimentManager,
    setup_results_dir,
    save_metrics,
    print_section_header
)

from processmine.core.runner import (
    run_analysis,
    run_training,
    run_optimization,
    run_full_pipeline
)

from processmine.core.ablation_runner import (
    run_ablation_study
)

from processmine.core.advanced_workflow import (
    run_advanced_workflow
)