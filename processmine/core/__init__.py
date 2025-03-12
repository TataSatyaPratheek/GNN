# processmine/core/__init__.py
"""
Core functionality for ProcessMine including training, evaluation, and experiment management.
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