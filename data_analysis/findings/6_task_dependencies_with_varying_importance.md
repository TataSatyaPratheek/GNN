# Task Dependencies with Varying Importance in Different Contexts

## Overview

Within the BPI2020 Domestic Declarations process, task dependencies exhibit varying levels of importance depending on the context and perspective from which the process is viewed. This document explores how different aspects of task dependencies require selective focus in various situations, and illustrates this with specific examples from the declarations process.

## 1. Contextual Importance of Dependencies

Task dependencies in business processes don't have uniform importance across all contexts. Their significance varies based on:

1. **Organizational Objective** (operational efficiency, compliance, cost reduction)
2. **Stakeholder Perspective** (employee, manager, auditor, finance department)
3. **Time Horizon** (daily operations, monthly planning, strategic improvement)
4. **Process State** (normal operation, exception handling, peak load)
5. **External Factors** (regulatory changes, seasonal patterns, resource availability)

The BPI2020 Domestic Declarations process clearly demonstrates this varying importance through multiple examples.

## 2. Selective Focus for Different Organizational Roles

### 2.1 Employee Perspective

For employees submitting declarations, certain dependencies have heightened importance:

```graphviz
digraph G {
    rankdir=LR;
    node [shape=box];
    
    // Employee-focused activities
    submit [label="Declaration\nSubmission", color=blue, penwidth=2];
    admin_approve [label="Administrative\nApproval"];
    admin_reject [label="Administrative\nRejection", color=red, penwidth=2];
    super_approve [label="Supervisor\nApproval"];
    payment [label="Payment\nHandled", color=green, penwidth=2];
    emp_reject [label="Employee\nRejection/Resubmission", color=red, penwidth=2];
    
    // Dependencies with employee focus
    submit -> admin_approve;
    submit -> admin_reject [color=red, penwidth=2];
    admin_reject -> emp_reject [color=red, penwidth=2];
    emp_reject -> submit [label="Resubmit", color=red, penwidth=2];
    admin_approve -> super_approve;
    super_approve -> payment;
    payment -> end [color=green, penwidth=2];
    
    end [shape=circle];
    
    // Focus explanation
    label = "Employee Focus: Submission requirements, rejection handling, and payment receipt";
    labelloc = "t";
}
```

**Key Dependencies for Employees:**
- Submission → Administrative Approval/Rejection (will my submission be accepted?)
- Administrative Rejection → Resubmission requirements (what must I fix?)
- Final Approval → Payment (when will I be reimbursed?)

**Less Important Dependencies:**
- Specific approval routing (which manager approves?)
- Internal system processes (how is payment processed?)

### 2.2 Manager Perspective

For supervisors and budget owners approving declarations, different dependencies become critical:

```graphviz
digraph G {
    rankdir=LR;
    node [shape=box];
    
    // Manager-focused activities
    submit [label="Declaration\nSubmission"];
    admin_approve [label="Administrative\nApproval", color=blue, penwidth=2];
    budget_approve [label="Budget Owner\nApproval", color=blue, penwidth=2];
    super_approve [label="Supervisor\nApproval", color=blue, penwidth=2];
    super_reject [label="Supervisor\nRejection", color=red, penwidth=2];
    payment [label="Payment\nHandled"];
    
    // Dependencies with manager focus
    submit -> admin_approve;
    admin_approve -> budget_approve [color=blue, penwidth=2];
    admin_approve -> super_approve [color=blue, penwidth=2];
    budget_approve -> super_approve [color=blue, penwidth=2];
    super_approve -> payment;
    super_approve -> super_reject [color=red, penwidth=2];
    
    // Budget constraints
    budget [label="Budget\nConstraints", shape=diamond, color=red, penwidth=2];
    budget -> budget_approve [color=red, penwidth=2];
    budget -> super_approve [color=red, penwidth=2];
    
    // Focus explanation
    label = "Manager Focus: Approval routing, budget constraints, and authorization requirements";
    labelloc = "t";
}
```

**Key Dependencies for Managers:**
- Administrative Approval → Budget/Supervisor Approval (proper routing)
- Budget Constraints → Approval Decisions (budget availability)
- Budget Owner Approval → Supervisor Approval (proper sequencing)
- Amount Thresholds → Approval Requirements (authorization levels)

**Less Important Dependencies:**
- Initial submission details (already vetted by administration)
- Payment execution details (handled by system)

### 2.3 Finance Department Perspective

For the finance department processing payments, another set of dependencies becomes primary:

```graphviz
digraph G {
    rankdir=LR;
    node [shape=box];
    
    // Finance-focused activities
    super_approve [label="Final\nApproval", color=blue, penwidth=2];
    request [label="Payment\nRequest", color=blue, penwidth=2];
    payment [label="Payment\nHandling", color=blue, penwidth=2];
    reconciliation [label="Accounting\nReconciliation", color=blue, penwidth=2];
    
    // Dependencies with finance focus
    super_approve -> request [color=blue, penwidth=2];
    request -> payment [color=blue, penwidth=2];
    payment -> reconciliation [color=blue, penwidth=2];
    
    // Financial constraints
    timing [label="Payment\nSchedule", shape=diamond, color=red, penwidth=2];
    budget_check [label="Budget\nVerification", shape=diamond, color=red, penwidth=2];
    
    timing -> payment [color=red, penwidth=2];
    budget_check -> payment [color=red, penwidth=2];
    
    // Focus explanation
    label = "Finance Focus: Payment processing, budget verification, and accounting integration";
    labelloc = "t";
}
```

**Key Dependencies for Finance:**
- Final Approval → Payment Request (authorization complete)
- Payment Request → Payment Handling (proper documentation)
- Payment Handling → Accounting Reconciliation (financial records)
- Payment Schedule → Payment Execution (batch processing)

**Less Important Dependencies:**
- Specific approval routing (already completed)
- Rejection handling (not finance responsibility)

## 3. Time Horizon Perspectives

The importance of dependencies also shifts depending on the time horizon under consideration.

### 3.1 Daily Operational Focus

For day-to-day operations, immediate tactical dependencies dominate:

```graphviz
digraph G {
    rankdir=LR;
    node [shape=box];
    
    // Daily operational activities with counts
    submit [label="Submissions\n(32 today)", color=blue, penwidth=2];
    admin [label="Admin Review\n(28 today)", color=blue, penwidth=2];
    approve [label="Approvals\n(25 today)", color=blue, penwidth=2];
    payment [label="Payments\n(20 today)", color=blue, penwidth=2];
    
    // Immediate operational dependencies
    submit -> admin [label="Same day", color=blue, penwidth=2];
    admin -> approve [label="0-1 days", color=blue, penwidth=2];
    approve -> payment [label="1-2 days", color=blue, penwidth=2];
    
    // Resource constraints (immediate)
    admin_res [label="Admin\nStaff (3)", shape=diamond, color=red, penwidth=2];
    approver_res [label="Approvers\nAvailable (6)", shape=diamond, color=red, penwidth=2];
    
    admin_res -> admin [color=red, penwidth=2];
    approver_res -> approve [color=red, penwidth=2];
    
    // Focus explanation
    label = "Daily Focus: Current workload, immediate resource constraints, and short-term throughput";
    labelloc = "t";
}
```

**Key Dependencies for Daily Operations:**
- Current submissions → Administrative capacity (today's workload)
- Today's approvals → Tomorrow's payment requests (short workflow)
- Available resources → Activity completion (immediate throughput)

**Less Important Dependencies:**
- Long-term process patterns (not immediately actionable)
- Historical performance trends (background context)

### 3.2 Monthly Management Focus

For monthly management review, medium-term dependencies and patterns become more relevant:

```graphviz
digraph G {
    rankdir=LR;
    node [shape=box];
    
    // Monthly pattern nodes
    volume [label="Monthly\nVolume\n(458 declarations)", color=blue, penwidth=2];
    approvals [label="Approval\nRate\n(92%)", color=blue, penwidth=2];
    durations [label="Avg Duration\n(3.65 days)", color=blue, penwidth=2];
    cost [label="Processing\nCost\n(€4,580)", color=blue, penwidth=2];
    
    // Pattern dependencies
    volume -> approvals [color=blue, penwidth=2];
    volume -> durations [color=blue, penwidth=2];
    approvals -> cost [color=blue, penwidth=2];
    durations -> cost [color=blue, penwidth=2];
    
    // Monthly constraints
    budget_cycle [label="Budget\nCycle", shape=diamond, color=red, penwidth=2];
    staffing [label="Staff\nAvailability", shape=diamond, color=red, penwidth=2];
    
    budget_cycle -> volume [color=red, penwidth=2];
    staffing -> durations [color=red, penwidth=2];
    
    // Focus explanation
    label = "Monthly Focus: Volume patterns, performance metrics, and resource utilization";
    labelloc = "t";
}
```

**Key Dependencies for Monthly Management:**
- Declaration volume → Approval rates (workload impact)
- Approval patterns → Process duration (efficiency measure)
- Process duration → Processing cost (resource efficiency)
- Monthly cycles → Declaration volume (predictable patterns)

**Less Important Dependencies:**
- Individual case details (too granular)
- Day-to-day variations (noise rather than signal)

### 3.3 Strategic Improvement Focus

For strategic process improvement, long-term structural dependencies gain importance:

```graphviz
digraph G {
    rankdir=LR;
    node [shape=box];
    
    // Strategic elements
    process_design [label="Process\nDesign", color=blue, penwidth=2];
    approval_rules [label="Approval\nRules", color=blue, penwidth=2];
    automation [label="Automation\nLevel", color=blue, penwidth=2];
    compliance [label="Compliance\nRequirements", color=blue, penwidth=2];
    performance [label="Overall\nPerformance", color=blue, penwidth=2];
    
    // Strategic dependencies
    process_design -> performance [color=blue, penwidth=2];
    approval_rules -> performance [color=blue, penwidth=2];
    automation -> performance [color=blue, penwidth=2];
    compliance -> approval_rules [color=red, penwidth=2];
    compliance -> automation [color=red, penwidth=2];
    
    // Focus explanation
    label = "Strategic Focus: Process design, compliance requirements, and automation potential";
    labelloc = "t";
}
```

**Key Dependencies for Strategic Improvement:**
- Process design → Overall performance (fundamental structure)
- Approval rules → Process efficiency (policy constraints)
- Automation level → Resource requirements (technology leverage)
- Compliance requirements → Process constraints (regulatory boundaries)

**Less Important Dependencies:**
- Individual resource allocation (tactical detail)
- Day-to-day performance variations (short-term noise)

## 4. Process State Perspectives

The process state dramatically changes which dependencies are most critical.

### 4.1 Normal Operation State

During normal operations, standard flow dependencies dominate:

```graphviz
digraph G {
    rankdir=LR;
    node [shape=box];
    
    // Normal process flow (highlighted)
    submit [label="Submission", color=blue, penwidth=2];
    admin [label="Admin Review", color=blue, penwidth=2];
    approve [label="Approval", color=blue, penwidth=2];
    payment [label="Payment", color=blue, penwidth=2];
    
    // Normal dependencies (highlighted)
    submit -> admin [color=blue, penwidth=2];
    admin -> approve [color=blue, penwidth=2];
    approve -> payment [color=blue, penwidth=2];
    
    // Exception handling (faded)
    admin_reject [label="Admin Rejection", color=lightgray];
    super_reject [label="Supervisor Rejection", color=lightgray];
    emp_reject [label="Employee Rejection", color=lightgray];
    
    admin -> admin_reject [color=lightgray];
    approve -> super_reject [color=lightgray];
    admin_reject -> emp_reject [color=lightgray];
    super_reject -> emp_reject [color=lightgray];
    emp_reject -> submit [color=lightgray];
    
    // Focus explanation
    label = "Normal Operations: Standard approval flow and expected transitions";
    labelloc = "t";
}
```

**Key Dependencies in Normal Operations:**
- Submission → Admin Review (standard initiation)
- Admin Review → Approval (expected progression)
- Approval → Payment (successful completion)
- Sequential transitions (orderly flow)

**Less Important Dependencies:**
- Exception handling paths (rare occurrences)
- Resubmission loops (unusual cases)

### 4.2 Exception Handling State

When dealing with exceptions, rejection and rework dependencies become critical:

```graphviz
digraph G {
    rankdir=LR;
    node [shape=box];
    
    // Normal process flow (faded)
    submit [label="Submission"];
    admin [label="Admin Review"];
    approve [label="Approval"];
    payment [label="Payment"];
    
    // Normal dependencies (faded)
    submit -> admin [color=lightgray];
    admin -> approve [color=lightgray];
    approve -> payment [color=lightgray];
    
    // Exception handling (highlighted)
    admin_reject [label="Admin Rejection", color=red, penwidth=2];
    super_reject [label="Supervisor Rejection", color=red, penwidth=2];
    emp_reject [label="Employee Rejection", color=red, penwidth=2];
    resubmit [label="Resubmission", color=red, penwidth=2];
    
    admin -> admin_reject [color=red, penwidth=2];
    approve -> super_reject [color=red, penwidth=2];
    admin_reject -> emp_reject [color=red, penwidth=2];
    super_reject -> emp_reject [color=red, penwidth=2];
    emp_reject -> resubmit [color=red, penwidth=2];
    resubmit -> submit [color=red, penwidth=2];
    
    // Rejection causes (highlighted)
    documentation [label="Missing\nDocumentation", shape=diamond, color=red, penwidth=2];
    policy [label="Policy\nViolation", shape=diamond, color=red, penwidth=2];
    
    documentation -> admin_reject [color=red, penwidth=2];
    policy -> super_reject [color=red, penwidth=2];
    
    // Focus explanation
    label = "Exception Handling: Rejection paths, causes, and resubmission requirements";
    labelloc = "t";
}
```

**Key Dependencies in Exception Handling:**
- Administrative Review → Rejection (problem identification)
- Rejection → Rejection Reason (correction guidance)
- Rejection → Employee Notification (communication)
- Resubmission → Submission (rework cycle)

**Less Important Dependencies:**
- Standard approval flow (temporarily irrelevant)
- Payment processing (not yet applicable)

### 4.3 Peak Load State

During peak load periods, resource constraints and bottlenecks become the critical dependencies:

```graphviz
digraph G {
    rankdir=LR;
    node [shape=box];
    
    // High-volume activities
    submit [label="Submission\n(63 pending)"];
    admin [label="Admin Review\n(41 pending)", color=red, penwidth=2];
    budget [label="Budget Approval\n(18 pending)", color=red, penwidth=2];
    super [label="Supervisor Approval\n(27 pending)", color=red, penwidth=2];
    payment [label="Payment\n(32 pending)"];
    
    // Process flow
    submit -> admin;
    admin -> budget;
    admin -> super;
    budget -> super;
    super -> payment;
    
    // Resource constraints (highlighted)
    admin_res [label="Admin Staff\n(3/5 available)", shape=diamond, color=red, penwidth=2];
    budget_res [label="Budget Owners\n(1/3 available)", shape=diamond, color=red, penwidth=2];
    super_res [label="Supervisors\n(4/9 available)", shape=diamond, color=red, penwidth=2];
    
    admin_res -> admin [color=red, penwidth=2, label="bottleneck"];
    budget_res -> budget [color=red, penwidth=2, label="bottleneck"];
    super_res -> super [color=red, penwidth=2, label="bottleneck"];
    
    // Backlog growth
    backlog [label="Growing\nBacklog", color=red, penwidth=2];
    
    admin -> backlog [color=red, penwidth=2];
    budget -> backlog [color=red, penwidth=2];
    super -> backlog [color=red, penwidth=2];
    
    // Focus explanation
    label = "Peak Load: Resource bottlenecks, backlog growth, and capacity constraints";
    labelloc = "t";
}
```

**Key Dependencies During Peak Load:**
- Resource availability → Activity completion (throughput constraint)
- Process bottlenecks → Backlog growth (queue management)
- Workload balancing → Process flow (resource optimization)
- Prioritization rules → Processing sequence (backlog management)

**Less Important Dependencies:**
- Detailed process rules (overwhelmed by volume)
- Exception handling details (secondary to throughput)

## 5. Stakeholder Perspective Dependencies

Different stakeholders focus on different aspects of process dependencies.

### 5.1 Process Participant View

Process participants (employees, approvers) focus on activity-level dependencies:

```graphviz
digraph G {
    rankdir=LR;
    node [shape=box];
    
    // Activity-level focus
    my_task [label="My Current\nTask", color=blue, penwidth=2];
    next_tasks [label="Next\nTasks", color=blue, penwidth=2];
    my_inputs [label="Required\nInputs", color=blue, penwidth=2];
    my_outputs [label="Expected\nOutputs", color=blue, penwidth=2];
    
    // Immediate dependencies
    my_inputs -> my_task [color=blue, penwidth=2];
    my_task -> my_outputs [color=blue, penwidth=2];
    my_outputs -> next_tasks [color=blue, penwidth=2];
    
    // Task constraints
    deadline [label="Due\nDate", shape=diamond, color=red, penwidth=2];
    instructions [label="Task\nInstructions", shape=diamond, color=red, penwidth=2];
    
    deadline -> my_task [color=red, penwidth=2];
    instructions -> my_task [color=red, penwidth=2];
    
    // Focus explanation
    label = "Participant View: Task inputs, instructions, and immediate next steps";
    labelloc = "t";
}
```

**Key Dependencies for Participants:**
- Required inputs → My task (what I need)
- My task → Expected outputs (what I produce)
- Task instructions → Correct execution (how to do it)
- Due date → Task priority (when it's needed)

**Less Important Dependencies:**
- Overall process structure (bigger picture)
- Upstream process history (prior activities)
- Downstream consequences (future handling)

### 5.2 Process Owner View

Process owners focus on end-to-end flow and performance dependencies:

```graphviz
digraph G {
    rankdir=LR;
    node [shape=box];
    
    // End-to-end process
    start [label="Process\nStart", color=blue, penwidth=2];
    middle [label="Critical\nActivities", color=blue, penwidth=2];
    end [label="Process\nEnd", color=blue, penwidth=2];
    
    // Performance metrics
    volume [label="Process\nVolume", color=blue, penwidth=2];
    duration [label="Process\nDuration", color=blue, penwidth=2];
    quality [label="Output\nQuality", color=blue, penwidth=2];
    cost [label="Process\nCost", color=blue, penwidth=2];
    
    // Process-level dependencies
    start -> middle [color=blue, penwidth=2];
    middle -> end [color=blue, penwidth=2];
    
    volume -> duration [color=blue, penwidth=2];
    duration -> cost [color=blue, penwidth=2];
    middle -> quality [color=blue, penwidth=2];
    quality -> cost [color=blue, penwidth=2];
    
    // Focus explanation
    label = "Process Owner View: End-to-end flow, performance metrics, and outcome quality";
    labelloc = "t";
}
```

**Key Dependencies for Process Owners:**
- Process start → Process end (complete execution)
- Critical activities → Process outputs (value delivery)
- Process volume → Resource requirements (capacity planning)
- Process duration → Process cost (efficiency)
- Output quality → Process success (effectiveness)

**Less Important Dependencies:**
- Individual task details (too granular)
- Specific resource assignments (delegation detail)

### 5.3 Auditor View

Auditors focus on compliance and control dependencies:

```graphviz
digraph G {
    rankdir=LR;
    node [shape=box];
    
    // Control points
    submit_controls [label="Submission\nControls", color=blue, penwidth=2];
    approval_controls [label="Approval\nControls", color=blue, penwidth=2];
    payment_controls [label="Payment\nControls", color=blue, penwidth=2];
    
    // Compliance elements
    policy [label="Policy\nRequirements", color=blue, penwidth=2];
    segregation [label="Duty\nSegregation", color=blue, penwidth=2];
    documentation [label="Evidence\nDocumentation", color=blue, penwidth=2];
    limits [label="Authority\nLimits", color=blue, penwidth=2];
    
    // Control dependencies
    policy -> submit_controls [color=blue, penwidth=2];
    policy -> approval_controls [color=blue, penwidth=2];
    segregation -> approval_controls [color=blue, penwidth=2];
    limits -> approval_controls [color=blue, penwidth=2];
    documentation -> submit_controls [color=blue, penwidth=2];
    documentation -> approval_controls [color=blue, penwidth=2];
    documentation -> payment_controls [color=blue, penwidth=2];
    
    // Risk connections
    fraud_risk [label="Fraud\nRisk", shape=diamond, color=red, penwidth=2];
    error_risk [label="Error\nRisk", shape=diamond, color=red, penwidth=2];
    
    fraud_risk -> segregation [color=red, penwidth=2];
    fraud_risk -> limits [color=red, penwidth=2];
    error_risk -> documentation [color=red, penwidth=2];
    
    // Focus explanation
    label = "Auditor View: Control points, policy compliance, and risk mitigation";
    labelloc = "t";
}
```

**Key Dependencies for Auditors:**
- Policy requirements → Control implementation (compliance)
- Role segregation → Approval controls (fraud prevention)
- Authority limits → Approval levels (authorization control)
- Evidence documentation → Process verification (audit trail)
- Control effectiveness → Risk mitigation (risk management)

**Less Important Dependencies:**
- Process efficiency (secondary to control)
- Resource optimization (outside audit scope)

## 6. Specific BPI2020 Declaration Process Examples

### 6.1 Amount-Based Approval Routing

The declaration amount creates critical dependencies for routing approvals:

```graphviz
digraph G {
    node [shape=box];
    
    // Declaration amount thresholds
    amount [label="Declaration\nAmount", shape=diamond, color=blue, penwidth=2];
    
    // Approval paths
    low_path [label="Standard\nApproval Path\n(<€100)", color=green];
    med_path [label="Extended\nApproval Path\n(€100-€500)", color=orange];
    high_path [label="Complex\nApproval Path\n(>€500)", color=red];
    
    // Amount-based dependencies
    amount -> low_path [label="<€100", color=green, penwidth=2];
    amount -> med_path [label="€100-€500", color=orange, penwidth=2];
    amount -> high_path [label=">€500", color=red, penwidth=2];
    
    // Path activities
    subgraph cluster_low {
        label = "Standard Path";
        color = green;
        low_admin [label="Admin\nApproval"];
        low_super [label="Supervisor\nApproval"];
        low_admin -> low_super;
    }
    
    subgraph cluster_med {
        label = "Extended Path";
        color = orange;
        med_admin [label="Admin\nApproval"];
        med_super [label="Supervisor\nApproval"];
        med_add [label="Additional\nVerification"];
        med_admin -> med_add -> med_super;
    }
    
    subgraph cluster_high {
        label = "Complex Path";
        color = red;
        high_admin [label="Admin\nApproval"];
        high_budget [label="Budget Owner\nApproval"];
        high_super [label="Supervisor\nApproval"];
        high_admin -> high_budget -> high_super;
    }
    
    // Path dependencies
    low_path -> low_admin;
    med_path -> med_admin;
    high_path -> high_admin;
}
```

In this example, declaration amount creates critical routing dependencies that determine the entire approval path. This dependency has highest importance during the administrative review stage, but becomes less important once the declaration is routed to the appropriate path.

### 6.2 Budget Owner Availability Impact

The availability of budget owners creates a critical dependency during peak periods:

```graphviz
digraph G {
    rankdir=TB;
    node [shape=box];
    
    // Budget owner availability states
    available [label="Budget Owner\nAvailable", color=green];
    unavailable [label="Budget Owner\nUnavailable", color=red, penwidth=2];
    
    // Process impacts
    normal_flow [label="Normal\nApproval Flow\n(~3 days)"];
    delayed_flow [label="Delayed\nApproval Flow\n(~7+ days)", color=red, penwidth=2];
    
    // Alternative paths
    delegation [label="Delegation to\nAlternate Approver", color=orange];
    escalation [label="Escalation to\nSenior Management", color=orange];
    
    // Dependencies
    available -> normal_flow [color=green];
    unavailable -> delayed_flow [color=red, penwidth=2];
    
    // Contingency dependencies
    unavailable -> delegation [style=dashed, color=orange];
    unavailable -> escalation [style=dashed, color=orange];
    delegation -> normal_flow [style=dashed, color=orange];
    escalation -> normal_flow [style=dashed, color=orange];
    
    // Time impact
    time_impact [label="+ 4.2 days\naverage delay", color=red, penwidth=2];
    unavailable -> time_impact [color=red, penwidth=2];
    time_impact -> delayed_flow [color=red, penwidth=2];
}
```

This dependency becomes critical during vacation periods or times when budget owners are unavailable, causing significant delays. However, during normal operations when budget owners are available, this dependency has minimal impact.

### 6.3 Rejection-Resubmission Cycle

The rejection-resubmission cycle creates a feedback loop dependency that significantly impacts process duration:

```graphviz
digraph G {
    rankdir=LR;
    node [shape=box];
    
    // Main process activities
    submit [label="Submit\nDeclaration"];
    review [label="Review\nDeclaration"];
    approve [label="Approve\nDeclaration"];
    payment [label="Process\nPayment"];
    
    // Rejection cycle (highlighted)
    reject [label="Reject\nDeclaration", color=red, penwidth=2];
    notify [label="Notify\nEmployee", color=red, penwidth=2];
    resubmit [label="Resubmit\nDeclaration", color=red, penwidth=2];
    
    // Normal flow
    submit -> review;
    review -> approve;
    approve -> payment;
    
    // Rejection cycle dependencies
    review -> reject [color=red, penwidth=2];
    approve -> reject [color=red, penwidth=2];
    reject -> notify [color=red, penwidth=2];
    notify -> resubmit [color=red, penwidth=2];
    resubmit -> review [color=red, penwidth=2];
    
    // Cycle impact
    cycle_count [label="Number of\nRejection Cycles", shape=diamond, color=red, penwidth=2];
    duration_impact [label="Process Duration\nImpact", shape=diamond, color=red, penwidth=2];
    
    cycle_count -> duration_impact [label="+3.8 days\nper cycle", color=red, penwidth=2];
    reject -> cycle_count [color=red, penwidth=2];
}
```

This dependency becomes dominant for cases experiencing rejection, adding significant duration to the process. For cases that proceed without rejection, this entire dependency structure can be ignored.

### 6.4 Payment Schedule Constraints

Payment processing depends critically on payment scheduling constraints:

```graphviz
digraph G {
    rankdir=TB;
    node [shape=box];
    
    // Payment scheduling constraints
    payment_day [label="Payment\nProcessing Day\n(Tue, Thu)", shape=diamond, color=blue, penwidth=2];
    payment_time [label="Payment\nCutoff Time\n(3:00 PM)", shape=diamond, color=blue, penwidth=2];
    batch_size [label="Batch\nSize Limit\n(500 payments)", shape=diamond, color=blue, penwidth=2];
    
    // Payment states
    current_batch [label="Current Batch\nProcessing"];
    next_batch [label="Next Batch\nProcessing"];
    expedited [label="Expedited\nProcessing"];
    
    // Timing impacts
    fast_path [label="Same-day\nProcessing\n(0 day wait)", color=green];
    normal_path [label="Next-cycle\nProcessing\n(2-3 day wait)", color=orange];
    slow_path [label="Delayed\nProcessing\n(5+ day wait)", color=red];
    
    // Dependencies
    payment_day -> current_batch [label="Is payment day", color=green];
    payment_day -> next_batch [label="Not payment day", color=red];
    
    payment_time -> current_batch [label="Before cutoff", color=green];
    payment_time -> next_batch [label="After cutoff", color=red];
    
    batch_size -> current_batch [label="Capacity available", color=green];
    batch_size -> next_batch [label="Batch full", color=red];
    
    current_batch -> fast_path [color=green];
    next_batch -> normal_path [color=orange];
    next_batch -> slow_path [color=red];
    
    // Priority override
    priority [label="High\nPriority", shape=diamond];
    priority -> expedited [style=dashed, color=blue];
    expedited -> fast_path [style=dashed, color=blue];
}
```

This dependency becomes critical at payment processing time but is irrelevant during earlier approval stages. The importance also varies by day of the week, with heightened importance near payment processing days and cutoff times.

## 7. Application Areas for Selective Dependency Focus

Understanding which dependencies matter most in different contexts has several practical applications:

### 7.1 Process Monitoring Dashboards

Different dashboards can be created for different stakeholders, focusing on the dependencies most relevant to them:

1. **Employee Dashboard**: Submission status, rejection reasons, payment timing
2. **Manager Dashboard**: Pending approvals, budget impacts, approval timing
3. **Finance Dashboard**: Payment batches, processing schedule, reconciliation status
4. **Executive Dashboard**: Process volume, duration trends, cost metrics

### 7.2 Process Improvement Initiatives

Improvement efforts can target different dependencies based on organizational priorities:

1. **Efficiency Focus**: Target sequential dependencies and approval paths
2. **Compliance Focus**: Enhance control dependencies and authorization rules
3. **Cost Reduction Focus**: Optimize resource dependencies and automation
4. **User Experience Focus**: Improve information dependencies and communication

### 7.3 Resource Allocation Decisions

Resource allocation can prioritize the most critical dependencies based on current needs:

1. **Normal Operations**: Balance resources across standard activities
2. **Peak Periods**: Allocate additional resources to bottleneck activities
3. **Exception Handling**: Dedicate resources to rejection handling and resubmission
4. **Process Changes**: Focus resources on training and transition management

### 7.4 Risk Management Strategies

Risk mitigation can focus on dependencies with the highest potential impact:

1. **Operational Risks**: Monitor resource dependencies and bottlenecks
2. **Compliance Risks**: Strengthen authorization dependencies and controls
3. **Financial Risks**: Enhance budget verification and payment processing dependencies
4. **Reputational Risks**: Improve communication dependencies and transparency

## Conclusion

The BPI2020 Domestic Declarations process clearly demonstrates how task dependencies vary in importance across different contexts, stakeholders, time horizons, and process states. By applying selective focus to the most relevant dependencies in each situation, organizations can more effectively manage, monitor, and improve their processes.

Key insights from this analysis include:

1. **Contextual Relevance**: Dependencies that are critical in one context may be peripheral in another
2. **Stakeholder Perspectives**: Different roles naturally focus on different aspects of the process
3. **Temporal Variation**: The importance of dependencies shifts over time and at different process stages
4. **State-Based Priorities**: Normal operations, exceptions, and peak loads each have different critical dependencies
5. **Selective Focus Benefits**: Focusing on the right dependencies for each situation improves decision-making and process management

By understanding and applying selective focus to task dependencies, organizations can develop more nuanced, effective approaches to process management and improvement, ultimately leading to better outcomes for all stakeholders involved in the declaration process.