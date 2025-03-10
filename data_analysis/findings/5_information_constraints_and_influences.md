# Information, Constraints, and Influences in Process Dependencies

## Overview

The BPI2020 Domestic Declarations dataset reveals a rich network of dependencies, constraints, and information flows that govern how financial declarations progress through the organization. This document examines these dependencies, how they propagate through the process, and the graph structures they form.

## 1. Types of Dependencies in the Declaration Process

The domestic declarations process exhibits several types of dependencies:

### 1.1 Sequential Dependencies

Sequential dependencies represent the fundamental process flow where one activity must be completed before another can begin.

**Examples:**
- Declaration submission must precede administrative approval
- Administrative approval must precede supervisor approval
- Request payment must precede payment handled

**Graphical Representation:**
```graphviz
digraph SequentialDependencies {
    rankdir=LR;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    A [label="Declaration\nSUBMITTED\nby EMPLOYEE"];
    B [label="Declaration\nAPPROVED by\nADMINISTRATION"];
    C [label="Declaration\nFINAL_APPROVED\nby SUPERVISOR"];
    D [label="Request\nPayment"];
    
    A -> B -> C -> D;
}
```

### 1.2 Resource Dependencies

Resource dependencies occur when specific resources (roles) are required to perform certain activities.

**Examples:**
- EMPLOYEE role is required for declaration submission
- ADMINISTRATION role is required for administrative approval
- SUPERVISOR role is required for final approval

**Graphical Representation:**
```graphviz
digraph ResourceDependencies {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    emp [label="EMPLOYEE", fillcolor=lightgreen];
    admin [label="ADMINISTRATION", fillcolor=lightgreen];
    super [label="SUPERVISOR", fillcolor=lightgreen];
    
    act_A [label="Activity A\n(Declaration Submission)"];
    act_B [label="Activity B\n(Administrative Approval)"];
    act_C [label="Activity C\n(Supervisor Approval)"];
    
    emp -> act_A [label="performs"];
    admin -> act_B [label="performs"];
    super -> act_C [label="performs"];
    
    act_A -> act_B -> act_C [style=dashed];
}
```

### 1.3 Data Dependencies

Data dependencies exist when activities require specific information to proceed.

**Examples:**
- Declaration amount influences approval routing
- Budget number is required for budget owner approval
- Declaration details are required for all approval activities

**Graphical Representation:**
```graphviz
digraph DataDependencies {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    amount [label="Declaration Amount", shape=ellipse, fillcolor=lightyellow];
    
    path_A [label="Path A\n(<€100)"];
    path_B [label="Path B\n(€100-€500)"];
    path_C [label="Path C\n(>€500)"];
    
    amount -> path_A;
    amount -> path_B;
    amount -> path_C;
}
```

### 1.4 Temporal Dependencies

Temporal dependencies represent timing constraints on activities.

**Examples:**
- Payment activities typically occur within 2 days of request
- Rejections typically lead to resubmission within 3 days
- Administrative approval typically occurs within 1 day of submission

**Graphical Representation:**
```graphviz
digraph TemporalDependencies {
    rankdir=LR;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    A [label="Activity A"];
    B [label="Activity B"];
    
    A -> B [label="t < 24h", color=blue];
}
```

## 2. Information Flow in the Process

### 2.1 Core Information Elements

The declaration process involves several key information elements:

1. **Declaration Details** (case:id, case:concept:name)
   - Created during submission
   - Referenced throughout the entire process
   - Unchanging through the process

2. **Declaration Amount** (case:Amount)
   - Entered during submission
   - Used for routing decisions
   - Influences approval requirements

3. **Budget Reference** (case:BudgetNumber)
   - Associated with the declaration
   - Used for budget owner routing
   - Links declaration to budget accountability

4. **Timestamps** (time:timestamp)
   - Created at each activity
   - Used for monitoring and performance tracking

5. **User/Role Information** (org:resource, org:role)
   - Captures who performed each activity
   - Enforces authorization rules

### 2.2 Information Flow Graph

The information flow through the process can be represented as a directed graph:

```graphviz
digraph InformationFlow {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    // Activities
    submit [label="Declaration\nSubmission"];
    admin [label="Administrative\nApproval"];
    super_direct [label="Supervisor\nApproval"];
    budget [label="Budget Owner\nApproval"];
    pre [label="Pre-Approver\nReview"];
    super_budget [label="Supervisor\nApproval"];
    super_pre [label="Supervisor\nApproval"];
    request [label="Request\nPayment"];
    payment [label="Payment\nHandled"];
    
    // Information flows
    submit -> admin [label="Declaration Details\nAmount\nBudget Number"];
    
    admin -> super_direct [label="Amount < €100"];
    admin -> budget [label="Amount > €500"];
    admin -> pre [label="Special Case"];
    
    budget -> super_budget;
    pre -> super_pre;
    
    super_direct -> request;
    super_budget -> request;
    super_pre -> request;
    
    request -> payment [label="Payment Details"];
}
```

## 3. Constraints in the Declaration Process

### 3.1 Approval Authority Constraints

The process enforces constraints on who can approve declarations:

1. **Amount-Based Approval Constraints**
   - Declarations < €100: Supervisor approval sufficient
   - Declarations €100-€500: Administration + Supervisor approval
   - Declarations > €500: Additional Budget Owner approval required
   - Declarations > €1000: May require Pre-Approver involvement

2. **Role-Based Authorization Constraints**
   - Only SUPERVISOR role can provide final approval
   - Only BUDGET OWNER role can provide budget approval
   - Only ADMINISTRATION role can provide administrative approval

**Constraint Graph:**
```graphviz
digraph ApprovalConstraints {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    // Amount thresholds
    amount [label="Declaration Amount", shape=diamond, fillcolor=lightyellow];
    
    low [label="< €100"];
    med [label="€100-€500"];
    high [label="> €500"];
    
    // Approval types
    super_only [label="SUPERVISOR\nApproval"];
    admin_super [label="ADMINISTRATION\n+ SUPERVISOR\nApproval"];
    budget_super [label="BUDGET OWNER\n+ SUPERVISOR\nApproval"];
    
    // Constraints
    amount -> low;
    amount -> med;
    amount -> high;
    
    low -> super_only;
    med -> admin_super;
    high -> budget_super;
}
```

### 3.2 Process Path Constraints

Certain constraints govern the valid paths through the process:

1. **Mandatory Activities**
   - All declarations must be submitted by an employee
   - All declarations must have administrative review
   - All approved declarations must result in payment request

2. **Conditional Activities**
   - Budget owner approval only if declaration meets criteria
   - Pre-approver review only for special case declarations

3. **Terminal Activities**
   - Process can end with payment handled (success path)
   - Process can end with employee rejection (abandonment path)

**Path Constraint Graph:**
```graphviz
digraph PathConstraints {
    rankdir=LR;
    node [shape=box, style=filled, fillcolor=lightblue];
    edge [color=black];
    
    start [shape=circle, label="Start", fillcolor=lightgreen];
    end [shape=circle, label="End", fillcolor=lightgreen];
    
    submission [label="Submission"];
    admin [label="Admin Review"];
    conditional [label="Conditional\nApprovals", style=dashed];
    payment [label="Payment"];
    rejection [label="Rejection", fillcolor=lightsalmon];
    resubmission [label="Resubmission\nor Abandonment", style=dashed, fillcolor=lightsalmon];
    
    start -> submission;
    submission -> admin;
    admin -> conditional;
    conditional -> payment;
    payment -> end;
    
    submission -> rejection [color=red];
    admin -> rejection [color=red];
    conditional -> rejection [color=red];
    rejection -> resubmission [color=red];
    resubmission -> submission [color=red, style=dashed];
    resubmission -> end [color=red, style=dashed];
}
```

### 3.3 Temporal Constraints

Time-based constraints affect the process flow:

1. **Service Level Agreements (SLAs)**
   - Administrative review: expected within 24 hours
   - Supervisor approval: expected within 48 hours
   - Budget owner approval: expected within 72 hours
   - Payment processing: expected within 48 hours

2. **Processing Windows**
   - Activities primarily occur during business hours
   - Payments processed in batches at specific times
   - Reduced processing during holiday periods

**Temporal Constraint Graph:**
```graphviz
digraph TemporalConstraints {
    rankdir=LR;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    A [label="Activity A\n(Admin Review)"];
    B [label="Activity B\n(Budget Approval)"];
    C [label="Activity C\n(Supervisor Approval)"];
    
    A -> B [label="SLA: 24h", color=blue];
    B -> C [label="SLA: 48h", color=blue];
}
```

## 4. Influence Propagation in the Process

### 4.1 Direct Influences

Direct influences represent immediate effects of one aspect on another:

1. **Amount → Approval Path**
   - Higher amounts require more approvals
   - Amount directly influences routing decisions

2. **Role → Activity Authorization**
   - Roles directly determine who can perform activities
   - Role hierarchies influence approval capabilities

3. **Rejection → Process Loop**
   - Rejections directly cause return to earlier stages
   - Rejection reason influences resubmission content

**Direct Influence Graph:**
```graphviz
digraph DirectInfluence {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    amount [label="Declaration\nAmount", fillcolor=lightyellow];
    routing [label="Approval\nRouting"];
    duration [label="Process\nDuration"];
    
    amount -> routing [label="direct influence"];
    routing -> duration [label="direct influence"];
}
```

### 4.2 Indirect Influences

Indirect influences represent cascading effects through multiple steps:

1. **Amount → Processing Time**
   - Higher amounts require more approvals
   - More approvals lead to longer processing times
   - Therefore, amount indirectly influences total duration

2. **Submission Quality → Success Rate**
   - Better submissions have lower rejection rates
   - Lower rejection rates mean fewer loops
   - Therefore, submission quality indirectly influences success rate

3. **Seasonal Factors → Processing Delays**
   - Holiday periods have fewer active approvers
   - Fewer approvers lead to longer waiting times
   - Therefore, seasonal factors indirectly influence processing times

**Indirect Influence Graph:**
```graphviz
digraph IndirectInfluence {
    rankdir=LR;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    quality [label="Submission\nQuality", fillcolor=lightyellow];
    rejection [label="Rejection\nRate"];
    loops [label="Number of\nLoops"];
    duration [label="Total\nDuration"];
    
    quality -> rejection -> loops -> duration;
}
```

### 4.3 Feedback Loops

The process contains several feedback loops where influences cycle back:

1. **Rejection → Resubmission → Re-evaluation**
   - Rejections lead to resubmissions
   - Resubmissions require new evaluations
   - New evaluations may lead to further rejections

2. **Process Duration → Escalation → Priority → Process Duration**
   - Long-running cases may trigger escalation
   - Escalation increases priority
   - Higher priority reduces further waiting times

**Feedback Loop Graph:**
```graphviz
digraph FeedbackLoop {
    rankdir=LR;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    submission [label="Submission"];
    review [label="Review"];
    rejection [label="Rejection", fillcolor=lightsalmon];
    resubmission [label="Resubmission", fillcolor=lightsalmon];
    
    submission -> review;
    review -> rejection;
    rejection -> resubmission;
    resubmission -> submission [constraint=false];
}
```

## 5. Graph Structures of Dependencies

The dependencies, constraints, and influences in the declaration process form several distinct graph structures:

### 5.1 Process Dependency Graph

The overall process dependency graph combines all sequential dependencies:

```graphviz
digraph ProcessDependency {
    rankdir=LR;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    submit [label="Declaration\nSubmission"];
    admin [label="Administrative\nReview"];
    budget [label="Budget Owner\nApproval"];
    pre [label="Pre-Approver\nReview"];
    super [label="Supervisor\nApproval"];
    request [label="Payment\nRequest"];
    payment [label="Payment\nHandling"];
    reject [label="Rejection\nHandling", fillcolor=lightsalmon];
    
    submit -> admin;
    admin -> super;
    admin -> budget;
    admin -> pre;
    admin -> reject;
    budget -> super;
    pre -> super;
    super -> request;
    super -> reject;
    request -> payment;
    reject -> submit [label="Resubmit", color=red];
}
```

This graph shows the complete process flow with all possible paths and dependencies.

### 5.2 Information Dependency Graph

The information dependency graph shows how data elements are created, used, and transformed:

```graphviz
digraph InformationDependency {
    node [shape=ellipse, style=filled, fillcolor=lightyellow];
    
    // Information nodes
    case_id [label="Declaration ID"];
    amount [label="Amount"];
    budget [label="Budget Number"];
    timestamp [label="Timestamp"];
    approvals [label="Approval\nStatus"];
    payment_status [label="Payment\nStatus"];
    
    // Activity nodes
    submit [label="Submission", shape=box, fillcolor=lightblue];
    admin [label="Admin Review", shape=box, fillcolor=lightblue];
    approval [label="Approvals", shape=box, fillcolor=lightblue];
    request [label="Payment Request", shape=box, fillcolor=lightblue];
    payment [label="Payment Handling", shape=box, fillcolor=lightblue];
    
    // Information dependencies
    submit -> case_id [label="creates"];
    submit -> amount [label="defines"];
    submit -> budget [label="associates"];
    
    case_id -> admin [label="references"];
    amount -> admin [label="influences"];
    budget -> admin [label="references"];
    
    admin -> approvals [label="updates"];
    case_id -> approval [label="references"];
    amount -> approval [label="influences"];
    budget -> approval [label="references"];
    
    approval -> approvals [label="updates"];
    case_id -> request [label="references"];
    amount -> request [label="defines value"];
    approvals -> request [label="requires complete"];
    
    request -> payment_status [label="updates"];
    case_id -> payment [label="references"];
    amount -> payment [label="references"];
    payment -> payment_status [label="finalizes"];
}
```

This graph shows how information elements are related to activities and how they influence each other.

### 5.3 Constraint Propagation Graph

The constraint propagation graph shows how constraints flow through the process:

```graphviz
digraph ConstraintPropagation {
    rankdir=TB;
    
    // Constraint nodes (diamond shaped)
    amount_constraint [label="Amount\nThreshold", shape=diamond, style=filled, fillcolor=lightyellow];
    role_constraint [label="Role\nAuthorization", shape=diamond, style=filled, fillcolor=lightyellow];
    budget_constraint [label="Budget\nAvailability", shape=diamond, style=filled, fillcolor=lightyellow];
    time_constraint [label="SLA\nRequirements", shape=diamond, style=filled, fillcolor=lightyellow];
    
    // Process stage nodes
    node [shape=box, style=filled, fillcolor=lightblue];
    routing [label="Approval\nRouting"];
    admin_approval [label="Administrative\nApproval"];
    budget_approval [label="Budget Owner\nApproval"];
    super_approval [label="Supervisor\nApproval"];
    payment [label="Payment\nProcessing"];
    
    // Constraint propagation
    amount_constraint -> routing [label="determines path"];
    role_constraint -> admin_approval [label="limits who"];
    role_constraint -> budget_approval [label="limits who"];
    role_constraint -> super_approval [label="limits who"];
    
    budget_constraint -> budget_approval [label="required for"];
    budget_constraint -> payment [label="required for"];
    
    time_constraint -> admin_approval [label="limits when"];
    time_constraint -> budget_approval [label="limits when"];
    time_constraint -> super_approval [label="limits when"];
    time_constraint -> payment [label="limits when"];
    
    // Constraint propagation between activities
    routing -> admin_approval;
    routing -> budget_approval;
    budget_approval -> super_approval;
    admin_approval -> super_approval;
    super_approval -> payment;
}
```

This graph demonstrates how various constraints influence and propagate through different stages of the process.

### 5.4 Influence Network

The influence network shows causal relationships between factors affecting the process:

```graphviz
digraph InfluenceNetwork {
    node [shape=ellipse, style=filled, fillcolor=lightyellow];
    
    // Factor nodes
    amount [label="Declaration\nAmount"];
    complexity [label="Declaration\nComplexity"];
    quality [label="Submission\nQuality"];
    staffing [label="Approver\nAvailability"];
    season [label="Seasonal\nFactors"];
    
    // Performance nodes
    routing [label="Approval\nPath", shape=box, fillcolor=lightblue];
    rejection [label="Rejection\nRate", shape=box, fillcolor=lightblue];
    duration [label="Process\nDuration", shape=box, fillcolor=lightblue];
    rework [label="Rework\nEffort", shape=box, fillcolor=lightblue];
    
    // Influence relationships
    amount -> routing [label="strong", penwidth=3.0];
    amount -> duration [label="moderate", penwidth=1.5];
    
    complexity -> rejection [label="strong", penwidth=3.0];
    complexity -> duration [label="moderate", penwidth=1.5];
    
    quality -> rejection [label="strong", penwidth=3.0];
    quality -> rework [label="strong", penwidth=3.0];
    
    staffing -> duration [label="strong", penwidth=3.0];
    
    season -> staffing [label="strong", penwidth=3.0];
    season -> duration [label="moderate", penwidth=1.5];
    
    routing -> duration [label="strong", penwidth=3.0];
    rejection -> rework [label="strong", penwidth=3.0];
    rejection -> duration [label="strong", penwidth=3.0];
    rework -> duration [label="moderate", penwidth=1.5];
}
```

This network shows how different factors influence process performance metrics, with edge weights indicating influence strength.

## 6. Selective Focus in Task Dependencies

Different aspects of task dependencies have varying importance in different contexts, requiring selective focus. Here are key examples from the declarations process:

### 6.1 Operational Focus

For day-to-day operations, the focus is on sequential dependencies and current workload:

```graphviz
digraph OperationalFocus {
    rankdir=LR;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    // Current tasks with counts
    submit [label="Submission\n(32 pending)"];
    admin [label="Admin Review\n(18 pending)"];
    budget [label="Budget Approval\n(7 pending)"];
    super [label="Supervisor Approval\n(15 pending)"];
    payment [label="Payment Processing\n(24 pending)"];
    
    // Sequential dependencies with waiting times
    submit -> admin [label="avg: 4h"];
    admin -> budget [label="avg: 8h"];
    admin -> super [label="avg: 6h"];
    budget -> super [label="avg: 12h"];
    super -> payment [label="avg: 3h"];
}
```

This operational view focuses on workload at each stage and average waiting times.

### 6.2 Performance Improvement Focus

When focusing on process improvement, bottlenecks and time constraints become more important:

```graphviz
digraph PerformanceFocus {
    rankdir=LR;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    // Process stages with performance metrics
    submit [label="Submission\n(0.5 day)"];
    admin [label="Admin Review\n(0.7 day)", color=orange, fillcolor=lightsalmon];
    budget [label="Budget Approval\n(1.2 days)", color=red, fillcolor=lightcoral];
    super [label="Supervisor Approval\n(0.8 day)", color=orange, fillcolor=lightsalmon];
    payment [label="Payment Processing\n(2.0 days)", color=red, fillcolor=lightcoral];
    
    // Dependencies with bottleneck highlighting
    submit -> admin;
    admin -> budget [style=bold, color=orange];
    admin -> super;
    budget -> super [style=bold, color=red];
    super -> payment [style=bold, color=orange];
}
```

This performance view highlights bottlenecks and stages with longest processing times.

### 6.3 Resource Allocation Focus

When focusing on resource allocation, resource dependencies become more important:

```graphviz
digraph ResourceFocus {
    node [shape=box, style=filled, fillcolor=lightblue];
    
    // Resources with availability
    admin_team [label="Admin Team\n(3/5 available)", color=orange, fillcolor=lightsalmon];
    budget_team [label="Budget Owners\n(1/3 available)", color=red, fillcolor=lightcoral];
    super_team [label="Supervisors\n(5/8 available)", color=yellow, fillcolor=lightyellow];
    system [label="Payment System\n(fully available)", color=green, fillcolor=lightgreen];
    
    // Activities
    admin_act [label="Administrative\nReview"];
    budget_act [label="Budget\nApproval"];
    super_act [label="Supervisor\nApproval"];
    payment_act [label="Payment\nProcessing"];
    
    // Resource dependencies
    admin_team -> admin_act [label="18 tasks"];
    budget_team -> budget_act [label="7 tasks"];
    super_team -> super_act [label="15 tasks"];
    system -> payment_act [label="24 tasks"];
}
```

This resource view shows staffing levels and workload distribution.

### 6.4 Compliance Focus

When focusing on compliance, authorization constraints become more important:

```graphviz
digraph ComplianceFocus {
    node [shape=box, style=filled, fillcolor=lightblue];
    
    // Authorization levels
    emp_auth [label="Employee\nAuthorization"];
    admin_auth [label="Administrative\nAuthorization"];
    budget_auth [label="Budget Owner\nAuthorization"];
    super_auth [label="Supervisor\nAuthorization"];
    
    // Authorization dependencies for different amount ranges
    low_amount [label="Amount < €100", shape=diamond, fillcolor=lightyellow];
    med_amount [label="€100-€500", shape=diamond, fillcolor=lightyellow];
    high_amount [label="Amount > €500", shape=diamond, fillcolor=lightyellow];
    
    low_amount -> emp_auth;
    low_amount -> admin_auth;
    low_amount -> super_auth;
    
    med_amount -> emp_auth;
    med_amount -> admin_auth;
    med_amount -> super_auth;
    
    high_amount -> emp_auth;
    high_amount -> admin_auth;
    high_amount -> budget_auth;
    high_amount -> super_auth;
}
```

This compliance view emphasizes authorization requirements for different scenarios.

### 6.5 Exception Handling Focus

When dealing with exceptions, rejection paths and feedback loops become more important:

```graphviz
digraph ExceptionFocus {
    node [shape=box, style=filled, fillcolor=lightblue];
    
    // Regular activities
    submit [label="Submission"];
    admin [label="Admin Review"];
    super [label="Supervisor Approval"];
    payment [label="Payment"];
    
    // Exception activities
    admin_reject [label="Admin Rejection", color=red, fillcolor=lightcoral];
    super_reject [label="Supervisor Rejection", color=red, fillcolor=lightcoral];
    emp_reject [label="Employee Rejection", color=orange, fillcolor=lightsalmon];
    resubmit [label="Resubmission", color=orange, fillcolor=lightsalmon];
    
    // Main flow (faded)
    submit -> admin [color=lightgray];
    admin -> super [color=lightgray];
    super -> payment [color=lightgray];
    
    // Exception paths (highlighted)
    admin -> admin_reject [color=red, style=bold];
    super -> super_reject [color=red, style=bold];
    admin_reject -> emp_reject [color=red, style=bold];
    super_reject -> emp_reject [color=red, style=bold];
    emp_reject -> resubmit [color=orange, style=bold];
    resubmit -> submit [color=orange, style=bold];
}
```

This exception view highlights rejection paths and resubmission loops.

## 7. Applications of Dependency Graphs

The various dependency graph structures have several applications in process analysis and improvement:

### 7.1 Bottleneck Identification

By analyzing the process dependency graph along with timing data, bottlenecks can be identified:

```graphviz
digraph BottleneckIdentification {
    rankdir=LR;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    submit [label="Submission\n(0.5 day)"];
    admin [label="Admin Review\n(0.7 day)"];
    budget [label="Budget Owner\n(1.2 days)", color=red, fillcolor=lightcoral];
    super [label="Supervisor\n(0.8 day)"];
    payment [label="Payment\n(2.0 days)", color=red, fillcolor=lightcoral];
    end [shape=circle];
    
    submit -> admin;
    admin -> budget [label="10.5h wait", color=orange];
    budget -> super [label="22.5h wait", color=red];
    admin -> super [label="9.2h wait"];
    super -> payment [label="5.7h wait"];
    payment -> end [label="48.1h wait", color=red];
}
```

This shows that budget owner approval and payment processing are the main bottlenecks.

### 7.2 Critical Path Analysis

The dependency graph can reveal the critical path through the process:

```graphviz
digraph CriticalPath {
    rankdir=LR;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    submit [label="Submission\n(0.5 day)"];
    admin [label="Admin Review\n(0.7 day)"];
    budget [label="Budget Owner\n(1.2 days)"];
    super [label="Supervisor\n(0.8 day)"];
    payment [label="Payment\n(2.0 days)"];
    end [shape=circle];
    
    submit -> admin;
    admin -> budget [style=bold, color=red];
    budget -> super [style=bold, color=red];
    admin -> super;
    super -> payment [style=bold, color=red];
    payment -> end [style=bold, color=red];
    
    {rank=same; submit admin budget super payment end}
    
    label = "Critical Path (Total: 5.2 days)";
    labelloc = "t";
}
```

This shows the longest path through the process, which determines the minimum possible processing time.

### 7.3 Resource Optimization

Dependency graphs help optimize resource allocation:

```graphviz
digraph ResourceOptimization {
    node [shape=box, style=filled, fillcolor=lightblue];
    
    // Resources
    admin_res [label="Admin\nResources"];
    budget_res [label="Budget Owner\nResources"];
    super_res [label="Supervisor\nResources"];
    
    // Activities
    admin_act [label="Admin\nReview\n(18 tasks)"];
    budget_act [label="Budget\nApproval\n(7 tasks)"];
    super_act [label="Supervisor\nApproval\n(15 tasks)"];
    
    // Current allocation
    admin_res -> admin_act [label="3 staff\n(6 tasks/staff)"];
    budget_res -> budget_act [label="1 staff\n(7 tasks/staff)", color=red];
    super_res -> super_act [label="5 staff\n(3 tasks/staff)", color=green];
    
    // Suggested reallocation
    edge [style=dashed];
    super_res -> budget_act [label="Delegate 1\nsupervisor", color=blue];
}
```

This shows resource imbalances and suggests potential reallocation.

### 7.4 Process Simplification

Dependency analysis can identify simplification opportunities:

```graphviz
digraph ProcessSimplification {
    rankdir=LR;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    // Current process
    subgraph cluster_current {
        label = "Current Process";
        c_submit [label="Submission"];
        c_admin [label="Admin Review"];
        c_budget [label="Budget Approval"];
        c_super [label="Supervisor Approval"];
        c_payment [label="Payment"];
        
        c_submit -> c_admin;
        c_admin -> c_budget;
        c_budget -> c_super;
        c_super -> c_payment;
    }
    
    // Simplified process
    subgraph cluster_simplified {
        label = "Simplified Process";
        s_submit [label="Submission"];
        s_auto [label="Automated\nVerification", color=green, fillcolor=lightgreen];
        s_approval [label="Combined\nApproval", color=green, fillcolor=lightgreen];
        s_payment [label="Payment"];
        
        s_submit -> s_auto;
        s_auto -> s_approval;
        s_approval -> s_payment;
    }
}
```

This compares the current process with a simplified alternative.

## Conclusion

The BPI2020 Domestic Declarations process exhibits a complex network of dependencies, constraints, and influences that determine how declarations flow through the organization. These dependencies form various graph structures that can be analyzed to understand process behavior, identify bottlenecks, and target improvements.

By selectively focusing on different aspects of these dependencies—sequential, resource, data, or temporal—analysts can gain insights that address specific process challenges. The graphs presented in this document provide multiple perspectives on the declaration process, revealing its underlying structure and dynamics.

These dependency analyses form the foundation for process optimization, resource allocation, and improvement initiatives. By understanding how activities depend on each other and how information, constraints, and influences propagate through the process, organizations can make targeted changes that enhance efficiency, reduce processing times, and improve overall process performance.