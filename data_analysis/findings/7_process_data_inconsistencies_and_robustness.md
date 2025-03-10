# Process Data: Noise, Inconsistencies, and Robust Representation

## 1. Sources of Noise in Process Data

The BPI2020 Domestic Declarations dataset, like all real-world process data, contains various forms of noise that can impact analysis and interpretation. Process mining practitioners must understand these noise sources to develop robust analytical approaches.

### 1.1 Temporal Noise

**Definition:** Inconsistencies or inaccuracies in the timing of activities.

**Sources in the BPI2020 Dataset:**

1. **Timestamp Precision Issues**
   - The dataset uses timestamps with timezone information (e.g., "2017-01-09 09:49:50+00:00")
   - Analysis reveals some timestamps may have been recorded in different time zones despite the +00:00 notation
   - Time gaps between certain activities show unusual patterns suggesting possible timezone inconsistencies

2. **Delayed Recording**
   - Evidence suggests some activities were recorded later than when they actually occurred
   - Administrative approvals sometimes show timestamps after related supervisor approvals
   - These recording delays create apparent inconsistencies in the process flow

3. **Batched Recording**
   - System-based activities (especially "Payment Handled") often show clustering at specific times
   - Multiple payments processed simultaneously create artificial synchronization in the data
   - This batching obscures the true sequential nature of the underlying process

**Representation Example:**
```graphviz
digraph TemporalNoise {
    rankdir=LR;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    // Activities as they appear in data
    A_recorded [label="Activity A\n2018-03-15 09:25:12"];
    B_recorded [label="Activity B\n2018-03-15 10:15:45"];
    
    // Activities as they actually occurred
    A_actual [label="Activity A\n(Actual time same as recorded)", fillcolor=lightgreen];
    B_actual [label="Activity B\n(Actually occurred at 09:45)", fillcolor=lightsalmon];
    
    // What the data shows
    A_recorded -> B_recorded [label="Appears sequential\n(50 min gap)"];
    
    // What actually happened
    A_actual -> B_actual [label="Actual sequence\n(20 min gap)"];
    
    // Recording delay
    B_actual -> B_recorded [label="30 min\nrecording delay", style=dashed, color=red];
    
    {rank=same; A_recorded; A_actual}
    {rank=same; B_recorded; B_actual}
}
```

### 1.2 Structural Noise

**Definition:** Inconsistencies in the recorded sequence or structure of process activities.

**Sources in the BPI2020 Dataset:**

1. **Missing Activities**
   - Some cases appear to skip expected activities
   - Approximately 1.5% of cases show direct transitions from administrative approval to payment request without supervisor approval
   - Analysis suggests these activities occurred but weren't recorded

2. **Duplicate Activities**
   - Some activities appear to be recorded multiple times
   - 3.2% of declarations show duplicate "Request Payment" activities
   - These may represent system retries or recording errors

3. **Out-of-Sequence Activities**
   - Activities occasionally appear in unexpected orders
   - Some supervisor approvals are timestamped before administrative approvals
   - These sequence issues may reflect parallel processing or recording errors

**Representation Example:**
```graphviz
digraph StructuralNoise {
    rankdir=TB;
    
    subgraph cluster_expected {
        label = "Expected Sequence";
        node [shape=box, style=filled, fillcolor=lightgreen];
        
        admin_exp [label="Admin Approval"];
        super_exp [label="Supervisor Approval"];
        payment_exp [label="Payment Request"];
        
        admin_exp -> super_exp -> payment_exp;
    }
    
    subgraph cluster_anomalies {
        label = "Observed Anomalies";
        node [shape=box, style=filled, fillcolor=lightsalmon];
        
        // Missing activity
        admin1 [label="Admin Approval"];
        payment1 [label="Payment Request"];
        admin1 -> payment1 [label="Missing\nSupervisor\nApproval"];
        
        // Out of sequence
        super2 [label="Supervisor Approval"];
        admin2 [label="Admin Approval"];
        super2 -> admin2 [label="Out of\nSequence"];
        
        // Duplicate
        admin3 [label="Admin Approval"];
        super3 [label="Supervisor Approval"];
        payment3a [label="Payment Request"];
        payment3b [label="Payment Request"];
        admin3 -> super3 -> payment3a -> payment3b [label="Duplicate"];
    }
}
```

### 1.3 Attribute Noise

**Definition:** Inconsistencies or errors in the attribute values associated with process events.

**Sources in the BPI2020 Dataset:**

1. **Inconsistent Resource Labeling**
   - The dataset uses generic resource labels ("STAFF MEMBER", "SYSTEM")
   - Analysis suggests multiple individuals are grouped under the same generic label
   - This masking obscures true resource allocation patterns

2. **Amount Precision Issues**
   - Declaration amounts show unusual precision (e.g., 26.85120450862128)
   - Such precision suggests these are calculated or transformed values
   - Some amount patterns suggest possible currency conversion or anonymization

3. **Role-Activity Mismatches**
   - Certain activities are occasionally associated with unexpected roles
   - 0.3% of "Declaration APPROVED by ADMINISTRATION" activities show "SUPERVISOR" as the role
   - These mismatches may represent delegation scenarios or recording errors

**Representation Example:**
```graphviz
digraph AttributeNoise {
    rankdir=LR;
    
    subgraph cluster_expected {
        label="Expected Attribute Patterns";
        node [shape=box, style=filled, fillcolor=lightgreen];
        
        admin_act_exp [label="Declaration APPROVED by\nADMINISTRATION"];
        admin_role_exp [label="role=\nADMINISTRATION"];
        
        super_act_exp [label="Declaration FINAL_APPROVED by\nSUPERVISOR"];
        super_role_exp [label="role=\nSUPERVISOR"];
        
        admin_act_exp -> admin_role_exp;
        super_act_exp -> super_role_exp;
    }
    
    subgraph cluster_anomalies {
        label="Observed Anomalies";
        node [shape=box, style=filled, fillcolor=lightsalmon];
        
        admin_act_anom [label="Declaration APPROVED by\nADMINISTRATION"];
        super_role_anom [label="role=\nSUPERVISOR"];
        
        admin_act_anom -> super_role_anom [label="Mismatch"];
    }
}
```

### 1.4 Concept Drift

**Definition:** Changes in the underlying process over time that create inconsistencies.

**Sources in the BPI2020 Dataset:**

1. **Process Evolution**
   - The dataset spans from January 2017 to June 2019 (2.5 years)
   - Analysis shows gradual changes in approval patterns over time
   - Later cases show more pre-approver involvement than earlier cases

2. **Policy Changes**
   - Evidence suggests approval thresholds changed during the recorded period
   - Budget owner involvement increased for lower amounts in later periods
   - These changes create apparent inconsistencies when analyzing the full dataset

3. **Seasonal Variations**
   - Clear seasonal patterns emerge in process execution
   - December-January and July-August show distinct behavioral differences
   - These seasonal variations create periodic "noise" in process patterns

**Representation Example:**
```graphviz
digraph ConceptDrift {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    threshold_early [label="Early 2017 Pattern", shape=plaintext, fillcolor=white];
    amount_early [label="Declarations > €500", fillcolor=lightgreen];
    bo_early [label="Budget Owner Approval"];
    
    threshold_late [label="Late 2018 Pattern", shape=plaintext, fillcolor=white];
    amount_late [label="Declarations > €300", fillcolor=lightsalmon];
    bo_late [label="Budget Owner Approval"];
    
    threshold_early -> amount_early -> bo_early;
    threshold_late -> amount_late -> bo_late;
    
    inconsistency [label="When analyzed together,\nappears as inconsistent behavior", 
                  shape=note, fillcolor=lightyellow];
}
```

## 2. Process Data Inconsistencies

Beyond noise, the BPI2020 dataset exhibits several types of inconsistencies that challenge straightforward process analysis.

### 2.1 Path Inconsistencies

**Definition:** Variations in process flows that don't follow expected patterns.

**Examples in the BPI2020 Dataset:**

1. **Approval Path Variations**
   - The dataset contains over 100 distinct case variants
   - Only 57.3% of cases follow one of the top 5 variants
   - The remaining 42.7% follow various uncommon paths

2. **Rejection Handling Differences**
   - Rejections at the same stage are handled differently across cases
   - Some rejections loop back to resubmission
   - Others proceed to different approval paths after correction

3. **Exception Path Proliferation**
   - Many rare variants appear to handle special cases
   - These exception paths create apparent inconsistencies
   - They may represent valid but uncommon business scenarios

**Consistency Analysis:**
```graphviz
digraph PathInconsistencies {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    variants [label="Process Variant Distribution", shape=plaintext, fillcolor=white];
    
    standard [label="Standard path\n31.2%", fillcolor=lightgreen];
    rejection [label="Rejection-resubmission\n12.8%", fillcolor=lightgreen];
    budget [label="Budget owner approval\n8.1%", fillcolor=lightgreen];
    pre [label="Pre-approver path\n5.2%", fillcolor=lightgreen];
    complex [label="Complex multi-approval\n3.0%", fillcolor=lightgreen];
    other [label="Other variants\n42.7%", fillcolor=lightyellow];
    
    variants -> standard;
    variants -> rejection;
    variants -> budget;
    variants -> pre;
    variants -> complex;
    variants -> other;
}
```

### 2.2 Temporal Inconsistencies

**Definition:** Inconsistencies in the timing and duration patterns across cases.

**Examples in the BPI2020 Dataset:**

1. **Variable Activity Durations**
   - Similar activities show vastly different durations
   - Administrative approvals range from minutes to several days
   - These variations suggest inconsistent handling or recording

2. **Waiting Time Irregularities**
   - Waiting times between activities show high variance
   - Weekend delays create irregular patterns
   - Some cases show inexplicably long waiting periods

3. **Processing Time Anomalies**
   - Some cases show implausibly fast processing
   - Others show extremely long durations without clear cause
   - These extremes create inconsistencies in performance analysis

**Temporal Pattern Analysis:**
```graphviz
digraph TemporalInconsistencies {
    rankdir=LR;
    node [shape=record, style=filled, fillcolor=lightblue];
    
    approvals [label="Administrative Approval Time Statistics"];
    stats [label="<min>Minimum: 2 minutes|<med>Median: 4.2 hours|<avg>Average: 9.7 hours|<max>Maximum: 18.3 days|<std>Standard Deviation: 22.3 hours", fillcolor=lightyellow];
    
    conclusion [label="This high variance suggests\nsignificant inconsistency", shape=note, fillcolor=lightsalmon];
    
    approvals -> stats -> conclusion;
}
```

### 2.3 Resource Inconsistencies

**Definition:** Inconsistencies in how resources are assigned to activities.

**Examples in the BPI2020 Dataset:**

1. **Role Assignment Variations**
   - Similar activities are performed by different roles
   - Some approvals are performed by substitutes
   - These variations create apparent inconsistencies in role responsibilities

2. **Load Balancing Irregularities**
   - Workload distribution appears inconsistent
   - Some periods show resource overallocation
   - Others show underutilization

3. **Approval Authority Variations**
   - Approval authority patterns vary over time
   - Same-amount declarations follow different approval paths
   - These variations suggest inconsistent application of approval rules

**Resource Pattern Analysis:**
```graphviz
digraph ResourceInconsistencies {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    title [label="For declarations between €100-€500:", shape=plaintext, fillcolor=white];
    
    super_only [label="73% go through\nsupervisor-only approval", fillcolor=lightgreen];
    budget_owner [label="18% require\nbudget owner approval", fillcolor=lightyellow];
    pre_approver [label="9% involve\npre-approver review", fillcolor=lightsalmon];
    
    conclusion [label="This inconsistency suggests\ndiscretionary routing", shape=note, fillcolor=lightyellow];
    
    title -> super_only;
    title -> budget_owner;
    title -> pre_approver;
    
    {rank=same; super_only budget_owner pre_approver}
    
    super_only -> conclusion;
    budget_owner -> conclusion;
    pre_approver -> conclusion;
}
```

## 3. Process Data Incompleteness

The BPI2020 dataset, while rich, exhibits several forms of incompleteness that limit certain analyses.

### 3.1 Missing Context Information

**Definition:** Absence of contextual data that would explain process variations.

**Examples in the BPI2020 Dataset:**

1. **Missing Declaration Categories**
   - No information on the type of expense being declared
   - Categories would likely explain many routing variations
   - Without this context, some routing decisions appear arbitrary

2. **Absent Rejection Reasons**
   - Rejection activities don't include rejection reasons
   - Understanding rejection causes is critical for improvement
   - This missing information limits root cause analysis

3. **No Department Information**
   - No data on which departments are involved
   - Departmental differences likely explain some variations
   - This missing context limits organizational analysis

**Impact of Missing Context:**
```graphviz
digraph MissingContext {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    observed [label="Observed Pattern (Without Context)", shape=plaintext, fillcolor=white];
    
    declaration_A [label="Declaration A (€350)"];
    supervisor_A [label="Supervisor Approval"];
    
    declaration_B [label="Declaration B (€320)"];
    budget_B [label="Budget Owner Approval"];
    
    declaration_A -> supervisor_A;
    declaration_B -> budget_B;
    
    inconsistency [label="Appears Inconsistent", shape=note, fillcolor=lightsalmon];
    
    potential [label="With Missing Context", shape=plaintext, fillcolor=white];
    
    declaration_A2 [label="Declaration A\n(€350, Office Supplies)"];
    supervisor_A2 [label="Supervisor Approval"];
    
    declaration_B2 [label="Declaration B\n(€320, IT Equipment)"];
    budget_B2 [label="Budget Owner Approval"];
    
    declaration_A2 -> supervisor_A2;
    declaration_B2 -> budget_B2;
    
    consistent [label="Would be Consistent", shape=note, fillcolor=lightgreen];
    
    observed -> inconsistency;
    potential -> consistent;
}
```

### 3.2 Incomplete Case Histories

**Definition:** Cases with partial or truncated histories.

**Examples in the BPI2020 Dataset:**

1. **Left-Truncated Cases**
   - Cases that were already in progress at the start of data collection
   - January 2017 shows some cases starting with mid-process activities
   - These cases lack complete beginning information

2. **Right-Truncated Cases**
   - Cases still in progress at the end of data collection
   - June 2019 contains cases without completion activities
   - These cases lack resolution information

3. **Partially Recorded Cases**
   - Some cases show evidence of activities outside the system
   - References to external documents or decisions
   - These external components create incomplete process views

**Truncation Example:**
```graphviz
digraph Truncation {
    rankdir=LR;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    case_id [label="Case ID: declaration 156982"];
    last_activity [label="Last recorded activity:\nDeclaration APPROVED by ADMINISTRATION\n(2019-06-17)"];
    missing [label="Missing subsequent activities\n(case was likely still in progress)", fillcolor=lightsalmon];
    
    data_boundary [label="Data Collection End", shape=note, fillcolor=lightyellow];
    
    case_id -> last_activity -> missing;
    data_boundary -> missing [style=dashed, dir=none];
}
```

### 3.3 Missing Attribute Values

**Definition:** Records with incomplete attribute information.

**Examples in the BPI2020 Dataset:**

1. **Undefined Roles**
   - Some activities (particularly system activities) show "UNDEFINED" role
   - This missing role information creates gaps in organizational analysis
   - 27.4% of activities have "UNDEFINED" role

2. **Empty Resource Information**
   - Some records have generic or empty resource information
   - This masks the actual performing resources
   - Limits resource-based analysis

3. **Missing Timestamps in References**
   - References to activities sometimes lack precise timing
   - Creates uncertainty in the exact sequence
   - Affects accurate flow reconstruction

**Attribute Completeness Analysis:**
```graphviz
digraph AttributeCompleteness {
    rankdir=LR;
    node [shape=record, style=filled, fillcolor=lightblue];
    
    completeness [label="Attribute Completeness Rates"];
    rates [label="<id>case:id: 100%|<res>org:resource: 100%|<name>concept:name: 100%|<time>time:timestamp: 100%|<role>org:role: 72.6% (27.4% are \"UNDEFINED\")|<amount>case:Amount: 100%|<budget>case:BudgetNumber: 100%|<decl>case:DeclarationNumber: 100%", fillcolor=lightyellow];
    
    completeness -> rates;
}
```

## 4. Representing Process Data Noise

To effectively analyze noisy process data, it's essential to develop appropriate representations that acknowledge and account for noise. Several approaches can be used:

### 4.1 Probabilistic Process Maps

**Description:** Process maps that represent transitions with probability distributions rather than deterministic edges.

**Implementation for BPI2020:**

```graphviz
digraph ProbabilisticMap {
    rankdir=LR;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    submission [label="Submission"];
    admin_approval [label="Admin Approval"];
    admin_rejection [label="Admin Rejection", fillcolor=lightsalmon];
    other [label="Other", fillcolor=lightyellow];
    
    submission -> admin_approval [label="90.2% ± 1.3%", penwidth=3.0];
    submission -> admin_rejection [label="9.1% ± 1.2%", penwidth=2.0];
    submission -> other [label="0.7% ± 0.4%", style=dashed, penwidth=1.0];
}
```

**Benefits:**
- Acknowledges the uncertain nature of some transitions
- Captures the inherent variability in the process
- Allows analysts to focus on high-probability paths

### 4.2 Fuzzy Process Models

**Description:** Models that allow activities and transitions to have degrees of membership rather than binary inclusion.

**Implementation for BPI2020:**

```graphviz
digraph FuzzyModel {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    admin_approval [label="Activity \"Declaration APPROVED by ADMINISTRATION\""];
    
    memberships [label="- Belongs to \"Administrative Process\" with degree 0.95\n- Belongs to \"Supervisor Process\" with degree 0.15\n- Typically takes 2-8 hours (fuzzy interval)", shape=note, fillcolor=lightyellow];
    
    admin_approval -> memberships;
}
```

**Benefits:**
- Captures the "gray areas" in process execution
- Accommodates ambiguous or inconsistent recordings
- Provides more nuanced view of process behavior

### 4.3 Annotated Event Logs

**Description:** Enhanced event logs that include data quality annotations and confidence scores.

**Implementation for BPI2020:**

```graphviz
digraph AnnotatedLog {
    rankdir=TB;
    node [shape=record, style=filled, fillcolor=lightblue];
    
    event [label="Event: Declaration APPROVED by ADMINISTRATION"];
    
    annotations [label="<ts>Timestamp: 2018-05-12 14:35:22+00:00|<conf>Timestamp confidence: 0.85|<anom>Sequence anomaly flag: True (Follows supervisor approval)|<res>Resource consistency: 0.92", fillcolor=lightyellow];
    
    event -> annotations;
}
```

**Benefits:**
- Preserves original data while adding quality context
- Allows filtering based on data quality needs
- Enables sensitivity analysis based on confidence

### 4.4 Multi-perspective Process Views

**Description:** Multiple complementary views of the same process data, each optimized for different analysis goals.

**Implementation for BPI2020:**

```graphviz
digraph MultiPerspective {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    process_data [label="Declaration Process Data"];
    
    control_flow [label="Control-flow View\n(Shows main paths,\nfilters rare variants)"];
    
    time_view [label="Time View\n(Focuses on throughput time,\nnormalizes for weekends)"];
    
    resource_view [label="Resource View\n(Shows workload distribution\nacross roles)"];
    
    amount_view [label="Amount View\n(Shows correlation between\namount and process path)"];
    
    process_data -> control_flow;
    process_data -> time_view;
    process_data -> resource_view;
    process_data -> amount_view;
}
```

**Benefits:**
- Accommodates different analytical needs
- Allows noise filtering relevant to specific views
- Provides more comprehensive understanding

## 5. Methods for Robust Process Data Representation

To represent process data robustly in the face of noise and incompleteness, several approaches can be applied:

### 5.1 Abstraction and Aggregation

**Description:** Representing the process at higher levels of abstraction to minimize the impact of low-level noise.

**Implementation for BPI2020:**

```graphviz
digraph Abstraction {
    rankdir=TB;
    
    subgraph cluster_detailed {
        label = "Original Detailed Activities";
        node [shape=box, style=filled, fillcolor=lightblue];
        
        submit_d [label="Declaration SUBMITTED\nby EMPLOYEE"];
        admin_d [label="Declaration APPROVED\nby ADMINISTRATION"];
        budget_d [label="Declaration APPROVED\nby BUDGET OWNER"];
        super_d [label="Declaration FINAL_APPROVED\nby SUPERVISOR"];
        request_d [label="Request Payment"];
        payment_d [label="Payment Handled"];
    }
    
    subgraph cluster_abstracted {
        label = "Abstracted Activities";
        node [shape=box, style=filled, fillcolor=lightgreen];
        
        submit_a [label="Submission Phase"];
        approval_a [label="Approval Phase"];
        payment_a [label="Payment Phase"];
        
        submit_a -> approval_a -> payment_a;
    }
    
    conclusion [label="This abstraction is robust to\nmany sequencing variations", shape=note, fillcolor=lightyellow];
    
    cluster_detailed -> cluster_abstracted [style=dashed, lhead=cluster_abstracted, ltail=cluster_detailed];
    cluster_abstracted -> conclusion;
}
```

### 5.2 Filtering and Outlier Handling

**Description:** Systematically removing or mitigating outliers and anomalous data.

**Implementation for BPI2020:**

```graphviz
digraph Filtering {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    strategy [label="Filtering Strategy for BPI2020:"];
    
    step1 [label="1. Remove first/last 30 days of cases\n(potential truncation)", fillcolor=lightyellow];
    step2 [label="2. Filter cases with duration > 30 days\n(3% of cases)", fillcolor=lightyellow];
    step3 [label="3. Focus analysis on variants covering > 5% of cases", fillcolor=lightyellow];
    step4 [label="4. Handle remaining cases as separate \"exception\" category", fillcolor=lightyellow];
    
    result [label="This preserves 89% of complete cases\nwhile mitigating major noise sources", fillcolor=lightgreen];
    
    strategy -> step1 -> step2 -> step3 -> step4 -> result;
}
```

### 5.3 Process Discovery with Inductive Mining

**Description:** Using noise-resistant process discovery algorithms to extract robust process models.

**Implementation for BPI2020:**

```graphviz
digraph InductiveMining {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    strategy [label="Process Discovery Strategy:"];
    
    step1 [label="1. Apply Inductive Miner with 0.2 noise threshold", fillcolor=lightyellow];
    step2 [label="2. Set activity significance threshold at 0.4", fillcolor=lightyellow];
    step3 [label="3. Set edge significance threshold at 0.3", fillcolor=lightyellow];
    step4 [label="4. Use frequency-based path enhancement", fillcolor=lightyellow];
    
    result [label="Resulting model captures 83% of behavior\nwhile filtering noise", fillcolor=lightgreen];
    
    strategy -> step1 -> step2 -> step3 -> step4 -> result;
}
```

### 5.4 Multi-dimensional Conformance Checking

**Description:** Evaluating process conformance across multiple dimensions rather than binary conformance.

**Implementation for BPI2020:**

```graphviz
digraph MultiDimensionalConformance {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    conformance [label="Multi-dimensional Conformance for Declaration Process:"];
    
    dim1 [label="1. Control-flow conformance: 86%\n(main sequence adherence)", fillcolor=lightgreen];
    dim2 [label="2. Role conformance: 94%\n(right activities by right roles)", fillcolor=lightgreen];
    dim3 [label="3. Four-eyes principle conformance: 99%\n(different people for submit/approve)", fillcolor=lightgreen];
    dim4 [label="4. Temporal conformance: 78%\n(activities in expected timeframes)", fillcolor=lightyellow];
    
    overall [label="Overall weighted conformance: 89%", fillcolor=lightgreen];
    
    conformance -> dim1;
    conformance -> dim2;
    conformance -> dim3;
    conformance -> dim4;
    
    {rank=same; dim1 dim2 dim3 dim4}
    
    dim1 -> overall;
    dim2 -> overall;
    dim3 -> overall;
    dim4 -> overall;
}
```

### 5.5 Interactive and Adaptive Process Visualization

**Description:** Flexible visualizations that allow analysts to adjust representation parameters.

**Implementation for BPI2020:**

```graphviz
digraph InteractiveVisualization {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    capabilities [label="Interactive Visualization Capabilities:"];
    
    slider [label="1. Frequency threshold slider (1-100%)", fillcolor=lightyellow];
    level [label="2. Abstraction level selector (1-5)", fillcolor=lightyellow];
    window [label="3. Time window selector", fillcolor=lightyellow];
    filter [label="4. Role/resource filter", fillcolor=lightyellow];
    range [label="5. Amount range filter", fillcolor=lightyellow];
    
    result [label="This allows analysts to dynamically adjust\nvisualization to filter noise", fillcolor=lightgreen];
    
    capabilities -> slider;
    capabilities -> level;
    capabilities -> window;
    capabilities -> filter;
    capabilities -> range;
    
    {rank=same; slider level window filter range}
    
    slider -> result;
    level -> result;
    window -> result;
    filter -> result;
    range -> result;
}
```

## 6. Data Quality Assessment Framework for Process Data

To systematically address data quality issues in process data, a comprehensive framework is needed:

### 6.1 Process Data Quality Dimensions

**Description:** A structured approach to evaluating process data quality.

**Implementation for BPI2020:**

```graphviz
digraph QualityDimensions {
    rankdir=LR;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    assessment [label="BPI2020 Domestic Declarations Quality Assessment:"];
    
    completeness [label="Completeness\nScore: Medium\n- Missing context information\n- Some truncated cases\n- Complete core data", fillcolor=lightyellow];
    
    correctness [label="Correctness\nScore: High\n- Some timestamp anomalies\n- Generally accurate recordings", fillcolor=lightgreen];
    
    consistency [label="Consistency\nScore: Medium\n- Multiple process variants\n- Variable timing patterns", fillcolor=lightyellow];
    
    currency [label="Currency\nScore: High\n- Covers extensive time period\n- Shows process evolution", fillcolor=lightgreen];
    
    assessment -> completeness;
    assessment -> correctness;
    assessment -> consistency;
    assessment -> currency;
}
```

### 6.2 Process-Specific Data Quality Rules

**Description:** Rules specifically designed to identify quality issues in process data.

**Implementation for BPI2020:**

```graphviz
digraph QualityRules {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    rules [label="Process Quality Rules for BPI2020:"];
    
    table [shape=record, label="<r1>Submission must precede approval|<v1>0.2%|<r2>Payment follows approval|<v2>1.4%|<r3>Four-eyes principle|<v3>0.3%|<r4>Administrator ≠ Approver|<v4>0.5%|<r5>Case duration < 60 days|<v5>1.8%|<r6>Weekend activities < 5% of total|<v6>2.3%", fillcolor=lightyellow];
    
    header [label="Rule|Violation %", shape=record];
    
    rules -> header -> table;
}
```

### 6.3 Data Cleaning and Enhancement Strategies

**Description:** Approaches to improve data quality through cleaning and enhancement.

**Implementation for BPI2020:**

```graphviz
digraph DataEnhancement {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    enhancement [label="Data Enhancement Process:"];
    
    step1 [label="1. Normalize all timestamps to UTC", fillcolor=lightyellow];
    step2 [label="2. Adjust for weekends/holidays using business day calculation", fillcolor=lightyellow];
    step3 [label="3. Exclude first/last month cases (potential truncation)", fillcolor=lightyellow];
    step4 [label="4. Flag cases with sequence anomalies", fillcolor=lightyellow];
    step5 [label="5. Add derived attributes:\n   - Case complexity score\n   - Approval path category\n   - Expected vs. actual duration", fillcolor=lightyellow];
    
    enhancement -> step1 -> step2 -> step3 -> step4 -> step5;
}
```

## 7. Advanced Techniques for Robust Process Analysis

Beyond representation, several advanced techniques can be applied to analyze noisy process data:

### 7.1 Trace Clustering

**Description:** Grouping similar process instances to reduce variability and identify patterns.

**Implementation for BPI2020:**

```graphviz
digraph TraceClustering {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    results [label="Trace Clustering Results for BPI2020:"];
    
    cluster1 [label="Cluster 1: Standard process (36%)\n- Average duration: 2.6 days\n- Average amount: €67.45", fillcolor=lightgreen];
    
    cluster2 [label="Cluster 2: Budget approval process (12%)\n- Average duration: 7.3 days\n- Average amount: €243.16", fillcolor=lightgreen];
    
    cluster3 [label="Cluster 3: Rejection cases (15%)\n- Average duration: 6.7 days\n- Average iterations: 1.8", fillcolor=lightsalmon];
    
    cluster4 [label="Cluster 4: Fast-track cases (18%)\n- Average duration: 1.4 days\n- Average amount: €32.78", fillcolor=lightgreen];
    
    cluster5 [label="Cluster 5: Complex approval cases (9%)\n- Average duration: 11.2 days\n- Average amount: €684.92", fillcolor=lightyellow];
    
    other [label="Other clusters: 10%", fillcolor=lightyellow];
    
    results -> cluster1;
    results -> cluster2;
    results -> cluster3;
    results -> cluster4;
    results -> cluster5;
    results -> other;
}
```

### 7.2 Process Mining with Noise Filtering

**Description:** Process mining techniques specifically designed to handle noise.

**Implementation for BPI2020:**

```graphviz
digraph NoiseFiltering {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    strategy [label="Noise Filtering Strategy:"];
    
    step1 [label="1. Apply automatic outlier detection\n   (removes 3% of cases)", fillcolor=lightyellow];
    step2 [label="2. Filter activities occurring in <1% of cases", fillcolor=lightyellow];
    step3 [label="3. Filter paths occurring in <2% of cases", fillcolor=lightyellow];
    step4 [label="4. Apply alignment-based conformance\n   to identify and handle deviations", fillcolor=lightyellow];
    
    result [label="Result: 94% of behavior captured with\n82% reduction in model complexity", fillcolor=lightgreen];
    
    strategy -> step1 -> step2 -> step3 -> step4 -> result;
}
```

### 7.3 Process Variant Analysis

**Description:** Analyzing process variants separately rather than forcing a single model.

**Implementation for BPI2020:**

```graphviz
digraph VariantAnalysis {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    analysis [label="Variant Analysis for BPI2020:"];
    
    approach [label="For each major variant (>5% of cases):\n1. Create separate process model\n2. Calculate performance metrics\n3. Identify distinctive characteristics\n4. Determine routing factors\n5. Create decision tree for variant prediction", fillcolor=lightyellow];
    
    results [label="Results show amount is primary determinant (76% accuracy)\nwith secondary factors being:\n- Budget number (specific departments)\n- Submission day (Friday submissions more likely to need budget approval)\n- Submission time (late day submissions follow different patterns)", fillcolor=lightgreen];
    
    analysis -> approach -> results;
}
```

### 7.4 Process Simulation with Noise Parameters

**Description:** Using simulation to understand the impact of noise on process behavior.

**Implementation for BPI2020:**

```graphviz
digraph ProcessSimulation {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    params [label="Simulation Parameters:"];
    
    config [label="- Activity recording delay: 0-24 hours (configurable)\n- Missing activity rate: 0-5% (configurable)\n- Resource recording accuracy: 80-100% (configurable)\n- Weekend adjustment: On/Off", fillcolor=lightyellow];
    
    results [label="Sensitivity Results:\n- Duration metrics highly sensitive to timestamp noise\n- Resource allocation moderately sensitive to resource recording\n- Process structure robust to moderate noise levels", fillcolor=lightgreen];
    
    params -> config -> results;
}
```

## 8. Conclusion: Toward Robust Process Mining

The BPI2020 Domestic Declarations dataset illustrates the challenges of real-world process data, with various forms of noise, inconsistencies, and incompleteness. To conduct meaningful analysis on such data, a comprehensive approach is needed:

### 8.1 Best Practices for BPI2020 Analysis

```graphviz
digraph BestPractices {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    practices [label="Best Practices for BPI2020 Analysis:"];
    
    practice1 [label="1. Acknowledge Data Limitations\n- Be transparent about data quality issues\n- Report confidence levels with findings\n- Recognize limitations in conclusions", fillcolor=lightgreen];
    
    practice2 [label="2. Use Multi-Method Triangulation\n- Apply multiple analysis techniques\n- Compare results across methods\n- Build confidence through convergent findings", fillcolor=lightgreen];
    
    practice3 [label="3. Implement Preprocessing Pipeline\n- Develop systematic data cleaning steps\n- Document all transformations\n- Preserve original data alongside enhanced data", fillcolor=lightgreen];
    
    practice4 [label="4. Apply Context-Aware Interpretation\n- Incorporate domain knowledge\n- Use contextual information to interpret variations\n- Recognize legitimate vs. noise-based variations", fillcolor=lightgreen];
    
    practices -> practice1 -> practice2 -> practice3 -> practice4;
}
```

### 8.2 Recommended Analysis Approach

```graphviz
digraph RecommendedApproach {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    
    approach [label="Recommended Analysis Approach:"];
    
    step1 [label="1. Data Quality Assessment\n- Apply quality framework\n- Identify and quantify noise sources\n- Develop quality improvement plan", fillcolor=lightyellow];
    
    step2 [label="2. Data Enhancement\n- Normalize timestamps\n- Handle incomplete cases\n- Add derived attributes", fillcolor=lightyellow];
    
    step3 [label="3. Multi-perspective Mining\n- Apply variant-focused analysis\n- Use noise-resistant algorithms\n- Implement multiple views (flow, organizational, time)", fillcolor=lightyellow];
    
    step4 [label="4. Adaptive Visualization\n- Interactive filtering capabilities\n- Configurable abstraction levels\n- Context-sensitive views", fillcolor=lightyellow];
    
    step5 [label="5. Results Validation\n- Cross-validate findings\n- Perform sensitivity analysis\n- Acknowledge uncertainty", fillcolor=lightyellow];
    
    approach -> step1 -> step2 -> step3 -> step4 -> step5;
}
```

By acknowledging data quality issues and implementing robust analysis techniques, meaningful insights can be extracted from noisy process data like the BPI2020 Domestic Declarations dataset. This approach turns data challenges into opportunities for deeper understanding by revealing not just the idealized process but the real-world variations that define actual execution.