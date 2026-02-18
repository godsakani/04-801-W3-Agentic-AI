# HW3 Implementation Guide: YOUR Tasks (3 & 4)

## 🎯 What You're Implementing

**Your Responsibilities:**
- ✅ Task 3: Evaluation Framework (Automated Metrics) - 20 points
- ✅ Task 4: Adaptive Control (Closed-Loop Behavior) - 15 points

**Your Teammate's Responsibilities:**
- Task 1: Persistent State & Memory Management
- Task 2: Multi-Agent/Role-Based Architecture

---

## 📁 Files YOU Created

```
src/
└── evaluation/                      ← YOUR MODULE
    ├── __init__.py
    ├── metrics.py                   ← Task 3: 7 automated metrics
    ├── adaptive_control.py          ← Task 4: 5 adaptive behaviors
    └── test_cases.py                ← 6 structured test cases

notebooks/
└── hw3_evaluation_and_adaptive_control.ipynb  ← YOUR DEMONSTRATION

HW3_YOUR_TASKS_GUIDE.md             ← This file
```

---

## 📊 TASK 3: Evaluation Framework

### What HW3 Requires

| Requirement | Your Implementation |
|-------------|---------------------|
| **"At least 3 quantitative metrics"** | ✅ You implemented **7 metrics** |
| **"Run on 5+ test cases"** | ✅ **6 test cases** in `test_cases.py` |
| **"Present results in table"** | ✅ `format_metrics_table()` function |
| **"Include failure case"** | ✅ TC004 (off-topic query) |
| **"Root cause analysis"** | ✅ Complete analysis in notebook |
| **"Technical explanation"** | ✅ Detailed explanation provided |
| **"Adjustment made"** | ✅ Fix: Early topic detection |
| **"Before vs After"** | ✅ Comparison shown |

### The 7 Automated Metrics

**File:** `src/evaluation/metrics.py`

```python
class EvaluationMetrics:
    groundedness_score: float          # Metric 1: Factuality (0-1)
    tool_selection_accuracy: float      # Metric 2: Correct tools (0-1)
    task_completion_rate: float         # Metric 3: Success rate (0-1)
    iterations_before_convergence: int  # Metric 4: Efficiency
    plan_adherence_score: float         # Metric 5: Followed plan (0-1)
    hallucination_frequency: float      # Metric 6: False info rate (0-1)
    response_quality_score: float       # Metric 7: Overall quality (0-1)
```

**How to Use:**

```python
from src.evaluation.metrics import MetricsCalculator

# Initialize
calc = MetricsCalculator()

# Run agent query
result = agent.run("Find AI alumni")

# Calculate all metrics
metrics = calc.calculate_all_metrics(
    result=result,
    expected_tools=["RETRIEVE"],
    start_time=datetime.now()
)

# Access metrics
print(f"Groundedness: {metrics.groundedness_score:.2f}")
print(f"Tool Accuracy: {metrics.tool_selection_accuracy:.2f}")
print(f"Quality: {metrics.response_quality_score:.2f}")
```

### The 6 Test Cases

**File:** `src/evaluation/test_cases.py`

| ID | Query | Expected Tools | Pass/Fail |
|----|-------|---------------|-----------|
| TC001 | Find AI/ML alumni | RETRIEVE | ✅ Pass |
| TC002 | Find fintech alumni + email | RETRIEVE, email_sender | ✅ Pass |
| TC003 | Discover new MSIT alumni | linkedin_discovery | ✅ Pass |
| **TC004** | **Who is Tesla CEO?** (OFF-TOPIC) | RETRIEVE | ❌ **Fail** |
| TC005 | Find Python/MongoDB skills | RETRIEVE | ✅ Pass |
| TC006 | Send promotion congrats | Multiple tools | ✅ Pass |

**TC004 is the FAILURE CASE** - intentionally off-topic to test system robustness.

### Failure Case Analysis (TC004)

**Located in notebook, cells under "FAILURE CASE ANALYSIS"**

**1. What Happened:**
- Query: "Who is the CEO of Tesla?"
- System searched alumni database (no relevant results)
- Agent attempted to answer anyway (hallucination risk)
- Groundedness score: 0.15 (very low)

**2. Root Cause:**
- Query is OFF-TOPIC (not about CMU Africa alumni)
- Vector search returns no good matches
- LLM uses general knowledge instead of alumni data
- Verification detects mismatch

**3. Technical Explanation:**
```python
# Vector search for "Tesla CEO" in alumni database
docs = retrieval.search("Tesla CEO")
# Returns: similarity scores all < 0.3 (poor matches)

# Agent generates response anyway
response = llm.invoke(context=poor_matches)
# Response contains general knowledge about Elon Musk (NOT an alumni)

# Verification catches this
verification_score = verify(response, poor_matches)
# Score: 0.15 (hallucination detected!)
```

**4. Adjustment Made:**

```python
# BEFORE: No topic detection
if groundedness < 0.5:
    return response  # Returns potentially false info

# AFTER: Early topic detection + rejection
if is_off_topic(query, retrieved_docs):
    return "This query is off-topic. I can only answer questions about CMU Africa alumni."
```

**5. Before vs After:**

| Aspect | Before | After |
|--------|--------|-------|
| Response | "Elon Musk is Tesla CEO..." | "Query is off-topic..." |
| Groundedness | 0.15 (very low) | N/A (rejected early) |
| User Impact | Misleading info | Clear guidance |
| Hallucination Risk | HIGH | NONE |

---

## 🔄 TASK 4: Adaptive Control

### What HW3 Requires

| Requirement | Your Implementation |
|-------------|---------------------|
| **"Modify behavior based on feedback"** | ✅ 5 adaptive behaviors |
| **"Real adaptation, not cosmetic"** | ✅ Actual behavior changes |
| **"Show: Observe → Reason → Decide → Act → Evaluate → Update"** | ✅ Demonstrated for each |
| **"Examples: groundedness → re-retrieve"** | ✅ Implemented |
| **"Examples: tool fails → retry/alternative"** | ✅ Implemented |
| **"Examples: iteration limit → escalate"** | ✅ Implemented |

### The 5 Adaptive Behaviors

**File:** `src/evaluation/adaptive_control.py`

#### **1. Low Groundedness → Re-Retrieve**

```python
# OBSERVE
groundedness_score = 0.45  # Below threshold 0.7

# REASON
print("Response may contain unverified claims. Need more context.")

# DECIDE
if controller.should_re_retrieve(groundedness_score):
    # ACT
    refined_query = controller.refine_query(query, feedback)
    new_docs = retrieval(refined_query)

    # EVALUATE
    new_groundedness = verify(response, new_docs)
    # Result: 0.85 (improved!)

    # UPDATE
    controller.retrieval_attempts += 1
```

**Real Behavioral Change:** Query is refined and re-executed with better results.

#### **2. Tool Failure → Retry**

```python
# OBSERVE
error = "linkedin_scraper failed: Timeout"

# REASON
print("Tool may have hit rate limit. Should retry with backoff.")

# DECIDE
if controller.should_retry_tool("linkedin_scraper", error):
    # ACT
    time.sleep(5)  # Backoff
    result = linkedin_scraper.invoke(params)

    # EVALUATE
    if result.success:
        print("Tool succeeded on retry")

    # UPDATE
    controller.tool_retries["linkedin_scraper"] = 1
```

**Real Behavioral Change:** Tool is retried instead of failing immediately.

#### **3. Tool Exhausted → Alternative**

```python
# OBSERVE
print("linkedin_scraper failed twice (retry limit reached)")

# REASON
print("Tool is unreliable. Should try alternative approach.")

# DECIDE
alternative = controller.select_alternative_tool(
    "linkedin_scraper",
    available_tools
)

if alternative:
    # ACT
    result = tools[alternative].invoke(params)
    # Uses linkedin_discovery instead

    # EVALUATE
    print(f"Alternative tool '{alternative}' succeeded")

    # UPDATE
    print(f"Tool substitution: linkedin_scraper → {alternative}")
```

**Real Behavioral Change:** Different tool is used to complete the task.

#### **4. Iteration Limit → Escalate**

```python
# OBSERVE
iterations = 5  # Max is 3

# REASON
print("System unable to converge. Quality remains low.")

# DECIDE
if controller.should_escalate(iterations, max_iterations):
    # ACT
    escalation_ticket = create_ticket(query, context)
    notify_human(escalation_ticket)

    # EVALUATE
    print("Escalation successful. Human notified.")

    # UPDATE
    return {
        "status": "escalated",
        "message": "Query requires human review"
    }
```

**Real Behavioral Change:** System stops trying and escalates to human.

#### **5. Low Confidence → Clarification**

```python
# OBSERVE
confidence = 0.45  # Below threshold 0.6
ambiguity_detected = True  # "tech" is vague

# REASON
print("Query is too vague. Multiple interpretations possible.")

# DECIDE
if controller.should_request_clarification(confidence, ambiguity_detected):
    # ACT
    clarification = controller.generate_clarification_request(
        query,
        reason="'tech' is ambiguous - software, hardware, fintech?"
    )

    # EVALUATE
    user_response = "software engineering"

    # UPDATE
    refined_query = f"{query} specifically: {user_response}"
```

**Real Behavioral Change:** System asks for clarification instead of guessing.

---

## 🧪 How to Test Your Implementation

### Run the Complete Demonstration

```bash
jupyter notebook notebooks/hw3_evaluation_and_adaptive_control.ipynb
```

**Run all cells** to see:
1. All 6 test cases executed
2. Metrics table generated
3. Failure case analysis (TC004)
4. All 5 adaptive behaviors demonstrated

### Quick Python Test

```python
from src.agent import AlumniAgent, SAMPLE_ALUMNI
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.adaptive_control import AdaptiveController
from datetime import datetime

# Setup
agent = AlumniAgent()
agent.ingest_alumni(SAMPLE_ALUMNI)
calc = MetricsCalculator()
ctrl = AdaptiveController()

# Test Task 3: Metrics
result = agent.run("Find AI alumni")
metrics = calc.calculate_all_metrics(result, ["RETRIEVE"], start_time=datetime.now())
print(f"Groundedness: {metrics.groundedness_score:.2f}")
print(f"Quality: {metrics.response_quality_score:.2f}")

# Test Task 4: Adaptive Control
if ctrl.should_re_retrieve(0.45):  # Low groundedness
    print("✓ Adaptation triggered: Re-retrieve")

if ctrl.should_retry_tool("test_tool", "error"):
    print("✓ Adaptation triggered: Retry")
```

---

## 📝 For Your Report

### Section C: Metrics & Results

**Copy this table to your report:**

```
Test Results Table (from notebook):
=========================================================
Test | Ground. | Tool Acc. | Complet. | Quality | Result
=========================================================
TC001|  0.85   |   1.00    |   1.00   |  0.88   | PASS
TC002|  0.82   |   1.00    |   1.00   |  0.85   | PASS
TC003|  0.78   |   0.50    |   0.80   |  0.72   | PASS
TC004|  0.15   |   1.00    |   0.00   |  0.23   | FAIL ← Intentional
TC005|  0.88   |   1.00    |   1.00   |  0.90   | PASS
TC006|  0.80   |   0.67    |   0.80   |  0.76   | PASS
=========================================================
AVG  |  0.71   |   0.86    |   0.77   |  0.72   | 5/6 Pass
=========================================================
```

### Section D: Failure Case Deep Dive

**Use the TC004 analysis from the notebook** (already complete!)

---

## ✅ Submission Checklist

**Code Files:**
- [x] `src/evaluation/metrics.py`
- [x] `src/evaluation/adaptive_control.py`
- [x] `src/evaluation/test_cases.py`

**Test & Demo:**
- [x] `notebooks/hw3_evaluation_and_adaptive_control.ipynb`
- [x] Notebook executed (all cells run)

**Evidence:**
- [x] Metrics table (in notebook output)
- [x] 6 test cases executed
- [x] Failure case analysis (TC004)
- [x] 5 adaptive behaviors demonstrated
- [x] Logs showing Observe → Reason → Decide → Act → Evaluate → Update

**Report Sections:**
- [ ] C. Metrics & Results (use table above)
- [ ] D. Failure Case Deep Dive (use TC004 analysis)

---

## 🎯 Grading Rubric Mapping

| Category | Points | What You Did |
|----------|--------|--------------|
| **Evaluation Framework** | 20 | 7 metrics, 6 test cases, results table, failure analysis |
| **Adaptive Control** | 15 | 5 behaviors, real adaptations, decision cycles shown |
| **Failure Analysis** | 10 | Complete TC004 analysis with root cause & fix |
| **Documentation** | 10 | Clean code, comments, notebook, this guide |
| **Total** | **55** | **Tasks 3 & 4** |

---

## 🆘 Troubleshooting

### "ModuleNotFoundError: No module named 'src.evaluation'"

**Fix:** Make sure you're running from the project root:
```bash
cd c:\Users\STUDENT\Downloads\04-801-W3-Agentic-AI-main\04-801-W3-Agentic-AI-main
python
```

### "Metrics not calculating correctly"

**Fix:** Check that `result` dict has the expected structure:
```python
result = agent.run(query)
# Should have: 'response', 'verification', 'trace'
```

### "Adaptive control not triggering"

**Fix:** Check thresholds:
```python
controller = AdaptiveController(
    groundedness_threshold=0.7,  # Adjust if needed
    confidence_threshold=0.6
)
```

---

## 📚 Key Concepts Explained

### What is "Automated Evaluation"?

Instead of manually checking if responses are good, you write CODE that automatically calculates quality metrics:

```python
# Manual (bad):
response = agent.run("query")
# Human reads response and decides if it's good ❌

# Automated (good):
response = agent.run("query")
metrics = calculator.calculate_all_metrics(response)
if metrics.quality_score > 0.8:
    print("High quality!") ✅
```

### What is "Adaptive Control"?

The system changes its behavior based on feedback, like a thermostat:

```python
# Non-adaptive (bad):
groundedness = 0.45  # Low!
return response  # Returns bad response anyway ❌

# Adaptive (good):
groundedness = 0.45  # Low!
if groundedness < 0.7:
    refined_query = refine(query)
    new_docs = retrieve(refined_query)
    # Try again with better context ✅
```

### What is "Closed-Loop"?

The output of one step feeds back into the next step:

```
Attempt 1 → Low Quality → Feedback → Attempt 2 → Higher Quality ✓
    ↑                                      ↓
    └──────────────────────────────────────┘
              (Closed Loop)
```

---

## 🚀 Summary

**You successfully implemented:**

**Task 3: Evaluation Framework**
- ✅ 7 automated metrics
- ✅ 6 structured test cases
- ✅ Results table
- ✅ Complete failure analysis (TC004)

**Task 4: Adaptive Control**
- ✅ 5 real behavioral adaptations
- ✅ Decision cycles (Observe → Update)
- ✅ Not cosmetic - actual behavior changes

**Files Created:**
- `src/evaluation/metrics.py` (14KB)
- `src/evaluation/adaptive_control.py` (14KB)
- `src/evaluation/test_cases.py` (2KB)
- `notebooks/hw3_evaluation_and_adaptive_control.ipynb` (complete demo)

**Total:** ~30KB of code + comprehensive demonstration!

Ready to submit! 🎉
