Hello @all, 
Here are some updates from what I was working on on Task 2.
🇧🇹 Bhutan Assistant — Classifier Update

Original approach: one Gemini API call classifying all 4 dimensions 
(scope, safety, request type, service) in a single prompt. Simple but 
no reasoning chain.

New approach: task decomposition, each dimension gets its own focused 
prompt with chain-of-thought + 3 few-shot examples, with state passed 
forward between steps. Also benchmarked BART (facebook/bart-large-mnli), 
a free local HuggingFace model, as a drop-in replacement for any step.

Built a 36-question labeled test set (English, rural phrasing, edge 
cases included) and benchmarked 3 configs:

Results:
              in_scope  safety  req_type  service  OVERALL
BART only       100%    61.1%    58.3%    36.1%    63.9%
Gemini 1-call   91.7%   97.2%   69.4%    83.3%    85.4%
Gemini CoT      100%    88.9%   100%     91.7%    95.1%  ✅

Takeaway: Gemini sequential (CoT) wins at 95.1%. BART is reliable only 
for scope (100%) — useful as a free first gate to block irrelevant 
questions before spending any API budget. Safety is the weakest 
dimension across all models.

Best practical config: BART for scope → Gemini CoT for steps 2-4.
Cost: ~$0.34 per 1,000 in-scope questions (Gemini 2.5 Flash).

And if anyone want to contribute or follow up on this:
* Improve prompts → edit the 4 step methods in GeminiClassifier
* Improve BART → edit the hypothesis dicts (_SCOPE_H, _SAFETY_H, etc.)
* Expand test set → add rows to bhutan_classifier_test_set.csv and 
  re-run the benchmark

Happy to hear your comments on this and please also feel free to verify the test set created.
Tashi Delek 🙏

Quick Usage

# Load the pipeline (change mode as needed)
registry = ClassifierRegistry()
pipeline = SequentialPipeline(registry)

# Run a question
state = pipeline.run("What papers I need for timber permit?")

# See results
print(state.in_scope.label)      # in_scope
print(state.safety.label)        # safe
print(state.request_type.label)  # document_inquiry
print(state.service.label)       # permits
print(state.final_answer)        # bot response

# Switch mode (bart / gemini / hybrid)
STEP_CONFIG.update({
    "scope":        "bart",
    "safety":       "gemini",
    "request_type": "gemini",
    "service":      "gemini",
})