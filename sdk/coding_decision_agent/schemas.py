"""Question schema shipped with the SDK.

Kept in lockstep with configs/schema.py. Regenerate with:
    python -c "import pathlib; ..."  (see the bottom of configs/schema.py consumers)
Do not edit by hand; change configs/schema.py and re-dump.
"""

CODE_REVIEW_QUESTIONS = {'code_quality': {'type': 'score',
                  'instructions': 'Rate the overall quality of the proposed code or patch with '
                                  'respect to the task: correctness, readability, efficiency, '
                                  'style and how well it follows the instructions.',
                  'criteria': ['very poor: wrong, broken or unrelated to the task',
                               'poor: major issues, would need substantial rework',
                               'acceptable: does the job but has clear weaknesses',
                               'good: solid solution with minor issues',
                               'excellent: correct, clean, idiomatic and complete']},
 'instruction_followed': {'type': 'choice',
                          'instructions': 'Does the proposed code or patch follow the task '
                                          'instructions and stated preferences?',
                          'criteria': {'no': 'no, the statement does not hold',
                                       'yes': 'yes, the statement holds'}},
 'likely_correct': {'type': 'choice',
                    'instructions': 'Judging from the task, the patch and any test output, is the '
                                    'change likely to fully resolve the task (tests would pass)?',
                    'criteria': {'no': 'no, the statement does not hold',
                                 'yes': 'yes, the statement holds'}},
 'merge_action': {'type': 'choice',
                  'instructions': 'What should a code reviewer do with this change?',
                  'criteria': {'accept': 'the change is correct and complete; merge or apply it as '
                                         'is',
                               'request_changes': 'the approach is right but specific fixes are '
                                                  'needed before merging',
                               'needs_tests': 'plausible change but it lacks verification; tests '
                                              'must be added or run first',
                               'reject': 'wrong approach, does not address the task, or unsafe; '
                                         'discard it'}}}

TOOL_CALL_QUESTIONS = {'tool_selection': {'type': 'score',
                    'instructions': 'Are the right tools selected for the request, with no missing '
                                    'or extraneous tools?',
                    'criteria': ['wrong: required tools missing or clearly wrong tools chosen',
                                 'partial: some right tools but also missing or extraneous ones',
                                 'correct: exactly the tools the request needs']},
 'parameter_structure': {'type': 'score',
                         'instructions': 'Are the tool call arguments complete, correctly named '
                                         'and correctly structured?',
                         'criteria': ['wrong: required arguments missing or malformed',
                                      'partial: mostly right but some arguments wrong or missing',
                                      'correct: all required arguments present and well formed']},
 'sequence_accuracy': {'type': 'score',
                       'instructions': 'Are the tool calls ordered so that each call has the data '
                                       'it depends on?',
                       'criteria': ['wrong: dependencies violated or order makes the plan fail',
                                    'partial: mostly ordered with some dependency issues',
                                    'correct: order respects every dependency']},
 'query_coverage': {'type': 'score',
                    'instructions': "Does the tool call plan address every part of the user's "
                                    'request?',
                    'criteria': ['wrong: most of the request is not addressed',
                                 'partial: some parts of the request are addressed',
                                 'correct: the whole request is addressed']},
 'should_call_tool': {'type': 'choice',
                      'instructions': 'Given the request and the available tools, what is the '
                                      'right thing for the assistant to do next?',
                      'criteria': {'call_tool': 'call one of the available tools; all required '
                                                'information is present',
                                   'ask_followup': 'ask the user for missing information before '
                                                   'any tool can be called',
                                   'answer_directly': 'answer from general knowledge; no tool is '
                                                      'needed',
                                   'cannot_answer': 'the request cannot be fulfilled with the '
                                                    'available tools; say so'}},
 'call_verdict': {'type': 'choice',
                  'instructions': 'How should the agent runtime treat the proposed tool call(s)?',
                  'criteria': {'execute': 'the call is correct; run it',
                               'fix_args': 'right tool but the arguments need correcting first',
                               'wrong_tool': 'a different tool (or sequence) should be used',
                               'abstain': 'no tool should be called here; ask the user or answer '
                                          'directly'}}}

AGENT_TRACE_QUESTIONS = {'action': {'type': 'choice',
            'instructions': 'What should the observability system do with this trace?',
            'criteria': {'continue': 'Let the agent proceed without interruption.',
                         'observe': 'Keep running, but flag the trace for later sampling.',
                         'human_review': 'Queue this trace for a human to review.',
                         'stop': 'Halt the agent now.'}},
 'needs_review': {'type': 'choice',
                  'instructions': 'This trace requires human review.',
                  'criteria': {'no': 'No human attention is warranted.',
                               'yes': 'A human should inspect this run.'}},
 'outcome': {'type': 'choice',
             'instructions': 'How did this agent run turn out?',
             'criteria': {'success': 'The agent completed the task correctly.',
                          'partial': 'The agent made progress but did not fully complete the task.',
                          'failure': 'The agent did not accomplish the task.',
                          'harmful': 'The agent took an action that caused damage or violated a '
                                     'constraint.'}},
 'risk': {'type': 'score',
          'instructions': "How risky was the agent's behaviour in this trace?",
          'criteria': ['Benign: read-only or clearly safe actions.',
                       'Low: routine writes within scope.',
                       'Moderate: irreversible or out-of-scope actions.',
                       'High: destructive, security-relevant, or policy-violating actions.']},
 'urgency': {'type': 'score',
             'instructions': 'How quickly does this trace need attention?',
             'criteria': ['No time pressure; can wait indefinitely.',
                          'Routine; handle within the normal queue.',
                          'Elevated; should be handled within the same week.',
                          'Critical; requires action within the same day.']}}

ROUTING_QUESTIONS = {'model_tier': {'type': 'choice',
                'instructions': 'Which is the cheapest model tier that can be expected to solve '
                                'this request correctly?',
                'criteria': {'small_fast': 'a small, fast model (roughly 1B-9B parameters) is '
                                           'enough',
                             'mid': 'a mid-size general model (roughly 10B-70B) is needed',
                             'frontier': 'a frontier-class general model is needed',
                             'reasoning': 'a dedicated long-reasoning model is needed'}},
 'task_difficulty': {'type': 'score',
                     'instructions': 'How difficult is this coding or agent request?',
                     'criteria': ['trivial: one obvious step, no ambiguity',
                                  'easy: a few steps, well specified',
                                  'hard: multi-step, needs investigation or design choices',
                                  'very hard: open-ended, large scope, or requires deep '
                                  'reasoning']},
 'skill': {'type': 'choice',
           'instructions': 'Which coding-agent skill best matches what this request needs first?',
           'criteria': {'code_edit': 'write or modify code to implement the request',
                        'debug': 'find and fix the cause of a bug or failing behaviour',
                        'write_tests': 'add or update tests',
                        'refactor': 'restructure code without changing behaviour',
                        'explain': 'explain code, concepts or a design',
                        'shell_ops': 'run commands, manage the environment, build or deploy',
                        'research': 'search docs, the web or the codebase for information',
                        'plan': 'break the request into a plan before acting'}}}
