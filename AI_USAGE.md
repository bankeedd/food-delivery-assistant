# AI Usage Declaration
**Project:** Food Delivery Assistant · **Team:** self1sh · **Student:** Amantaev Islam

## Tools used
- **Claude (Anthropic)**, a conversational AI assistant.

## What the AI was used for
| Part of the project | How AI was used |
|---|---|
| Project idea and design | Suggested topic #20 from the course list and the A\* → Bayes → MDP → A\* pipeline |
| Core code (`delivery_assistant.py`) | Generated the A\*, Bayes and MDP implementations |
| Web UI (`app.py`, `static/index.html`) | Generated the server and the interface |
| Tests (`test_delivery.py`) | Generated the unit tests |
| Documentation | Drafted the README, the technical report and the slides |
| Demo video (`demo/`) | Demo video: recorded automatically by Claude Code (Playwright) from the real running app, with generated captions; the narration is a synthetic voice (edge-tts, en-US-AndrewNeural) |

## What I did myself
- Ran the program and the tests on my own machine (macOS) and checked the outputs.
- Recomputed the key numbers by hand: the Bayes posterior 0.18 / 0.30 = 0.60, and the Bellman values, for example Q(Blocked, Continue) = −25 + 0.9·(0.3·(−10) + 0.7·(−14)) = −36.52.
- Studied every algorithm in the project so I can explain it and answer questions at the defense.
- Prepared and recorded the demo video and the presentation.
- *[Add anything else you changed yourself: map, numbers, design, etc.]*

## Verification
All formulas shown in the UI and the README are checked against hand calculations and covered by 20 automated tests. All probabilities, rewards and maps are hypothetical toy values, as the assignment allows.

## Statement
I understand the code submitted in this repository and I can explain how each algorithm works and why it was chosen.
