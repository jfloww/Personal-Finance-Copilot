# Development history

One file per working day, named `YYYY-MM-DD.md`.

This is not a changelog. `git log` already records what changed, and the commit
messages already say why. What a changelog cannot hold is the reasoning that
did not survive: the option considered and rejected, the assumption that turned
out wrong, the bug found by checking rather than by trusting.

So each entry answers three questions:

- **What shipped**, with commit ranges so a claim can be checked against the code.
- **What was wrong**, including things this project got wrong about itself. A
  history that only records successes is a marketing document.
- **What was decided**, and what it costs if the decision was wrong.

Two rules, both inherited from the repository's own standard that documentation
never overstates:

**Numbers are verified, not remembered.** A test count in an entry was read off
a run on that day. If a figure cannot be checked, it does not appear.

**Nothing is claimed as deployed unless it is.** The demo is deployed. The
import path is not.
