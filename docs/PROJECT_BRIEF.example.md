# Goal
Find the fastest correct way to compute the number of primes below 10,000,000
(pi(10^7)), and print the count together with how long it took.

# Constraints
- Python standard library only (the sandbox has no network and no extra packages).
- Must finish in under ~20 seconds and print a final answer.
- Include a quick correctness check (e.g. assert the known value pi(10^7) = 620,381).

# What to Try (the decision tree — your judgment, encoded)
- Start: a Sieve of Eratosthenes over a boolean list.
- If it is correct but memory-heavy or slow: use a bytearray and skip even numbers.
- If an import fails: rewrite using only the standard library.
- If correct and under the time budget: stop.

# Current Status
No attempts yet. Starting from scratch.
