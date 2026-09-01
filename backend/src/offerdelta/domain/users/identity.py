"""Email identity.

Addresses are matched after `.strip().lower()`, so " Victim@X.Test " and
"victim@x.test" reach the same row. One function owns that rule so every
caller that keys on an address - `UserRepository`'s lookup, and the login
rate limiter's bucket - uses exactly the same one.

Two independent normalisations is how this broke the first time: the API
layer keyed the limiter on `body.email.lower()` while `UserRepository`
matched on `.strip().lower()`. A leading space authenticated against the
victim's real row (the repository's rule strips it) while opening a brand
new limiter bucket (the API's rule did not), so an attacker who varied
whitespace got unlimited attempts against one address. A single shared
function is what makes that drift impossible to reintroduce - there is only
one place either caller can get the rule from.
"""

from __future__ import annotations


def normalise_email(raw: str) -> str:
    """Collapse case and surrounding whitespace so one address is one key."""
    return raw.strip().lower()
