"""Writes what the review bot reads before it reviews a push.

The first review of a pull request (its `opened` event) covers the whole
diff. Every later push is reviewed incrementally: only the commits it adds
that are not already on the base branch, plus the conflict resolutions in
its merge commits. Merging a lower PR of a stack up into this one therefore
does not re-review that PR's changes here, and a push that only merges the
base has nothing to review at all. Every earlier review thread goes into the
context with its replies, so a point already settled is not raised again.

    python3 .github/scripts/review_context.py OUT_DIR

Reads PR, ACTION, BEFORE, AFTER, BASE_REF and REPO from the environment and
writes OUT_DIR/context.md, OUT_DIR/threads.md and, for an incremental review,
OUT_DIR/changes.patch. Sets the step output `mode`: full, incremental, or
none when the push adds nothing of this PR's own.
"""
import json
import os
import subprocess
import sys

THREADS = """
query($owner: String!, $name: String!, $pr: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $pr) {
      reviewThreads(first: 50, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes {
          isResolved isOutdated path line originalLine
          comments(first: 30) { nodes { author { login } body } }
        }
      }
    }
  }
}
"""


def run(*args, check=True):
    return subprocess.run(args, capture_output=True, text=True,
                          check=check).stdout


def succeeds(*args):
    return subprocess.run(args, capture_output=True).returncode == 0


def review_mode(action, before, after):
    if action == "opened" or not before or set(before) == {"0"}:
        return "full"
    if not succeeds("git", "cat-file", "-e", f"{before}^{{commit}}"):
        return "full"
    if not succeeds("git", "merge-base", "--is-ancestor", before, after):
        return "full"
    return "incremental"


def changes(before, after, base):
    span = [f"{before}..{after}", "--not", base]
    parts = []
    for sha in run("git", "rev-list", "--reverse", "--no-merges", *span).split():
        parts.append(run("git", "show", "--format=fuller", "--stat", "--patch",
                         sha))
    for sha in run("git", "rev-list", "--reverse", "--merges", *span).split():
        resolution = run("git", "show", "--remerge-diff", "--format=", sha)
        if resolution.strip():
            parts.append(run("git", "show", "--no-patch", "--format=fuller", sha)
                         + "\nConflict resolution in this merge:\n\n"
                         + resolution)
    return "\n".join(parts)


def threads(repo, pr):
    owner, name = repo.split("/")
    nodes, cursor = [], None
    while True:
        args = ["gh", "api", "graphql", "-f", f"query={THREADS}",
                "-f", f"owner={owner}", "-f", f"name={name}", "-F", f"pr={pr}"]
        if cursor:
            args += ["-f", f"after={cursor}"]
        page = json.loads(run(*args))["data"]["repository"]["pullRequest"][
            "reviewThreads"]
        nodes += page["nodes"]
        if not page["pageInfo"]["hasNextPage"]:
            return nodes
        cursor = page["pageInfo"]["endCursor"]


def threads_markdown(nodes):
    if not nodes:
        return "No earlier review threads.\n"
    out = []
    for t in nodes:
        state = "resolved" if t["isResolved"] else "open"
        if t["isOutdated"]:
            state += ", outdated"
        line = t["line"] or t["originalLine"]
        out.append(f"## {t['path']}:{line} ({state})\n")
        for c in t["comments"]["nodes"]:
            who = (c["author"] or {}).get("login", "ghost")
            body = c["body"].strip()
            if len(body) > 2000:
                body = body[:2000] + " [...]"
            out.append(f"**{who}:**\n\n{body}\n")
    return "\n".join(out)


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ
    before, after = env.get("BEFORE", ""), env["AFTER"]
    base = f"origin/{env['BASE_REF']}"
    mode = review_mode(env["ACTION"], before, after)
    if mode == "incremental":
        patch = changes(before, after, base)
        if patch.strip():
            with open(os.path.join(out_dir, "changes.patch"), "w") as f:
                f.write(patch)
        else:
            mode = "none"
    with open(os.path.join(out_dir, "threads.md"), "w") as f:
        f.write(threads_markdown(threads(env["REPO"], int(env["PR"]))))
    summary = {
        "full": "Full review: this is the pull request's first review, or "
                "its history was rewritten. Review the whole diff "
                "(`gh pr diff`).",
        "incremental": f"Incremental review of the push {before[:7]}.."
                       f"{after[:7]}. The rest of the pull request was "
                       "reviewed before. Review only `changes.patch`: the "
                       "commits this push adds that are not on the base "
                       f"branch ({base}), plus the conflict resolutions in "
                       "its merge commits.",
        "none": "Nothing to review: the push adds no commits of this pull "
                "request's own.",
    }[mode]
    with open(os.path.join(out_dir, "context.md"), "w") as f:
        f.write(f"# Review context\n\n{summary}\n\n"
                "`threads.md` lists every earlier review thread on this "
                "pull request with its replies.\n")
    with open(env["GITHUB_OUTPUT"], "a") as f:
        f.write(f"mode={mode}\n")
    print(f"review mode: {mode}")


if __name__ == "__main__":
    main()
