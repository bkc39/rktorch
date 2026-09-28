"""The review bot's context: full, incremental or none, the patch since the
last successful review, and the earlier threads; see claude-code-review.yml.

    python3 .github/scripts/review_context.py OUT_DIR
"""
import json
import os
import subprocess
import sys
import urllib.parse

THREADS = """
query($owner: String!, $name: String!, $pr: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $pr) {
      reviewThreads(first: 50, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id isResolved isOutdated path line originalLine
          comments(first: 100) {
            pageInfo { hasNextPage endCursor }
            nodes { author { login } body }
          }
        }
      }
    }
  }
}
"""

MORE_COMMENTS = """
query($id: ID!, $after: String) {
  node(id: $id) {
    ... on PullRequestReviewThread {
      comments(first: 100, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes { author { login } body }
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


def last_reviewed(repo, workflow, branch, pr, run_id):
    runs = json.loads(run(
        "gh", "api",
        f"repos/{repo}/actions/workflows/{workflow}/runs"
        f"?branch={urllib.parse.quote(branch, safe='')}"
        "&event=pull_request&status=success&per_page=50"))["workflow_runs"]
    for r in runs:
        if str(r["id"]) != run_id and any(
                p["number"] == pr for p in r["pull_requests"]):
            return r["head_sha"]
    return None


def review_mode(before, after):
    if not before:
        return "full"
    if not succeeds("git", "cat-file", "-e", f"{before}^{{commit}}"):
        return "full"
    if not succeeds("git", "merge-base", "--is-ancestor", before, after):
        return "full"
    return "incremental"


def changes(before, after, base):
    parts = []
    for line in run("git", "rev-list", "--reverse", "--topo-order", "--parents",
                    f"{before}..{after}", "--not", base).splitlines():
        sha, *parents = line.split()
        if len(parents) < 2:
            parts.append(run("git", "show", "--format=fuller", "--stat",
                             "--patch", sha))
            continue
        resolution = run("git", "show", "--remerge-diff", "--format=", sha)
        if resolution.strip():
            parts.append(run("git", "show", "--no-patch", "--format=fuller", sha)
                         + "\nConflict resolution in this merge:\n\n"
                         + resolution)
    return "\n".join(parts)


def all_comments(thread):
    comments = thread["comments"]
    nodes = list(comments["nodes"])
    while comments["pageInfo"]["hasNextPage"]:
        comments = json.loads(run(
            "gh", "api", "graphql", "-f", f"query={MORE_COMMENTS}",
            "-f", f"id={thread['id']}",
            "-f", f"after={comments['pageInfo']['endCursor']}"))["data"][
                "node"]["comments"]
        nodes += comments["nodes"]
    return nodes


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
        for thread in page["nodes"]:
            thread["comments"] = all_comments(thread)
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
        for c in t["comments"]:
            who = (c["author"] or {}).get("login", "ghost")
            out.append(f"**{who}:**\n\n{c['body'].strip()}\n")
    return "\n".join(out)


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ
    pr, after = int(env["PR"]), env["AFTER"]
    base = f"origin/{env['BASE_REF']}"
    workflow = env["GITHUB_WORKFLOW_REF"].split("@")[0].rsplit("/", 1)[-1]
    before = last_reviewed(env["REPO"], workflow, env["HEAD_REF"], pr,
                           env["GITHUB_RUN_ID"]) or ""
    mode = review_mode(before, after)
    if mode == "incremental":
        patch = changes(before, after, base)
        if patch.strip():
            with open(os.path.join(out_dir, "changes.patch"), "w") as f:
                f.write(patch)
        else:
            mode = "none"
    with open(os.path.join(out_dir, "threads.md"), "w") as f:
        f.write(threads_markdown(threads(env["REPO"], pr)))
    summary = {
        "full": "Full review: no earlier review of this pull request "
                "succeeded, or its history was rewritten since. Review the "
                "whole diff (`gh pr diff`).",
        "incremental": f"Incremental review of {before[:7]}..{after[:7]}: "
                       f"{before[:7]} is the head the last successful review "
                       "covered, and the rest of the pull request was "
                       "reviewed then. Review only `changes.patch`: the "
                       "commits since then that are not on the base branch "
                       f"({base}), in order, with the conflict resolutions "
                       "of their merge commits.",
        "none": "Nothing to review: the commits since the last review add "
                "nothing of this pull request's own.",
    }[mode]
    with open(os.path.join(out_dir, "context.md"), "w") as f:
        f.write(f"# Review context\n\n{summary}\n\n"
                "`threads.md` lists every earlier review thread on this "
                "pull request with its replies.\n")
    with open(env["GITHUB_OUTPUT"], "a") as f:
        f.write(f"mode={mode}\n")
    print(f"review mode: {mode} (last reviewed head: {before[:7] or 'none'})")


if __name__ == "__main__":
    main()
