"""The review bot's prompt and context: full, incremental or none, the patch
since the last review, the stack the pull request belongs to, and the earlier
threads; see claude-code-review.yml.

    python3 .github/scripts/review_context.py OUT_DIR
"""
import json
import os
import secrets
import subprocess
import sys
import urllib.parse

PROMPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                      "claude-review-prompt.md")

RECORD_STEP = "Record review"

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

STACK = """
query($owner: String!, $name: String!, $pr: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $pr) {
      stack {
        entries(first: 100) {
          nodes { position pullRequest { number title state baseRefName } }
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


def recorded(repo, run_id):
    jobs = json.loads(run(
        "gh", "api", f"repos/{repo}/actions/runs/{run_id}/jobs"))["jobs"]
    return any(step["name"] == RECORD_STEP and step["conclusion"] == "success"
               for job in jobs for step in job.get("steps", []))


def last_reviewed(repo, workflow, branch, pr, run_id):
    runs = json.loads(run(
        "gh", "api",
        f"repos/{repo}/actions/workflows/{workflow}/runs"
        f"?branch={urllib.parse.quote(branch, safe='')}"
        "&event=pull_request&status=success&per_page=50"))["workflow_runs"]
    for r in runs:
        if (str(r["id"]) != run_id
                and any(p["number"] == pr for p in r["pull_requests"])
                and recorded(repo, r["id"])):
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


def tree(rev):
    return run("git", "rev-parse", f"{rev}^{{tree}}").strip()


def already_merged(branch, other):
    return tree(other) in run("git", "log", "--format=%T", branch).split()


def merge_patch(sha, parents):
    first, others = parents[0], parents[1:]
    if all(already_merged(first, other) for other in others):
        return ("Everything this merge brings in was already on the branch; "
                "what it changes beyond its first parent",
                run("git", "diff", first, sha))
    return ("Conflict resolution in this merge",
            run("git", "show", "--remerge-diff", "--format=", sha))


def changes(before, after, exclude):
    parts = []
    for line in run("git", "rev-list", "--reverse", "--topo-order", "--parents",
                    f"{before}..{after}", "--not", *exclude).splitlines():
        sha, *parents = line.split()
        if len(parents) < 2:
            parts.append(run("git", "show", "--format=fuller", "--stat",
                             "--patch", sha))
            continue
        label, patch = merge_patch(sha, parents)
        if patch.strip():
            parts.append(run("git", "show", "--no-patch", "--format=fuller", sha)
                         + f"\n{label}:\n\n" + patch)
    return "\n".join(parts)


def layer(pr):
    return {"number": pr["number"], "title": pr["title"],
            "state": pr["state"].lower(),
            "base": pr.get("baseRefName") or pr["base"]["ref"]}


def native_stack(repo, number):
    owner, name = repo.split("/")
    result = subprocess.run(
        ["gh", "api", "graphql", "-f", f"query={STACK}", "-f", f"owner={owner}",
         "-f", f"name={name}", "-F", f"pr={number}"],
        capture_output=True, text=True)
    if result.returncode:
        return None
    stack = json.loads(result.stdout)["data"]["repository"]["pullRequest"][
        "stack"]
    if not stack:
        return None
    entries = sorted(stack["entries"]["nodes"], key=lambda e: e["position"])
    return [layer(e["pullRequest"]) for e in entries if e["pullRequest"]]


def open_pulls(repo, **query):
    query = urllib.parse.urlencode({"state": "open", "per_page": 100, **query})
    return json.loads(run("gh", "api", f"repos/{repo}/pulls?{query}"))


def branch_chain(pr, owner, default, pulls):
    below, branch, seen = [], pr["base"]["ref"], {pr["head"]["ref"]}
    while branch != default and branch not in seen:
        seen.add(branch)
        lower = pulls(head=f"{owner}:{branch}")
        if not lower:
            break
        below.insert(0, lower[0])
        branch = lower[0]["base"]["ref"]
    above, frontier = [], [pr["head"]["ref"]]
    while frontier:
        for upper in pulls(base=frontier.pop(0)):
            if upper["head"]["ref"] not in seen:
                seen.add(upper["head"]["ref"])
                above.append(upper)
                frontier.append(upper["head"]["ref"])
    return [layer(p) for p in below + [pr] + above]


def stack_markdown(layers, number, base):
    if len(layers) < 2:
        return ""
    lines = []
    for i, entry in enumerate(layers, 1):
        notes = [note for note, applies in
                 [("this pull request", entry["number"] == number),
                  (entry["state"], entry["state"] != "open")] if applies]
        suffix = f" ({', '.join(notes)})" if notes else ""
        lines.append(f"{i}. #{entry['number']} {entry['title']}{suffix}; "
                     f"base `{entry['base']}`")
    return ("## Stack\n\n"
            "This pull request is one layer of a stack of pull requests, "
            "listed from the bottom up:\n\n" + "\n".join(lines) + "\n\n"
            f"Review only this layer: its own diff against `{base}`. The "
            f"layers below it are already on `{base}` and in the checkout; "
            "the layers above it build on it. Each layer is reviewed on its "
            "own pull request, so do not flag code that lives in another "
            "layer, and do not ask for anything a layer below already "
            "provides.\n\n")


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
    repo, number, after = env["REPO"], int(env["PR"]), env["AFTER"]
    default = env["DEFAULT_BRANCH"]
    pr = json.loads(run("gh", "api", f"repos/{repo}/pulls/{number}"))
    base = pr["base"]["ref"]
    exclude = sorted({f"origin/{base}", f"origin/{default}"})
    workflow = env["GITHUB_WORKFLOW_REF"].split("@")[0].rsplit("/", 1)[-1]
    before = last_reviewed(repo, workflow, env["HEAD_REF"], number,
                           env["GITHUB_RUN_ID"]) or ""
    mode = review_mode(before, after)
    if mode == "incremental":
        patch = changes(before, after, exclude)
        if patch.strip():
            with open(os.path.join(out_dir, "changes.patch"), "w") as f:
                f.write(patch)
        else:
            mode = "none"
    layers = (native_stack(repo, number)
              or branch_chain(pr, repo.split("/")[0], default,
                              lambda **query: open_pulls(repo, **query)))
    with open(os.path.join(out_dir, "threads.md"), "w") as f:
        f.write(threads_markdown(threads(repo, number)))
    summary = {
        "full": "Full review: no review has run on this pull request yet, "
                "or its history was rewritten since. Review the whole diff "
                f"against `{base}` (`gh pr diff`).",
        "incremental": f"Incremental review of {before[:7]}..{after[:7]}: "
                       f"{before[:7]} is the head the last review "
                       "covered, and the rest of the pull request was "
                       "reviewed then. Review only `changes.patch`: the "
                       "commits since then that are not on "
                       f"{' or '.join(exclude)}, in order, with what their "
                       "merge commits change.",
        "none": "Nothing to review: the commits since the last review add "
                "nothing of this pull request's own.",
    }[mode]
    with open(os.path.join(out_dir, "context.md"), "w") as f:
        f.write(f"# Review context\n\n{summary}\n\n"
                + stack_markdown(layers, number, base)
                + "`threads.md` lists every earlier review thread on this "
                "pull request with its replies.\n")
    with open(PROMPT) as f:
        prompt = f"Review pull request #{number} in {repo}.\n\n{f.read()}"
    delimiter = f"PROMPT_{secrets.token_hex(16)}"
    with open(env["GITHUB_OUTPUT"], "a") as f:
        f.write(f"mode={mode}\nprompt<<{delimiter}\n{prompt}\n{delimiter}\n")
    stack = " ".join(f"#{entry['number']}" for entry in layers)
    print(f"review mode: {mode} (last reviewed head: {before[:7] or 'none'}; "
          f"base: {base}, event base: {env['BASE_REF']}; stack: {stack})")


if __name__ == "__main__":
    main()
