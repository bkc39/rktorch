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

PROMPT = ".github/claude-review-prompt.md"

RECORD = "reviewed-{pr}"

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

RETARGETS = """
query($owner: String!, $name: String!, $pr: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $pr) {
      timelineItems(itemTypes: [BASE_REF_CHANGED_EVENT], last: 100) {
        nodes { ... on BaseRefChangedEvent { createdAt previousRefName } }
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


def graphql(query, repo, pr):
    owner, name = repo.split("/")
    return json.loads(run(
        "gh", "api", "graphql", "-f", f"query={query}", "-f", f"owner={owner}",
        "-f", f"name={name}", "-F", f"pr={pr}"))["data"]["repository"][
            "pullRequest"]


def pulls(repo, **query):
    query = urllib.parse.urlencode({"state": "open", "per_page": 100, **query})
    return json.loads(run("gh", "api", f"repos/{repo}/pulls?{query}"))


def last_reviewed(repo, pr, run_id):
    rows = run("gh", "api", "--paginate",
               f"repos/{repo}/actions/artifacts?per_page=100"
               f"&name={RECORD.format(pr=pr)}",
               "--jq", ".artifacts[] | [.created_at, .workflow_run.id, "
               ".workflow_run.head_sha] | @tsv").splitlines()
    reviews = sorted(row.split("\t") for row in rows)
    for created, run_id_, head in reversed(reviews):
        if run_id_ != run_id:
            return head, created
    return None, None


def unmerged_bases_left(repo, pr, default, since):
    owner = repo.split("/")[0]
    left = {event["previousRefName"]
            for event in graphql(RETARGETS, repo, pr)["timelineItems"]["nodes"]
            if event["createdAt"] > since
            and event["previousRefName"] != default}
    return sorted(base for base in left
                  if not any(p["merged_at"] for p in
                             pulls(repo, state="all", head=f"{owner}:{base}")))


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
    return tree(other) in run("git", "log", "--format=%T", branch,
                              "--not", other).split()


def merge_patch(sha, parents):
    first, others = parents[0], parents[1:]
    if all(already_merged(first, other) for other in others):
        return ("What this merge changes beyond its first parent",
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


def native_stack(repo, pr):
    stack = graphql(STACK, repo, pr)["stack"]
    if not stack:
        return None
    entries = sorted(stack["entries"]["nodes"], key=lambda e: e["position"])
    return [layer(e["pullRequest"]) for e in entries if e["pullRequest"]]


def branch_chain(pr, owner, default, find):
    below, branch, seen = [], pr["base"]["ref"], {pr["head"]["ref"], default}
    while branch not in seen:
        seen.add(branch)
        lower = find(head=f"{owner}:{branch}")
        if not lower:
            break
        below.insert(0, lower[0])
        branch = lower[0]["base"]["ref"]
    above, frontier = [], [pr["head"]["ref"]]
    while frontier:
        for upper in find(base=frontier.pop(0)):
            if upper["head"]["ref"] not in seen:
                seen.add(upper["head"]["ref"])
                above.append(upper)
                frontier.append(upper["head"]["ref"])
    return [layer(p) for p in below + [pr] + above]


def stack_layers(repo, pr, default):
    lookups = [
        ("native stack", lambda: native_stack(repo, pr["number"])),
        ("branch chain", lambda: branch_chain(
            pr, repo.split("/")[0], default,
            lambda **query: pulls(repo, **query)))]
    for source, lookup in lookups:
        try:
            layers = lookup()
        except (subprocess.CalledProcessError, KeyError, TypeError,
                ValueError) as e:
            print(f"::warning::{source} lookup failed: {e}")
            continue
        if layers:
            return source, layers
    return "no stack", []


def stack_markdown(layers, number, base):
    if len(layers) < 2:
        return ""
    lines = []
    for entry in layers:
        notes = [note for note, applies in
                 [("this pull request", entry["number"] == number),
                  (entry["state"], entry["state"] != "open")] if applies]
        suffix = f" ({', '.join(notes)})" if notes else ""
        lines.append(f"- #{entry['number']} {entry['title']}{suffix}, "
                     f"on `{entry['base']}`")
    return ("## Stack\n\n"
            "This pull request is one layer of a stack of pull requests, "
            "each on the base branch named, lowest first:\n\n"
            + "\n".join(lines) + "\n\n"
            f"The layers below this one are already on `{base}` and in the "
            "checkout, and are reviewed on their own pull requests; the "
            "layers above build on this one. Within the scope above, do not "
            "flag what another layer's code already did, and do not ask for "
            "anything a layer below provides, but do flag anything this "
            "layer's changes break in another layer's code.\n\n")


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
    before, reviewed_at = last_reviewed(repo, number, env["GITHUB_RUN_ID"])
    before = before or ""
    mode = review_mode(before, after)
    left = (unmerged_bases_left(repo, number, default, reviewed_at)
            if mode == "incremental" else [])
    if left:
        mode = "full"
    if mode == "incremental":
        patch = changes(before, after, exclude)
        if patch.strip():
            with open(os.path.join(out_dir, "changes.patch"), "w") as f:
                f.write(patch)
        else:
            mode = "none"
    source, layers = stack_layers(repo, pr, default)
    with open(os.path.join(out_dir, "threads.md"), "w") as f:
        f.write(threads_markdown(threads(repo, number)))
    full = (f"Full review: since the last review this pull request moved off "
            f"{', '.join(f'`{b}`' for b in left)}, whose pull request did not "
            "merge, so commits reviewed only there are now part of this one."
            if left else
            "Full review: no review has run on this pull request yet, or its "
            "history was rewritten since.")
    summary = {
        "full": f"{full} Review the whole diff against `{base}` "
                "(`gh pr diff`).",
        "incremental": f"Incremental review of {before[:7]}..{after[:7]}: "
                       f"{before[:7]} is the head the last review covered, "
                       "and the rest of the pull request was reviewed then. "
                       "Review only `changes.patch`: the commits since then "
                       f"that are not on {' or '.join(exclude)}, in order, "
                       "with what their merge commits change.",
        "none": "Nothing to review: the commits since the last review add "
                "nothing of this pull request's own.",
    }[mode]
    with open(os.path.join(out_dir, "context.md"), "w") as f:
        f.write(f"# Review context\n\n{summary}\n\n"
                + stack_markdown(layers, number, base)
                + "`threads.md` lists every earlier review thread on this "
                "pull request with its replies.\n")
    prompt = (f"Review pull request #{number} in {repo}.\n\n"
              + run("git", "show", f"origin/{default}:{PROMPT}"))
    delimiter = f"PROMPT_{secrets.token_hex(16)}"
    with open(env["GITHUB_OUTPUT"], "a") as f:
        f.write(f"mode={mode}\nrecord={RECORD.format(pr=number)}\n"
                f"prompt<<{delimiter}\n{prompt}\n{delimiter}\n")
    stack = " ".join(f"#{entry['number']}" for entry in layers)
    print(f"review mode: {mode} (last reviewed head: {before[:7] or 'none'}; "
          f"head: {env['HEAD_REF']}, base: {base}, event base: "
          f"{env['BASE_REF']}; {source}: {stack})")


if __name__ == "__main__":
    main()
