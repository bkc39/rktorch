"""The review bot's prompt and context: full, incremental, none or held, the
patch since the last review, the stack the pull request belongs to, the
earlier threads, and master's review rules; see claude-code-review.yml.

    python3 .github/scripts/review_context.py OUT_DIR
"""
import datetime
import json
import os
import secrets
import subprocess
import sys
import urllib.parse

PROMPT = ".github/claude-review-prompt.md"

RECORD = "reviewed-{pr}"

WORKFLOW = ".github/workflows/claude-code-review.yml"

RULES = "## Code Review Rules"

ACTION_CONFIG = [".claude", ".mcp.json", ".claude.json", ".gitmodules",
                 ".ripgreprc", "CLAUDE.md", "CLAUDE.local.md", ".husky"]

LOOKUP_ERRORS = (subprocess.CalledProcessError, KeyError, TypeError,
                 ValueError)

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


def graphql(query, **variables):
    args = ["gh", "api", "graphql", "-f", f"query={query}"]
    for key, value in variables.items():
        if value is not None:
            args += ["-F" if isinstance(value, int) else "-f",
                     f"{key}={value}"]
    return json.loads(run(*args))["data"]


def pull_request(query, repo, pr, **variables):
    owner, name = repo.split("/")
    return graphql(query, owner=owner, name=name, pr=pr,
                   **variables)["repository"]["pullRequest"]


def pulls(repo, **query):
    query = urllib.parse.urlencode({"state": "open", "per_page": 100, **query})
    return json.loads(run("gh", "api", f"repos/{repo}/pulls?{query}"))


def last_reviewed(repo, pr):
    rows = run("gh", "api", "--paginate",
               f"repos/{repo}/actions/artifacts?per_page=100"
               f"&name={RECORD.format(pr=pr)}",
               "--jq", ".artifacts[] | [.workflow_run.id, "
               ".workflow_run.head_sha] | @tsv").split()
    for run_id, head in sorted(zip(map(int, rows[0::2]), rows[1::2]),
                               reverse=True):
        recorder = json.loads(run("gh", "api",
                                  f"repos/{repo}/actions/runs/{run_id}"))
        if recorder["path"] == WORKFLOW and recorder["event"] == "pull_request":
            return head, recorder["created_at"]
    return None, None


def merged_beyond(merged_head, merged_at, after, default):
    since = datetime.datetime.fromisoformat(
        merged_at.replace("Z", "+00:00")).timestamp()
    for line in run("git", "log", "--merges", "--format=%ct %P", after,
                    "--not", f"origin/{default}").splitlines():
        committed, _, *others = line.split()
        if int(committed) < since:
            continue
        for parent in others:
            if not any(succeeds("git", "merge-base", "--is-ancestor", parent,
                                known) for known in
                       (merged_head, f"origin/{default}")):
                return True
    return False


def lower_layer_merged(repo, branch, retargeted_at, after, default):
    owner = repo.split("/")[0]
    earlier = [p for p in pulls(repo, state="all", head=f"{owner}:{branch}")
               if p["created_at"] < retargeted_at]
    layer = max(earlier, key=lambda p: p["created_at"], default=None)
    return bool(layer and layer["merged_at"]
                and layer["merged_at"] <= retargeted_at
                and not merged_beyond(layer["head"]["sha"], layer["merged_at"],
                                      after, default))


def unmerged_bases_left(repo, pr, default, since, after):
    events = pull_request(RETARGETS, repo, pr)["timelineItems"]["nodes"]
    return sorted({e["previousRefName"] for e in events
                   if e["createdAt"] >= since
                   and e["previousRefName"] != default
                   and not lower_layer_merged(repo, e["previousRefName"],
                                              e["createdAt"], after,
                                              default)})


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


def config_from_base(base, default):
    return base != default and not succeeds(
        "git", "diff", "--quiet", f"origin/{default}...origin/{base}", "--",
        *ACTION_CONFIG)


def layer(pr):
    return {"number": pr["number"], "title": pr["title"],
            "state": pr["state"].lower(),
            "base": pr.get("baseRefName") or pr["base"]["ref"]}


def native_stack(repo, pr):
    stack = pull_request(STACK, repo, pr)["stack"]
    if not stack:
        return None
    entries = sorted(stack["entries"]["nodes"], key=lambda e: e["position"])
    return [layer(e["pullRequest"]) for e in entries if e["pullRequest"]]


def branch_chain(pr, repo, default, find):
    owner, head = repo.split("/")[0], pr["head"]["ref"]
    if (pr["head"].get("repo") or {}).get("full_name") != repo:
        return [layer(pr)]
    below, branch, seen = [], pr["base"]["ref"], {head, default}
    while branch not in seen:
        seen.add(branch)
        lower = find(head=f"{owner}:{branch}")
        if not lower:
            break
        below.insert(0, lower[0])
        branch = lower[0]["base"]["ref"]
    above, frontier = [], [] if head == default else [head]
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
            pr, repo, default, lambda **query: pulls(repo, **query)))]
    for source, lookup in lookups:
        try:
            layers = lookup()
        except LOOKUP_ERRORS as e:
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
            "checkout; the layers above build on this one. Each layer is "
            "reviewed on its own pull request. Within the scope above, the "
            "rule for stacked pull requests in `review-rules.md` says what "
            "to leave to the other layers.\n\n")


def master_agents(default):
    return run("git", "show", f"origin/{default}:AGENTS.md", check=False)


def review_rules(default, agents):
    section = agents.partition(f"\n{RULES}\n")[2]
    if not section:
        return f"{default}'s AGENTS.md has no `{RULES}` section yet.\n"
    return f"{RULES}\n" + section.split("\n## ", 1)[0].rstrip() + "\n"


def all_comments(thread):
    comments = thread["comments"]
    nodes = list(comments["nodes"])
    while comments["pageInfo"]["hasNextPage"]:
        comments = graphql(MORE_COMMENTS, id=thread["id"],
                           after=comments["pageInfo"]["endCursor"])["node"][
                               "comments"]
        nodes += comments["nodes"]
    return nodes


def threads(repo, pr):
    nodes, cursor = [], None
    while True:
        page = pull_request(THREADS, repo, pr, after=cursor)["reviewThreads"]
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


def write(out_dir, name, text):
    with open(os.path.join(out_dir, name), "w") as f:
        f.write(text)


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ
    repo, number, after = env["REPO"], int(env["PR"]), env["AFTER"]
    default = env["DEFAULT_BRANCH"]
    pr = json.loads(run("gh", "api", f"repos/{repo}/pulls/{number}"))
    base = pr["base"]["ref"]
    exclude = sorted({f"origin/{base}", f"origin/{default}"})
    before, reviewed_at = last_reviewed(repo, number)
    before = before or ""
    mode = review_mode(before, after)
    reason = ("no review has run on this pull request yet, or its history "
              "was rewritten since")
    if mode == "incremental":
        try:
            left = unmerged_bases_left(repo, number, default, reviewed_at,
                                       after)
        except LOOKUP_ERRORS as e:
            print(f"::warning::retarget lookup failed: {e}")
            left, mode, reason = [], "full", "the retarget lookup failed"
        if left:
            mode, reason = "full", (
                "since the last review this pull request moved off "
                f"{', '.join(f'`{b}`' for b in left)}, whose pull request did "
                "not merge, so commits reviewed only there are now part of "
                "this one")
    if mode == "incremental":
        patch = changes(before, after, exclude)
        if patch.strip():
            write(out_dir, "changes.patch", patch)
        else:
            mode = "none"
    if mode != "none" and config_from_base(base, default):
        mode = "held"
        print(f"::warning::not reviewed: the action restores its "
              f"configuration ({', '.join(ACTION_CONFIG)}) from `{base}`, "
              f"which changes it from {default}'s; this layer is reviewed "
              "once that change lands")
    summary = {
        "full": f"Full review: {reason}. Review the whole diff against "
                f"`{base}` (`gh pr diff`).",
        "incremental": f"Incremental review of {before[:7]}..{after[:7]}: "
                       f"{before[:7]} is the head the last review covered, "
                       "and the rest of the pull request was reviewed then. "
                       "Review only `changes.patch`: the commits since then "
                       f"that are not on {' or '.join(exclude)}, in order, "
                       "with what their merge commits change.",
        "none": "Nothing to review: the commits since the last review add "
                "nothing of this pull request's own.",
        "held": f"Held: `{base}` changes the configuration the action "
                "restores from it.",
    }[mode]
    source, layers = "no stack", []
    if mode in ("full", "incremental"):
        source, layers = stack_layers(repo, pr, default)
        write(out_dir, "threads.md", threads_markdown(threads(repo, number)))
        agents = master_agents(default)
        write(out_dir, "AGENTS.md", agents)
        write(out_dir, "review-rules.md", review_rules(default, agents))
    write(out_dir, "context.md",
          f"# Review context\n\n{summary}\n\n"
          + stack_markdown(layers, number, base)
          + "`threads.md` lists every earlier review thread on this pull "
          f"request with its replies; `AGENTS.md` is {default}'s, the "
          "conventions this pull request is held to, and `review-rules.md` "
          "its Code Review Rules section.\n")
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
