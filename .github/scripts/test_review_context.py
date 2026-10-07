"""Tests for review_context.py, run from the repository root:

    python3 .github/scripts/test_review_context.py
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import review_context as rc  # noqa: E402

WORKFLOW = ".github/workflows/claude-code-review.yml"
SQUASH_232 = "08a13db5d3372361effe89150b0b267350a79468"
BEFORE_SQUASH_233 = "26ff5892996b1ac24f58ddba02fc3ee29a75b54b"
OURS_MERGE_233 = "826b02c8a30c13e590c426c8a71ecb3afb1ccbe5"
LAST_PUSH_233 = "60177b2a6"
RESOLVED_MERGE_233 = "57f08da37"


def git(*args):
    return subprocess.run(("git",) + args, check=True, capture_output=True,
                          text=True).stdout.strip()


def have(sha):
    return rc.succeeds("git", "cat-file", "-e", f"{sha}^{{commit}}")


class Cascade233(unittest.TestCase):
    """#233 after #232 was squash-merged: 826b02c merged master with -s ours."""

    @classmethod
    def setUpClass(cls):
        if not have(OURS_MERGE_233):
            subprocess.run(["git", "fetch", "-q", "origin", "refs/pull/233/head"],
                           capture_output=True)
        if not have(OURS_MERGE_233):
            raise unittest.SkipTest("the #233 history is not fetchable")

    def test_ours_merge_resolution_is_noise_the_old_patch_carried(self):
        resolution = rc.run("git", "show", "--remerge-diff", "--format=",
                            OURS_MERGE_233)
        self.assertGreater(len(resolution.splitlines()), 100)

    def test_ours_merge_after_the_squash_adds_nothing(self):
        self.assertEqual(rc.tree(OURS_MERGE_233), rc.tree(BEFORE_SQUASH_233))
        self.assertTrue(rc.already_merged(BEFORE_SQUASH_233, SQUASH_232))
        self.assertEqual(
            rc.changes(BEFORE_SQUASH_233, OURS_MERGE_233, [SQUASH_232]), "")

    def test_the_next_commit_is_the_whole_increment(self):
        patch = rc.changes(BEFORE_SQUASH_233, LAST_PUSH_233, [SQUASH_232])
        self.assertIn(f"commit {git('rev-parse', LAST_PUSH_233)}", patch)
        self.assertNotIn(OURS_MERGE_233, patch)

    def test_the_squash_cascade_exposes_nothing(self):
        self.assertEqual(rc.exposed(BEFORE_SQUASH_233,
                                    "93caacd6fae0388893ba9b8562a47107db3edd66",
                                    [SQUASH_232]), [])

    def test_real_conflict_resolutions_are_still_reviewed(self):
        line = git("rev-list", "--parents", "-1", RESOLVED_MERGE_233).split()
        label, patch = rc.merge_patch(line[0], line[1:], [line[2]])
        self.assertEqual(label, "Conflict resolution in this merge")
        self.assertTrue(patch.strip())


class Synthetic(unittest.TestCase):
    """master <- layer1 <- layer2, layer1 squash-merged into master."""

    def setUp(self):
        environ = {k: v for k, v in os.environ.items()
                   if not k.startswith("GIT_")}
        environ.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
        patcher = mock.patch.dict(os.environ, environ, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(directory.name)
        git("init", "-q", "-b", "master")
        git("config", "user.email", "t@example.com")
        git("config", "user.name", "t")
        self.commit("f", "1\n2\n3\n", "A")
        git("checkout", "-q", "-b", "layer1")
        self.commit("f", "one\n2\n3\n", "L1")
        git("checkout", "-q", "-b", "layer2")
        self.commit("g", "layer two\n", "L2")
        self.reviewed = git("rev-parse", "HEAD")
        git("checkout", "-q", "master")
        git("merge", "-q", "--squash", "layer1")
        git("commit", "-q", "-m", "S: layer1 squashed")

    def commit(self, path, text, message):
        with open(path, "w") as f:
            f.write(text)
        git("add", path)
        git("commit", "-q", "-m", message)

    def merge_master_ours(self):
        git("checkout", "-q", "layer2")
        git("merge", "-q", "-s", "ours", "-m", "merge master", "master")
        return git("rev-parse", "HEAD")

    def test_ours_merge_of_the_squash_adds_nothing(self):
        head = self.merge_master_ours()
        self.assertEqual(rc.changes(self.reviewed, head, ["master"]), "")

    def test_ours_merge_that_drops_a_base_change_is_reviewed(self):
        self.commit("h", "on master after the squash\n", "X")
        head = self.merge_master_ours()
        patch = rc.changes(self.reviewed, head, ["master"])
        self.assertIn("Conflict resolution in this merge", patch)
        self.assertIn("on master after the squash", patch)

    def test_ours_merge_that_drops_a_revert_on_master_is_reviewed(self):
        git("checkout", "-q", "layer2")
        git("reset", "-q", "--hard", "layer1")
        self.commit("g", "layer two\n", "L2 again")
        reviewed = git("rev-parse", "HEAD")
        git("checkout", "-q", "master")
        git("reset", "-q", "--hard", "layer1")
        git("revert", "--no-edit", "HEAD")
        self.assertEqual(rc.tree("master"), rc.tree("master~2"))
        head = self.merge_master_ours()
        self.assertFalse(rc.already_merged(reviewed, "master"))
        self.assertIn("Conflict resolution in this merge",
                      rc.changes(reviewed, head, ["master"]))


    def test_merging_master_into_an_upper_layer_adds_nothing(self):
        self.commit("h", "on master\n", "X")
        git("checkout", "-q", "layer2")
        git("merge", "-q", "-m", "merge master", "master")
        head = git("rev-parse", "HEAD")
        self.assertIn("+on master", rc.changes(self.reviewed, head, ["layer1"]))
        self.assertEqual(
            rc.changes(self.reviewed, head, ["layer1", "master"]), "")

    def publish(self, *branches):
        for branch in branches:
            git("update-ref", f"refs/remotes/origin/{branch}", branch)

    def test_a_squashed_lower_layer_exposes_nothing(self):
        self.merge_master_ours()
        self.assertEqual(rc.exposed(self.reviewed, "layer1", ["master"]),
                         [])

    def test_a_lower_layer_that_never_merged_is_exposed(self):
        git("checkout", "-q", "master")
        git("reset", "-q", "--hard", "HEAD~1")
        self.assertEqual(
            rc.exposed(self.reviewed, "layer1", ["master"]),
            [git("rev-parse", "layer1")])

    def test_commits_the_lower_branch_got_after_its_squash_are_exposed(self):
        git("checkout", "-q", "layer1")
        self.commit("f", "one\ntwo\n3\n", "L1b, after the squash")
        git("checkout", "-q", "-b", "layer2b")
        self.commit("g", "layer two\n", "L2 on L1b")
        reviewed = git("rev-parse", "HEAD")
        self.merge_master_ours()
        git("checkout", "-q", "layer2b")
        git("merge", "-q", "-s", "ours", "-m", "merge master", "master")
        self.assertEqual(rc.exposed(reviewed, "layer1", ["master"]),
                         [git("rev-parse", "layer1")])
        self.assertEqual(rc.exposed(self.reviewed, "layer1~1",
                                    ["master"]), [])

    def test_a_reverted_squash_leaves_the_lower_layer_exposed(self):
        git("checkout", "-q", "master")
        git("revert", "--no-edit", "HEAD")
        self.assertEqual(rc.exposed(self.reviewed, "layer1", ["master"]),
                         [git("rev-parse", "layer1")])

    def test_merging_a_base_commit_the_base_later_reverted_adds_nothing(self):
        git("checkout", "-q", "master")
        self.commit("h", "reverted later\n", "X")
        x = git("rev-parse", "HEAD")
        git("revert", "--no-edit", "HEAD")
        git("checkout", "-q", "layer2")
        git("merge", "-q", "-s", "ours", "-m", "merge S", "master~2")
        git("merge", "-q", "-m", "merge the pre-revert X", x)
        head = git("rev-parse", "HEAD")
        baseline = git("merge-base", head, "master")
        self.assertEqual(
            rc.changes(self.reviewed, head, ["master"], [baseline, "master"]),
            "")
        merged = git("merge-tree", "--write-tree", "master", head)
        self.assertNotIn("\th\n", git("ls-tree", merged) + "\n")

    def test_a_rewritten_lower_layer_is_exposed(self):
        old = git("rev-parse", "layer1")
        git("checkout", "-q", "layer1")
        git("reset", "-q", "--hard", "HEAD~1")
        self.commit("f", "uno\n2\n3\n", "L1, rewritten")
        self.assertEqual(rc.exposed(self.reviewed, old,
                                    ["layer1", "master~1"]), [old])

    def test_a_record_is_trusted_by_the_workflow_its_merge_ref_ran(self):
        self.publish("master")
        workflow = ".github/workflows/claude-code-review.yml"
        os.makedirs(".github/workflows")
        git("checkout", "-q", "master")
        self.commit(workflow, "v2\n", "master's newer workflow")
        self.publish("master")
        master = git("rev-parse", "master")
        record = {"head": self.reviewed, "base_sha": master}
        self.assertTrue(rc.trusted(record, self.reviewed, "master"))
        self.assertFalse(rc.trusted(record, "layer1", "master"))
        git("checkout", "-q", "layer2")
        os.makedirs(".github/workflows", exist_ok=True)
        self.commit(workflow, "forged\n", "layer2 rewrites the workflow")
        forged = git("rev-parse", "HEAD")
        self.assertFalse(rc.trusted({"head": forged, "base_sha": master},
                                    forged, "master"))
        self.assertFalse(rc.trusted({"head": forged, "base_sha": forged},
                                    forged, "master"))
        self.assertFalse(rc.trusted(
            {"head": self.reviewed, "base_sha": git("rev-parse", "master~1")},
            self.reviewed, "master"))

    def test_a_base_no_longer_in_the_history_is_unknown(self):
        self.assertIsNone(rc.exposed(self.reviewed, "0" * 40,
                                     ["master"]))

    def test_a_base_whose_action_configuration_differs_holds_the_review(self):
        git("checkout", "-q", "master")
        self.commit("CLAUDE.md", "master's\n", "master changes its config")
        self.publish("master", "layer1")
        self.assertTrue(rc.config_from_base("layer1", "master"))
        git("checkout", "-q", "layer1")
        git("merge", "-q", "-m", "merge master", "master")
        self.publish("layer1")
        self.assertFalse(rc.config_from_base("layer1", "master"))
        os.mkdir(".claude")
        self.commit(".claude/settings.json", "{}\n", "layer1 adds settings")
        self.publish("layer1")
        self.assertTrue(rc.config_from_base("layer1", "master"))
        self.assertFalse(rc.config_from_base("master", "master"))

def pr(number, base, head, repo="o/r"):
    return {"number": number, "title": f"layer {number}", "state": "open",
            "base": {"ref": base},
            "head": {"ref": head, "repo": {"full_name": repo}}}


class Stack(unittest.TestCase):
    def setUp(self):
        self.prs = [pr(1, "master", "a"), pr(2, "a", "b"), pr(3, "b", "c"),
                    pr(4, "master", "z"), pr(5, "b", "master"),
                    pr(6, "a", "master", repo="fork/r")]

    def find(self, head=None, base=None):
        return [p for p in self.prs
                if (head is None or f"o:{p['head']['ref']}" == head)
                and (base is None or p["base"]["ref"] == base)]

    def chain(self, index):
        layers = rc.branch_chain(self.prs[index], "o/r", "master", self.find)
        return [e["number"] for e in layers]

    def test_chain_from_the_middle(self):
        self.assertEqual(self.chain(1), [1, 2, 3])

    def test_a_lone_pull_request_has_no_stack(self):
        layers = rc.branch_chain(self.prs[3], "o/r", "master", self.find)
        self.assertEqual(rc.stack_markdown(layers, 4, "master"), "")

    def test_a_head_named_like_master_has_nothing_above_it(self):
        self.assertEqual(self.chain(4), [1, 2, 5])

    def test_a_fork_pull_request_has_no_chain(self):
        self.assertEqual(self.chain(5), [6])

    def test_markdown_marks_this_layer_and_merged_ones(self):
        layers = [rc.layer({"number": 1, "title": "bottom", "state": "MERGED",
                            "baseRefName": "master"}),
                  rc.layer(self.prs[1])]
        text = rc.stack_markdown(layers, 2, "a")
        self.assertIn("- #1 bottom (merged), on `master`", text)
        self.assertIn("- #2 layer 2 (this pull request), on `a`", text)
        self.assertIn("Each layer is reviewed on its own pull request", text)
        self.assertIn("rule for stacked pull requests in `review-rules.md`",
                      text)

    def test_a_failed_lookup_leaves_the_stack_out(self):
        failure = subprocess.CalledProcessError(1, "gh")
        out = io.StringIO()
        with mock.patch.object(rc, "run", side_effect=failure), \
                contextlib.redirect_stdout(out):
            self.assertEqual(rc.stack_layers("o/r", self.prs[1], "master"),
                             ("no stack", []))
        self.assertIn("native stack lookup failed", out.getvalue())
        self.assertIn("branch chain lookup failed", out.getvalue())

    def test_a_failed_native_lookup_falls_back_to_the_branch_chain(self):
        failure = subprocess.CalledProcessError(1, "gh")
        with mock.patch.object(rc, "native_stack", side_effect=failure), \
                mock.patch.object(rc, "pulls",
                                  lambda repo, **query: self.find(**query)), \
                contextlib.redirect_stdout(io.StringIO()):
            source, layers = rc.stack_layers("o/r", self.prs[1], "master")
        self.assertEqual(source, "branch chain")
        self.assertEqual([e["number"] for e in layers], [1, 2, 3])


def zipped(record):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("record.json", json.dumps(record))
    return buffer.getvalue()


class LastReviewed(unittest.TestCase):
    def last(self, records, runs,
             trusted=lambda record, head, default: True):
        def run(*args, check=True):
            path = next(a for a in args[2:] if not a.startswith("-"))
            if "/artifacts?" in path:
                return "".join(f"{r}\t{a}\t{h}\n" for r, a, h in records)
            workflow, event = runs[path.rsplit("/", 1)[1]]
            return json.dumps({"path": workflow, "event": event})

        def fetch(args, **kwargs):
            artifact = args[2].split("/")[-2]
            return subprocess.CompletedProcess(
                args, 0, zipped({"head": f"head of {artifact}"}))
        with mock.patch.object(rc, "run", run), \
                mock.patch.object(rc, "trusted", trusted), \
                mock.patch.object(rc.subprocess, "run", fetch):
            return rc.last_reviewed("o/r", 7, "master")

    def test_the_last_review_is_the_latest_run_not_the_latest_finish(self):
        records = [("1", "a1", "c1"), ("3", "a3", "c3"), ("2", "a2", "c2")]
        runs = {r: (rc.WORKFLOW + "@refs/pull/7/merge", "pull_request")
                for r in "123"}
        self.assertEqual(self.last(records, runs), {"head": "head of a3"})

    def test_a_record_from_another_workflow_is_not_a_review(self):
        records = [("2", "a2", "c2"), ("3", "a3", "c3")]
        runs = {"3": (".github/workflows/x.yml", "pull_request"),
                "2": (rc.WORKFLOW, "pull_request")}
        self.assertEqual(self.last(records, runs), {"head": "head of a2"})

    def test_a_record_from_a_rewritten_review_workflow_is_not_a_review(self):
        records = [("2", "a2", "c2"), ("3", "a3", "c3")]
        runs = {r: (rc.WORKFLOW, "pull_request") for r in "23"}
        self.assertEqual(
            self.last(records, runs,
                      lambda record, head, default: head == "c2"),
            {"head": "head of a2"})

    def test_no_recorded_review(self):
        self.assertIsNone(self.last([], {}))

    def test_the_workflow_uploads_the_record_the_script_names(self):
        with open(WORKFLOW) as f:
            text = f.read()
        self.assertIn("uses: actions/upload-artifact@", text)
        self.assertIn("name: ${{ steps.context.outputs.record }}", text)
        self.assertIn(".review-context/record.json", text)
        self.assertIn("include-hidden-files: true", text)
        self.assertIn("overwrite: true", text)

    def test_a_retarget_runs_the_review_and_other_edits_do_not(self):
        with open(WORKFLOW) as f:
            text = f.read()
        self.assertIn("types: [opened, synchronize, edited]", text)
        self.assertIn("if: github.event.action != 'edited' || "
                      "github.event.changes.base", text)


class ReviewRules(unittest.TestCase):
    def rules(self, agents):
        return rc.review_rules("master", agents)

    def test_the_section_ends_at_the_next_top_level_heading(self):
        agents = ("# AGENTS.md\n\n## CI\n\nci\n\n## Code Review Rules\n\n"
                  "### Stacks\n\n- one layer\n\n## Later\n\nlater\n")
        self.assertEqual(self.rules(agents),
                         "## Code Review Rules\n\n### Stacks\n\n- one layer\n")

    def test_the_section_may_end_the_file(self):
        agents = "# AGENTS.md\n\n## Code Review Rules\n\n- last\n"
        self.assertEqual(self.rules(agents), "## Code Review Rules\n\n- last\n")

    def test_a_master_without_the_section_says_so(self):
        self.assertIn("has no `## Code Review Rules` section",
                      self.rules("# AGENTS.md\n\n## CI\n"))


if __name__ == "__main__":
    unittest.main()
