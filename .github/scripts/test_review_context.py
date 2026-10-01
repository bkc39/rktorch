"""Tests for review_context.py, run from the repository root:

    python3 .github/scripts/test_review_context.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import review_context as rc  # noqa: E402

SQUASH_232 = "08a13db5d3372361effe89150b0b267350a79468"
BEFORE_SQUASH_233 = "26ff5892996b1ac24f58ddba02fc3ee29a75b54b"
OURS_MERGE_233 = "826b02c8a30c13e590c426c8a71ecb3afb1ccbe5"
LAST_PUSH_233 = "60177b2a6"


def git(*args, cwd=None):
    return subprocess.run(("git",) + args, cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


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

    def test_real_conflict_resolutions_are_still_reviewed(self):
        merge = "57f08da37"
        parents = git("rev-list", "--parents", "-1", merge).split()[1:]
        label, patch = rc.merge_patch(git("rev-parse", merge), parents)
        self.assertEqual(label, "Conflict resolution in this merge")
        self.assertTrue(patch.strip())


class Synthetic(unittest.TestCase):
    """master <- layer1 <- layer2, layer1 squash-merged into master."""

    def setUp(self):
        self.cwd = os.getcwd()
        self.dir = tempfile.TemporaryDirectory()
        os.chdir(self.dir.name)
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

    def tearDown(self):
        os.chdir(self.cwd)
        self.dir.cleanup()

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

    def test_merging_master_into_an_upper_layer_adds_nothing(self):
        self.commit("h", "on master\n", "X")
        git("checkout", "-q", "layer2")
        git("merge", "-q", "-m", "merge master", "master")
        head = git("rev-parse", "HEAD")
        self.assertIn("+on master", rc.changes(self.reviewed, head, ["layer1"]))
        self.assertEqual(
            rc.changes(self.reviewed, head, ["layer1", "master"]), "")


def pr(number, base, head, title=None):
    return {"number": number, "title": title or f"layer {number}",
            "state": "open", "base": {"ref": base}, "head": {"ref": head}}


class Stack(unittest.TestCase):
    def setUp(self):
        self.prs = [pr(1, "master", "a"), pr(2, "a", "b"), pr(3, "b", "c"),
                    pr(4, "master", "z")]

    def pulls(self, head=None, base=None):
        return [p for p in self.prs
                if (head is None or f"o:{p['head']['ref']}" == head)
                and (base is None or p["base"]["ref"] == base)]

    def test_chain_from_the_middle(self):
        layers = rc.branch_chain(self.prs[1], "o", "master", self.pulls)
        self.assertEqual([e["number"] for e in layers], [1, 2, 3])

    def test_a_lone_pull_request_has_no_stack(self):
        layers = rc.branch_chain(self.prs[3], "o", "master", self.pulls)
        self.assertEqual(rc.stack_markdown(layers, 4, "master"), "")

    def test_markdown_marks_this_layer_and_merged_ones(self):
        layers = [rc.layer({"number": 1, "title": "bottom", "state": "MERGED",
                            "baseRefName": "master"}),
                  rc.layer(self.prs[1])]
        text = rc.stack_markdown(layers, 2, "a")
        self.assertIn("1. #1 bottom (merged); base `master`", text)
        self.assertIn("2. #2 layer 2 (this pull request); base `a`", text)
        self.assertIn("do not flag code that lives in another layer", text)


class LastReviewed(unittest.TestCase):
    """A run whose review step was skipped (workflow validation) covers nothing."""

    def setUp(self):
        self.real_run = rc.run
        runs = {"workflow_runs": [
            {"id": 3, "head_sha": "c3", "pull_requests": [{"number": 7}]},
            {"id": 2, "head_sha": "c2", "pull_requests": [{"number": 7}]},
            {"id": 1, "head_sha": "c1", "pull_requests": [{"number": 7}]}]}
        steps = {3: "skipped", 2: "skipped", 1: "success"}

        def fake(*args, check=True):
            path = args[2]
            if "/jobs" in path:
                run_id = int(path.split("/")[-2])
                return json.dumps({"jobs": [{"steps": [
                    {"name": rc.RECORD_STEP, "conclusion": steps[run_id]}]}]})
            return json.dumps(runs)
        rc.run = fake

    def tearDown(self):
        rc.run = self.real_run

    def test_skipped_reviews_are_not_counted(self):
        self.assertEqual(rc.last_reviewed("o/r", "w.yml", "b", 7, "9"), "c1")

    def test_the_current_run_is_not_counted(self):
        self.assertIsNone(rc.last_reviewed("o/r", "w.yml", "b", 7, "1"))

    def test_another_pull_requests_runs_are_not_counted(self):
        self.assertIsNone(rc.last_reviewed("o/r", "w.yml", "b", 8, "9"))


if __name__ == "__main__":
    unittest.main()
