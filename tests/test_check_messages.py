"""Meaningful format checks; run with standard-library unittest."""

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_messages", ROOT / "scripts/check_messages.py")
MESSAGES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MESSAGES)


class MessageChecks(unittest.TestCase):
    def test_valid_titles_and_breaking_change_marker(self):
        for title in ("docs: explain setup", "feat(env): add I shape", "fix(eval)!: change success rule"):
            with self.subTest(title=title):
                self.assertEqual(MESSAGES.check_title(title), [])

    def test_rejects_missing_type_empty_description_and_long_title(self):
        for title in ("Update files", "feat: ", "unknown: update files", "docs: " + "x" * 73):
            with self.subTest(title=title):
                self.assertTrue(MESSAGES.check_title(title))

    def test_commit_requires_blank_line_before_body(self):
        self.assertTrue(MESSAGES.check_commit("docs: explain setup\nWhy: help coauthors"))
        self.assertEqual(MESSAGES.check_commit("docs: explain setup\n\nWhy: help coauthors"), [])

    def test_unfilled_template_is_rejected(self):
        body = (ROOT / ".github/pull_request_template.md").read_text()
        self.assertEqual(len(MESSAGES.check_pr("docs: explain setup", body)), 5)

    def test_complete_body_is_accepted(self):
        body = "\n\n".join(f"## {section}\n\nRecorded information." for section in MESSAGES.SECTIONS)
        self.assertEqual(MESSAGES.check_pr("chore(git): add message rules", body), [])

    def test_duplicate_or_comment_only_section_is_rejected(self):
        body = "\n\n".join(f"## {section}\n\nRecorded information." for section in MESSAGES.SECTIONS)
        self.assertTrue(MESSAGES.check_pr("docs: explain setup", body + "\n\n## Summary\nDuplicate"))
        body = body.replace("## Validation\n\nRecorded information.", "## Validation\n\n<!-- pending -->")
        self.assertTrue(MESSAGES.check_pr("docs: explain setup", body))


if __name__ == "__main__":
    unittest.main()
