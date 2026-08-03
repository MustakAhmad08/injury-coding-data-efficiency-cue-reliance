# Anonymous repository release checklist

Before inserting the repository URL into the manuscript:

1. Create a new anonymous repository with no connection to a personal account
   visible to reviewers. Follow the venue's current anonymity policy.
2. Upload the contents of this directory as a new history. Do not push the
   working project's existing Git history, because author names, email
   addresses, paths, branches, and deleted files can remain recoverable.
3. Keep the repository name and description neutral. Do not mention an
   institution, laboratory, grant, course, or author.
4. Confirm that GitHub Pages, issue templates, workflows, badges, and external
   analytics do not reveal ownership.
5. Run `python tests/verify_release.py` from the repository root.
6. Open several text files through the public web interface and verify that the
   repository contains no credentials, private paths, identities, or restricted
   row-level artifacts.
7. Place only the anonymous URL in the review manuscript. After acceptance,
   transfer or recreate the repository under the permanent account, add author
   metadata, tag a release, and archive that release to obtain a DOI.

Suggested manuscript sentence:

> Code, prompts, aggregate results, per-diagnosis metrics, and figure-generation
> scripts are available in an anonymous repository: \url{ANONYMOUS-URL}.
