# Git Version Control in Teams

This file contains a summary of the most common git commands when working in teams (*fork-branch-merge* workflow) as well as best practices for git messages and *readme* files:

- [Fork-Branch-Merge Workflow](#fork-branch-merge-workflow)
- [Commit Messages: Best Practices](#commit-messages-best-practices)
- [Readme: Best Practices](#readme-best-practices)
    - [Markdown 101](#markdown-101)

## Fork-Branch-Merge Workflow

```bash
# -- 1. Create a new branch to work on, forking from the default branch (e.g., dev)

# Get everything from remote
git fetch # inspect remote changes before integrating them
git pull # integrate remote changes

# Check local branches
git branch

# Check local and remote branches
git branch -a

# Switch to branch we'd like to derive from, e.g., dev
git checkout dev
git pull

# Create new branch and switch to it
git checkout -b feature/jira-XXX-concept

# -- 2. Work on your branch

# Modify/edit files
# Then either add all to stage or selected ones
git add .

# Commit changes to local repo
git commit -m "feat: explain changes you made"

# Push changes to remote repo in branch we are in
# If first time, we publish branch the first time
# git push [remote] [branch_name]; remote = origin, branch_name = name of the remote branch,
# by default same as local
# If it's the first push, we can publish the branch with
# git push -u origin [branch_name]
# the -u flag when pushing for the first time,
# which sets up tracking between the local and remote branches
git push origin

# -- 3. Merge your branch to the branch you forked it from (e.g., dev)

# Now, we would like to merge our feature/jira-XXX-concept branch
# to the dev branch.
# To that end:
# - FIRST, in case we suspect that somebody changed in dev something that might cause conflicts, we merge dev to feature/jira-XXX-concept; that way we fix any merge issues beforehand. This strategy is well known: "upstream first: keeping your feature branch up to date with the main branch."
# - SECOND, we merge feature/jira-XXX-concept to dev

# FIRST step: Merge locally dev to our feature/jira-XXX-concept (where we are)
# The '--no-ff' flag ensures that even if a fast-forward merge is possible,
# a new merge commit is created to clearly show that a merge operation took place,
# making it easier to track when and where branches were merged.
# If in a file the same function/section was changed simultaneously,
# we have a conflict and the merge won't be done:
# git will mark the file as having a conflict and leave it to you to resolve.
# In that case, 
# - We need to manually edit the conflicted file
# to decide which changes to keep and which to discard.
# - After resolving the conflict,
# we can then commit the changes to complete the merge.
# - Once the conflict is resolved and committed,
# Git will create a new merge commit to record the resolution,
# and both sets of changes will be incorporated into the 'dev' branch.
git checkout dev
git pull
git checkout feature/jira-XXX-concept
git pull origin dev
git merge --no-ff dev
git checkout feature/jira-XXX-concept
git push origin
# NOTE: one way to perform a "dry-run" merge is with the flag "--no-commit"
# That way, the merge is not performed/committed, but instead it's checked for conflicts:
# git merge --no-commit --no-ff dev
# Then, if no conflicts/after resolving them, we finish the merge:
# git merge --continue

# SECOND step: Merge locally our feature/jira-XXX-concept (where we are) to dev
# The end result will be a 'dev' branch that incorporates our changes from feature/jira-XXX-concept
git checkout dev
git merge --no-ff feature/jira-XXX-concept
# BUT, if the feature/jira-XXX-concept branch is pushed to the remote
# we can do the merge via the web interface of Github/Gitlab
# We perform a Pull/Merge request via the web.
    
# GITHUB: Pull request (PR)
# Open a Pull Request on upstream remote:
# (base) upstream:dev <- (head, compare) origin:feature/jira-XXX-concept
# Steps:
# - Go to GitHub, OUR remote repo (origin)
# - Navigate to "feature/jira-XXX-concept" branch
# - "Compare & pull request"
# - "Create Pull Request"
# Then, the OTHERS would accept the PR
# Typically, they:
# - Review
# - Wait for checks in the web GUI, then "Merge" and "Confirm"
# However, we can get change requests back.

# GITLAB: Merge request (MR)
# Instructions vary slightly between platforms, but it's a similar process.
# - Go to GitLab and navigate to your project.
# - Click on "Merge Requests" in the left sidebar.
# - Click the "New merge request" button.
# - Set the source branch (e.g., feature/jira-XXX-concept) and the target branch (e.g., dev).
# - Review and provide a title and description for your MR.
# - Assign reviewers if needed.
# - Click the "Submit merge request" button.
# - Configure approval options
# - Others will review your changes, discuss them, and finally "Accept" the MR.
# - Once accepted, GitLab will automatically merge your changes into the target branch.

# -- 4. Tidy up

# Finally, it's good practice to delete the feature branch
# if not done via GitHub/Gitlab web
# Locally
git checkout dev
git pull origin
git branch -D feature/jira-XXX-concept
# and remotely, if not done automatically by Github/Gitlab
git push origin --delete feature/jira-XXX-concept
```

## Commit Messages: Best Practices

Check: [git-styleguide](https://udacity.github.io/git-styleguide/).

```
<type>: <Subject>
    type
        feat: a new feature
        fix: a bug fix
        docs: changes to documentation
        style: formatting, missing semi colons, etc; no code change
        refactor: refactoring production code
        test: adding tests, refactoring test; no production code change
        chore: updating build tasks, etc; no production code change
    Subject: 50 chars max
        start capital, use imperative, no . at the end
[<body>]
    optional
    longer text paragraph after space, but as concise as possible
    bullet point can go here
    not all commits require one

[<footer>]
    optional
    indicate which issues or bugs the commit addresses
```

## Readme: Best Practices

How to write *readmes*:

- Tutorial style
- Short getting started examples
  - Usage examples
  - Should get user ready as fast as possible
- Short clear sections
- Short simple sentences
- Links to further information
- License!

Anatomy of a *readme*:

```
Title
    Short description
Installation / Getting Started
    Dependencies
    Installation commands
Usage
    Commands
    Known bugs
Contributing
    Guidelines if people wantto contribute
Code Status
    are all tests passing?
    shields: build/passing
    if necessary
FAQs
    if necessary
License / Copyright
    By default, I have the intelectual property, but it's not bad stating it explicitly if necessary
    Choose appropriate license
```

### Markdown 101

Tool for checking our Markdown text: [https://dillinger.io/](https://dillinger.io/).

```
# Basic Syntax

# Heading level 1, H1
## H2
### H3

This is normal text.
**bold text**
*italicized text*
_italicized text_	
~~strikethrough~~
__underline__
> blockquote
1. First item
2. Second item
3. Third item
- First item
- Second item
- Third item
`code`
---
[title](https://www.example.com)
![alt text](image.jpg)

# Extended Syntax

## Tables
| Syntax | Description |
| ----------- | ----------- |
| Header | Title |
| Paragraph | Text |

## Fenced code block
```
```
{
    "firstName": "John",
    "lastName": "Smith",
    "age": 25
}
```
```

## Fenced code block with syntax highlighting
```
```python
while(True):
    pass
```
```

## Footnotes

Here's a sentence with a footnote. [^1]

[^1]: This is the footnote.

## Heading ID

### My Great Heading {#custom-id}

## Definitions

term
: definition

## Task list

- [x] Write the press release
- [ ] Update the website
- [ ] Contact the media

## Links to files and sections

Files should appear as links: [File](file.md)
README sections should appear as anchor links: [Links to files and sections](#links-to-files-and-sections)

## Collapsable text

For some amazing info, you can checkout the [below](#amazing-info) section.

<details><summary>Click to expand</summary>

## Amazing Info
It would be great if a hyperlink could directly show this title.

</details>

## Badges

Check: [https://shields.io/#your-badge](https://shields.io/#your-badge).

Example:

[![Unfinished](https://img.shields.io/badge/status-unfinished-orange)](https://shields.io/#your-badge)
```