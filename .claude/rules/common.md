# Common Project Rules

- Treat code as the source of truth; docs can lag.
- Keep edits focused on the user's request.
- Preserve existing behavior unless the user explicitly asks to change it.
- Never expose `.env`, API keys, cookies, auth secrets, database credentials, or
  generated private data.
- Do not run destructive commands such as `git reset --hard`, broad `git rm`,
  broad recursive deletes, or shell pipelines that delete generated paths unless
  the user explicitly asks and the target has been verified.
- Prefer project-local tools: `.\.venv\Scripts\python.exe -m pytest`,
  `.\.venv\Scripts\pyflakes`, and targeted module commands.
- Keep MCPs/hooks/plugins minimal. Do not bulk-install ECC or any other large
  pack. Add only files that directly help this project.
- Before saying done, report tests/checks actually run and any checks skipped.
