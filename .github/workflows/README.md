# GitHub Actions Workflows

This directory contains GitHub Actions workflows for the project.

## Documentation Workflows

### `docs.yml` - Build and Deploy Documentation

Builds and deploys the documentation to GitHub Pages.

**Triggers:**
- Push to `main` branch (with changes to docs, mkdocs.yml, or Python code)
- Pull requests (build only, no deploy)
- Manual trigger via workflow_dispatch

**What it does:**
1. Builds the MkDocs documentation
2. Uploads the built site as an artifact
3. Deploys to GitHub Pages (only on main branch)

**Setup Requirements:**

To enable GitHub Pages deployment:

1. Go to your repository Settings → Pages
2. Under "Build and deployment", select:
   - **Source**: GitHub Actions
3. The workflow will automatically deploy on the next push to main

### `docs-pr.yml` - Documentation PR Check

Validates documentation on pull requests without deploying.

**Triggers:**
- Pull requests that modify documentation files

**What it does:**
1. Builds the documentation to check for errors
2. Checks for broken internal links
3. Comments on the PR with build status

## Using the Workflows

### Local Testing

Before pushing, test your documentation locally:

```bash
# Install dependencies
pip install -r docs/requirements.txt

# Serve documentation locally
mkdocs serve

# Build documentation (as CI does)
mkdocs build --strict --verbose
```

### Viewing Deployed Docs

After the first successful deployment:
- Your docs will be available at: `https://your-org.github.io/rust-webnn-graph/`
- The URL will be shown in the workflow run

### Manual Deployment

You can manually trigger documentation deployment:

1. Go to Actions → Build and Deploy Documentation
2. Click "Run workflow"
3. Select the branch and run

## Troubleshooting

### Deployment Fails

If deployment fails with permissions error:
1. Go to Settings → Actions → General
2. Under "Workflow permissions", select:
   - ✅ Read and write permissions
3. Save and re-run the workflow

### Build Fails

Common issues:
- **Broken links**: Check that all internal links use correct paths
- **Missing files**: Ensure all referenced files exist in the docs directory
- **Markdown errors**: Validate your Markdown syntax
- **MkDocs config**: Check mkdocs.yml for syntax errors

### Pages Not Updating

If GitHub Pages aren't updating:
1. Check that the workflow completed successfully
2. Verify GitHub Pages is configured (Settings → Pages)
3. Wait a few minutes for cache to clear
4. Hard refresh your browser (Ctrl+Shift+R / Cmd+Shift+R)
