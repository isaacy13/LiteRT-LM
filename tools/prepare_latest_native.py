"""Prepare the next stable source merge on an isolated CI checkout.

This does not build, publish, qualify a model, or mutate a user's app checkout.
Source release selection is live. Custom fork changes are merged, not reset.
Conflicts and any changes to retained test files require review before a commit.
"""
from pathlib import Path
import json
import os
import re
import subprocess
import urllib.request

from ensure_latest_candidate import ensure_candidate

REPOSITORY = 'google-ai-edge/LiteRT-LM'


def run(*args, check=True):
    return subprocess.run(args, check=check, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)


def main():
    headers = {'User-Agent': 'GuideAI-latest-native-refresh'}
    token = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if token:
        headers['Authorization'] = 'Bearer ' + token
    req = urllib.request.Request(f'https://api.github.com/repos/{REPOSITORY}/releases/latest',
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=45) as response:
        data = response.read(1048577)
    if len(data) > 1048576:
        raise ValueError('Release response exceeded its bound.')
    release = json.loads(data)
    tag = release.get('tag_name')
    if (release.get('draft') is not False or release.get('prerelease') is not False
            or not isinstance(tag, str) or not re.fullmatch(r'v[0-9]+\.[0-9]+\.[0-9]+', tag)):
        raise ValueError('Latest release is not a stable version.')
    existing = json.loads(Path('UPSTREAM_RELEASE.json').read_text())
    if run('git', 'status', '--porcelain', '--untracked-files=no').stdout.strip():
        raise ValueError('The isolated source checkout is not clean.')
    base = run('git', 'rev-parse', 'HEAD').stdout.strip()
    branch = 'codex/native-latest-' + tag.replace('.', '-')
    run('git', 'fetch', '--no-tags', f'https://github.com/{REPOSITORY}.git', f'refs/tags/{tag}')
    source = run('git', 'rev-parse', 'FETCH_HEAD^{commit}').stdout.strip()
    if not re.fullmatch(r'[0-9a-f]{40}', source):
        raise ValueError('Stable source did not resolve to a complete commit.')
    if existing.get('tag') == tag:
        if existing.get('commit') != source:
            raise ValueError('Captured latest tag source identity has changed.')
        print('Latest stable source is already captured; qualification remains separate.')
        return
    retained = run('git', 'ls-files', '*_test.cc').stdout.splitlines()
    tests_before = {name: Path(name).read_bytes() for name in retained}
    run('git', 'checkout', '-b', branch)
    merge = run('git', 'merge', '--no-commit', '--no-ff', source, check=False)
    changed_tests = [name for name, value in tests_before.items()
                     if not Path(name).exists() or Path(name).read_bytes() != value]
    if merge.returncode != 0 or changed_tests:
        refusal = {'latestTag': tag, 'sourceRevision': source,
                   'mergeConflicts': run('git', 'diff', '--name-only', '--diff-filter=U').stdout.splitlines(),
                   'retainedTestFilesChanged': changed_tests, 'qualificationClaimed': False}
        Path('latest-refresh-refusal.json').write_text(json.dumps(refusal, indent=2) + '\n')
        raise ValueError('Merge or retained test changes need review; no candidate commit/push.')
    manifest = {'selection': 'latest-stable', 'tag': tag, 'commit': source,
                'releaseURL': release['html_url'], 'publishedAt': release['published_at']}
    Path('UPSTREAM_RELEASE.json').write_text(json.dumps(manifest, indent=2) + '\n')
    run('git', 'add', 'UPSTREAM_RELEASE.json')
    try:
        candidate = ensure_candidate(branch, base, 'UPSTREAM_RELEASE.json', manifest,
                                     f'Update native runtime to latest stable {tag}', upstream=source)
    except Exception as error:
        Path('latest-refresh-refusal.json').write_text(json.dumps({
            'latestTag': tag, 'sourceRevision': source, 'approvedBase': base,
            'errorType': type(error).__name__, 'error': str(error),
            'qualificationClaimed': False}, indent=2) + '\n')
        raise
    Path('latest-refresh.json').write_text(json.dumps({**manifest, **candidate}, indent=2) + '\n')
    print(branch)


if __name__ == '__main__':
    main()
