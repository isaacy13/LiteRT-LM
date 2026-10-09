"""Reuse or fast-forward an exact prepared candidate on an isolated CI checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

SHA = re.compile(r'[0-9a-f]{40}\Z')


def run(*args, check=True):
    return subprocess.run(args, check=check, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=180)


def identity(value):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError('Incomplete candidate source identity.')
    return value


def ensure_candidate(branch, base, manifest_path, manifest, message, upstream=None, runner=run):
    if not re.fullmatch(r'codex/(?:native-latest|litert-latest)-v[0-9]+-[0-9]+-[0-9]+', branch):
        raise ValueError('Unsupported latest candidate branch.')
    identity(base)
    if upstream is not None:
        identity(upstream)
    if identity(runner('git', 'rev-parse', 'HEAD').stdout.strip()) != base:
        raise ValueError('Candidate preparation did not start from its approved base.')
    expected = identity(runner('git', 'write-tree').stdout.strip())
    listing = runner('git', 'ls-remote', '--exit-code', 'origin', 'refs/heads/' + branch, check=False)
    remote = None
    if listing.returncode == 0:
        rows = listing.stdout.splitlines()
        fields = rows[0].split() if len(rows) == 1 else []
        if len(fields) != 2 or fields[1] != 'refs/heads/' + branch:
            raise ValueError('Ambiguous candidate remote reference.')
        remote = identity(fields[0])
        runner('git', 'fetch', '--no-tags', 'origin', remote)
        if identity(runner('git', 'rev-parse', 'FETCH_HEAD').stdout.strip()) != remote:
            raise ValueError('Fetched candidate identity differs from its remote proof.')
        encoded = runner('git', 'show', remote + ':' + manifest_path).stdout
        if len(encoded.encode('utf-8')) > 1048576 or json.loads(encoded) != manifest:
            raise ValueError('Existing candidate resolved-release identity differs.')
        if upstream is not None and runner('git', 'merge-base', '--is-ancestor', upstream, remote,
                                            check=False).returncode != 0:
            raise ValueError('Existing native candidate does not retain the resolved upstream.')
        remote_tree = identity(runner('git', 'rev-parse', remote + '^{tree}').stdout.strip())
        base_retained = runner('git', 'merge-base', '--is-ancestor', base, remote,
                               check=False).returncode == 0
        if remote_tree == expected and base_retained:
            if runner('git', 'rev-parse', '--verify', '--quiet', 'MERGE_HEAD', check=False).returncode == 0:
                runner('git', 'merge', '--abort')
            else:
                runner('git', 'restore', '--source=HEAD', '--staged', '--worktree', '.')
            runner('git', 'checkout', '--detach', remote)
            return {'branch': branch, 'forkRevision': remote, 'preparedTree': expected,
                    'approvedBase': base, 'observedRemote': remote, 'candidateReused': True, 'qualificationClaimed': False}
    elif listing.returncode != 2:
        raise ValueError('Candidate remote lookup failed; absence is not proven.')
    runner('git', 'commit', '-m', message)
    if remote is not None:
        merged = runner('git', 'merge', '--no-commit', '--no-ff', remote, check=False)
        if merged.returncode != 0 or runner('git', 'write-tree').stdout.strip() != expected:
            raise ValueError('Existing candidate content diverges; no commit/push of a replacement.')
        if runner('git', 'rev-parse', '--verify', '--quiet', 'MERGE_HEAD', check=False).returncode == 0:
            runner('git', 'commit', '-m', 'Update latest candidate from approved source')
        if runner('git', 'merge-base', '--is-ancestor', remote, 'HEAD', check=False).returncode != 0:
            raise ValueError('Candidate update is not a fast-forward.')
    return {'branch': branch, 'forkRevision': identity(runner('git', 'rev-parse', 'HEAD').stdout.strip()),
            'preparedTree': expected, 'approvedBase': base, 'observedRemote': remote,
            'candidateReused': False, 'qualificationClaimed': False}


def check_remote(record, runner=run):
    result = runner('git', 'ls-remote', '--exit-code', 'origin', 'refs/heads/' + record['branch'], check=False)
    expected = record['observedRemote']
    if expected is None:
        if result.returncode != 2:
            raise ValueError('Candidate appeared or remote proof failed before publication.')
    elif result.returncode != 0 or result.stdout.splitlines() != [expected + '\trefs/heads/' + record['branch']]:
        raise ValueError('Candidate remote identity changed before publication.')


# This digest binds the reviewed full push producer, not upstream version selection.
NATIVE_FULL_WORKFLOW_SHA256 = "10598def3fdc92cda6dd8abf31ed9b814d95bfc953109b56f274de4b63c7c4c2"


def native_full_runs(record, runner=run):
    encoded = runner('git', 'show', record['forkRevision'] +
                     ':.github/workflows/native-byte-contract.yml').stdout.encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != NATIVE_FULL_WORKFLOW_SHA256:
        raise ValueError('Native producer source differs from the reviewed full push workflow.')
    runs = json.loads(runner('gh', 'run', 'list', '--workflow', 'native-byte-contract.yml',
                      '--branch', record['branch'], '--commit', record['forkRevision'],
                      '--limit', '100', '--json',
                      'databaseId,headSha,headBranch,event,status,conclusion').stdout)
    if not isinstance(runs, list) or len(runs) >= 100 or any(
            not isinstance(row, dict) or type(row.get('databaseId')) is not int
            or row['databaseId'] <= 0 or row.get('headSha') != record['forkRevision']
            or row.get('headBranch') != record['branch']
            or row.get('event') not in {'push', 'workflow_dispatch'}
            or row.get('status') not in {'queued', 'requested', 'pending', 'waiting',
                                       'in_progress', 'completed'} for row in runs):
        raise ValueError('Ambiguous native candidate workflow inventory.')
    if len({row['databaseId'] for row in runs}) != len(runs):
        raise ValueError('Duplicate native candidate run identity.')
    runs.sort(key=lambda row: row['databaseId'], reverse=True)
    if not runs:
        return runs
    latest = runs[0]
    if latest['status'] == 'completed' and latest.get('conclusion') != 'success':
        raise ValueError('Latest native candidate workflow failed; correction or explicit rerun is required.')
    if latest['event'] == 'workflow_dispatch':
        # Inputs are absent from run-list JSON. A callback/host-only dispatch is
        # never interchangeable with the full producer or its push event.
        if latest['status'] != 'completed':
            raise ValueError('Active native dispatch lacks full-producer proof; do not start a competing run.')
        artifacts = json.loads(runner('gh', 'api',
                    'repos/{owner}/{repo}/actions/runs/' + str(latest['databaseId']) +
                    '/artifacts?per_page=100').stdout)
        if (not isinstance(artifacts, dict)
                or type(artifacts.get('total_count')) is not int
                or not 0 <= artifacts['total_count'] < 100
                or not isinstance(artifacts.get('artifacts'), list)
                or len(artifacts['artifacts']) != artifacts['total_count']):
            raise ValueError('Incomplete native dispatch artifact inventory.')
        rows = artifacts['artifacts']
        if (len(rows) != 1 or not isinstance(rows[0], dict)
                or rows[0].get('name') != 'native-byte-contract-' + record['forkRevision']
                or rows[0].get('expired') is not False
                or type(rows[0].get('size_in_bytes')) is not int
                or rows[0]['size_in_bytes'] <= 0):
            raise ValueError('Native dispatch does not retain the reviewed full-producer artifact.')
    return runs


def ensure_review(record, base_branch, workflow, title, body, runner=run):
    if workflow not in {'ios-ci.yml', 'native-byte-contract.yml'}:
        raise ValueError('Unsupported qualification workflow.')
    published = {**record, 'observedRemote': record['forkRevision']}
    check_remote(published, runner)
    prs = json.loads(runner('gh', 'pr', 'list', '--head', record['branch'], '--base', base_branch,
                     '--state', 'all', '--limit', '100', '--json',
                     'number,state,headRefName,headRefOid,baseRefName,isCrossRepository').stdout)
    if not isinstance(prs, list) or len(prs) >= 100 or any(
            not isinstance(row, dict) or type(row.get('isCrossRepository')) is not bool
            or row.get('state') not in {'OPEN', 'CLOSED', 'MERGED'}
            or row.get('headRefName') != record['branch'] or row.get('baseRefName') != base_branch
            or not isinstance(row.get('headRefOid'), str) or not SHA.fullmatch(row['headRefOid'])
            for row in prs):
        raise ValueError('Ambiguous existing candidate PR inventory.')
    owned = [row for row in prs if row['isCrossRepository'] is False]
    opened = [row for row in owned if row['state'] == 'OPEN']
    if len(opened) > 1 or (opened and opened[0]['headRefOid'] != record['forkRevision']):
        raise ValueError('Existing candidate PR head is ambiguous or stale.')
    if not opened and owned:
        raise ValueError('Candidate PR was closed or merged; no duplicate or automatic reopening.')
    if workflow == 'native-byte-contract.yml':
        runs = native_full_runs(record, runner)
    else:
        runs = json.loads(runner('gh', 'run', 'list', '--workflow', workflow,
                          '--branch', record['branch'], '--commit', record['forkRevision'],
                          '--event', 'workflow_dispatch', '--limit', '100',
                          '--json', 'databaseId,headSha,headBranch,status,conclusion').stdout)
    if not isinstance(runs, list) or len(runs) >= 100 or any(
            not isinstance(row, dict) or row.get('headSha') != record['forkRevision']
            or row.get('headBranch') != record['branch'] for row in runs):
        raise ValueError('Ambiguous existing candidate workflow inventory.')
    if runs and runs[0].get('status') == 'completed' and runs[0].get('conclusion') != 'success':
        raise ValueError('Latest existing candidate workflow failed; correction or explicit rerun is required.')
    if not runs:
        args = ['gh', 'workflow', 'run', workflow, '--ref', record['branch']]
        if workflow == 'ios-ci.yml':
            args += ['-f', 'force_full_ios_ci=true']
        runner(*args)
    if not opened:
        runner('gh', 'pr', 'create', '--draft', '--base', base_branch, '--head', record['branch'],
               '--title', title, '--body', body)
    return {**record, 'existingWorkflowRuns': runs,
            'existingPRNumber': opened[0]['number'] if opened else None,
            'qualificationClaimed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch')
    parser.add_argument('--base')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--message')
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--check-remote', action='store_true')
    parser.add_argument('--ensure-review', action='store_true')
    parser.add_argument('--pr-base')
    parser.add_argument('--workflow')
    parser.add_argument('--title')
    parser.add_argument('--body')
    args = parser.parse_args()
    if args.ensure_review:
        if not all([args.pr_base, args.workflow, args.title, args.body]):
            parser.error('Review reuse needs PR base/workflow/title/body.')
        result = ensure_review(json.loads(args.receipt.read_text()), args.pr_base, args.workflow,
                               args.title, args.body)
        args.receipt.write_text(json.dumps(result, indent=2) + '\n')
        return
    if args.check_remote:
        check_remote(json.loads(args.receipt.read_text()))
        return
    if not all([args.branch, args.base, args.manifest, args.message]):
        parser.error('Candidate preparation needs branch/base/manifest/message.')
    try:
        result = ensure_candidate(args.branch, args.base, str(args.manifest),
                                  json.loads(args.manifest.read_text()), args.message)
    except Exception as error:
        args.receipt.write_text(json.dumps({'branch': args.branch, 'approvedBase': args.base,
                    'candidatePrepared': False, 'qualificationClaimed': False,
                    'errorType': type(error).__name__, 'error': str(error)}, indent=2) + '\n')
        raise
    args.receipt.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
