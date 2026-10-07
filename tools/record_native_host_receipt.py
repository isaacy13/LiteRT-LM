"""Record source/run identity only; this file never claims test acceptance."""
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

SOURCE_PATHS = [
    '.bazelrc', '.bazelversion', 'WORKSPACE',
    '.github/workflows/native-byte-contract.yml',
    'tools/record_native_host_receipt.py',
    'runtime/framework/threadpool.h', 'runtime/framework/threadpool.cc',
    'runtime/framework/resource_management/execution_manager.h',
    'runtime/framework/resource_management/threaded_execution_manager.h',
    'runtime/framework/resource_management/threaded_execution_manager.cc',
    'runtime/framework/resource_management/BUILD',
    'runtime/framework/resource_management/scheduling_lifetime_test.cc',
    'runtime/framework/resource_management/fanout_lifetime_test.cc',
    'support/tokenizer/sentencepiece_tokenizer_test.cc',
    'support/tokenizer/huggingface_tokenizer_test.cc',
    'runtime/components/constrained_decoding/constraint_vocabulary_test.cc',
    'runtime/components/constrained_decoding/constraint_provider_factory_test.cc',
    'runtime/components/constrained_decoding/constraint_provider_cache_test.cc',
    'runtime/components/constrained_decoding/llg_constraint_provider_test.cc',
    'runtime/components/constrained_decoding/llg_constraint_test.cc',
    'runtime/components/constrained_decoding/llg_fc_tool_calls_test.cc',
    'runtime/components/constrained_decoding/llg_python_tool_calls_test.cc',
    'runtime/components/constrained_decoding/gemma_tool_constraint_abi_test.cc',
    'runtime/executor/fake_llm_executor_test.cc',
    'runtime/framework/resource_management/execution_manager_test.cc',
    'runtime/conversation/internal_callback_util_test.cc',
    'runtime/conversation/conversation_test.cc',
    'runtime/framework/threadpool_test.cc',
    'c/engine.cc', 'c/BUILD', 'c/engine_stream_terminal_test.cc',
    'c/engine_stream_fanout_lifetime_test.cc',
]

def main():
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
    inputs = event.get('inputs') or {}
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(
        ['git', '--no-optional-locks', 'status', '--porcelain', '--untracked-files=no'],
        text=True).strip()
    receipt = {
        'schema': 1,
        'repository': os.environ['GITHUB_REPOSITORY'],
        'runID': int(os.environ['GITHUB_RUN_ID']),
        'runAttempt': int(os.environ['GITHUB_RUN_ATTEMPT']),
        'jobKey': os.environ['GITHUB_JOB'],
        'event': os.environ['GITHUB_EVENT_NAME'],
        'ref': os.environ['GITHUB_REF'],
        'workflowRef': os.environ['GITHUB_WORKFLOW_REF'],
        'environmentSHA': os.environ['GITHUB_SHA'],
        'checkoutSHA': head,
        'hostOnly': str(inputs.get('host_only', '')).lower() == 'true',
        'callbackOnly': str(inputs.get('callback_only', '')).lower() == 'true',
        'machine': platform.machine(),
        'system': platform.system(),
        'toolchainFingerprint': os.environ['NATIVE_TOOLCHAIN_FINGERPRINT'],
        'bazelVersion': Path('.bazelversion').read_text().strip(),
        'trackedWorkingTreeClean': not dirty,
        'sourceSHA256': {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                         for path in SOURCE_PATHS},
        'testAcceptanceClaimed': False,
    }
    Path('native-host-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')

if __name__ == '__main__':
    main()
