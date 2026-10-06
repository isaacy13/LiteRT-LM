#!/usr/bin/env python3
"""Fail-closed package closure controls; no native model or binary is executed."""
import unittest
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

from package_apple_runtime import (LIBRARIES, install_name, validate_dependencies,
                                 retained_package_work, rewrite_linkage, run)


class ApplePackageSigningOrderTests(unittest.TestCase):
    def testLinkageRewriteDoesNotStripSignedInputs(self):
        binary = Path("fixture-framework/CLiteRTLM")
        linked = ["@rpath/" + name for name in LIBRARIES]
        with patch("package_apple_runtime.dependencies", return_value=linked), \
                patch("package_apple_runtime.run") as native, \
                patch("package_apple_runtime.subprocess.run") as signature_probe:
            rewrite_linkage(binary, "CLiteRTLM")
        signature_probe.assert_not_called()
        self.assertEqual(native.call_args_list[0].args,
                         ("install_name_tool", "-id", install_name("CLiteRTLM"), str(binary)))
        self.assertEqual(len(native.call_args_list), 1 + len(LIBRARIES))
        for call, dependency in zip(native.call_args_list[1:], linked):
            self.assertEqual(call.args,
                             ("install_name_tool", "-change", dependency,
                              install_name(LIBRARIES[dependency.removeprefix("@rpath/")]), str(binary)))


class AppleRuntimeClosureTests(unittest.TestCase):
    def setUp(self):
        self.allowed = {install_name(name) for name in ["CLiteRTLM", *LIBRARIES.values()]}
        self.linked = [install_name(name) for name in LIBRARIES.values()]
        self.linked += [install_name("CLiteRTLM"), "/usr/lib/libc++.1.dylib",
                        "/System/Library/Frameworks/Metal.framework/Metal"]

    def testCompleteExplicitFrameworkGraphAndSystemDependencies(self):
        validate_dependencies("CLiteRTLM", self.linked, self.allowed)

    def testActuallyObservedPreviousPackageDylibDependenciesAreRejected(self):
        with self.assertRaisesRegex(RuntimeError, "Unpackaged dependencies"):
            validate_dependencies("CLiteRTLM", ["@rpath/libLiteRt.dylib",
                                  "@rpath/libGemmaModelConstraintProvider.dylib"], self.allowed)

    def testMissingMetalSamplerCannotHideBehindSuccessfulCoreLinkage(self):
        linked = [name for name in self.linked if name != install_name("LiteRtTopKMetalSampler")]
        with self.assertRaisesRegex(RuntimeError, "not all strong dependencies"):
            validate_dependencies("CLiteRTLM", linked, self.allowed)

    def testFrameworkReferenceWithoutPackagedTargetIsRejected(self):
        allowed = self.allowed - {install_name("GemmaModelConstraintProvider")}
        with self.assertRaisesRegex(RuntimeError, "Unpackaged dependencies"):
            validate_dependencies("CLiteRTLM", self.linked, allowed)

    def testBuildMachineAbsoluteReferenceIsRejected(self):
        with self.assertRaisesRegex(RuntimeError, "Unpackaged dependencies"):
            validate_dependencies("CLiteRTLM", self.linked + ["/Users/runner/libOther.dylib"], self.allowed)

    def testSupportFrameworkCannotAddAnUnknownTransitiveDependency(self):
        with self.assertRaisesRegex(RuntimeError, "Unpackaged dependencies"):
            validate_dependencies("LiteRtRuntime", [install_name("UnpackagedRuntime")], self.allowed)


class ApplePackageFailureEvidenceTests(unittest.TestCase):
    def testCapturedNativeCommandDiagnosticIsExposedWithExitCode(self):
        failure = subprocess.CalledProcessError(
            1, ["install_name_tool", "-id", "framework name"],
            output="fixture command diagnostic\n")
        with patch("package_apple_runtime.subprocess.check_output", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, r"Command failed \(1\).*install_name_tool") as result:
                run("install_name_tool", "-id", "framework name")
        self.assertIn("fixture command diagnostic", str(result.exception))
        self.assertIs(result.exception.__cause__, failure)

    def testFailureRetainsExactPartialWorkAndOriginalArchiveDigest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "original.zip"
            archive.write_bytes(b"unit fixture, not an SDK binary")
            destination = root / "package"
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                with retained_package_work(archive, destination, "fixture-source") as work:
                    (work / "partial-binary").write_bytes(b"exact partial fixture bytes")
                    raise RuntimeError("fixture failure")
            failed = root / "package-failure"
            self.assertEqual((failed / "partial-binary").read_bytes(), b"exact partial fixture bytes")
            report = json.loads((failed / "FAILURE.json").read_text())
            self.assertEqual(report["source"], "fixture-source")
            self.assertEqual(report["sourceArchiveSHA256"],
                "ebd289a0da2f59788660bb9c1a95437495829198473637ab90f0233be066ade6")
            self.assertFalse(report["artifactAcceptance"])
            self.assertEqual(report["error"], "fixture failure")
            self.assertFalse(destination.exists())

    def testRetainedFailureEvidenceCannotBeReplaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "package-failure").mkdir()
            with self.assertRaisesRegex(RuntimeError, "Never replace"):
                with retained_package_work(root / "original.zip", root / "package", "fixture"):
                    self.fail("Existing evidence must prevent work")


if __name__ == "__main__":
    unittest.main()
