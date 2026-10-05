#!/usr/bin/env python3
"""Fail-closed package closure controls; no native model or binary is executed."""
import unittest

from package_apple_runtime import LIBRARIES, install_name, validate_dependencies


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


if __name__ == "__main__":
    unittest.main()
