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
from package_apple_runtime import (framework_metadata, minimum_os_components,
                                 minimum_os_from_build, validate_minimum_os_metadata, inspect)


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


def build_version_fixture(platform="IOS", minimum="26.4"):
    # Structural vtool input only, not fabricated native execution evidence.
    return ("fixture-binary:\nLoad command 10\n      cmd LC_BUILD_VERSION\n"
            "  cmdsize 32\n platform " + platform + "\n    minos " + minimum
            + "\n      sdk 26.5\n   ntools 1\n     tool LD\n  version 1230.1\n")


class AppleRuntimeMinimumOSTests(unittest.TestCase):
    def testDependencyMetadataUsesActualPerSliceMinimumInsteadOfSDKOrDefault(self):
        for platform, minimum in [("IOS", "15.0"), ("IOSSIMULATOR", "26.4")]:
            with self.subTest(platform=platform), patch("package_apple_runtime.run",
                    return_value=build_version_fixture(platform, minimum)) as native:
                binary = Path("fixture-framework") / platform / "GemmaModelConstraintProvider"
                metadata = framework_metadata("GemmaModelConstraintProvider", platform, binary)
                self.assertEqual(metadata["MinimumOSVersion"], minimum)
                self.assertEqual(metadata["CFBundleSupportedPlatforms"],
                    ["iPhoneSimulator" if platform == "IOSSIMULATOR" else "iPhoneOS"])
                self.assertEqual(metadata["CFBundleExecutable"], "GemmaModelConstraintProvider")
                native.assert_called_once_with("vtool", "-show-build", str(binary))

    def testParserRefusesWrongMissingDuplicateAndLegacyBuildMetadata(self):
        good = build_version_fixture()
        variants = [good.replace("platform IOS", "platform IOSSIMULATOR"),
                    good.replace("    minos 26.4\n", ""),
                    good + "    minos 26.4\n", good + "    minos\n",
                    good + "    minos 26.4 trailing\n",
                    good + " platform IOS\n", good + " platform\n",
                    good.replace("Load command 10\n", ""), good + good,
                    good.replace("LC_BUILD_VERSION", "LC_VERSION_MIN_IPHONEOS")]
        for index, value in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                minimum_os_from_build(value, "IOS")

    def testMalformedMinimumVersionsAreRefused(self):
        for value in [None, True, 26.4, "", "0.0", "26.4 beta", "26.4.0.1",
                      " 26.4", "26.4 ", "026.4", "26.04", "65536.0", "26.256"]:
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                minimum_os_components(value)
        self.assertEqual(minimum_os_components("26"), minimum_os_components("26.0.0"))
        self.assertLess(minimum_os_components("25.10"), minimum_os_components("26.0"))

    def testInvalidMetadataProductionRefusesInsteadOfSubstituting15(self):
        with patch("package_apple_runtime.run", return_value=build_version_fixture("MACOS")):
            with self.assertRaisesRegex(RuntimeError, "Wrong/ambiguous Mach-O platform"):
                framework_metadata("GemmaModelConstraintProvider", "IOS", Path("fixture-binary"))
        with patch("package_apple_runtime.run", return_value=""):
            with self.assertRaisesRegex(RuntimeError, "Ambiguous/missing"):
                framework_metadata("GemmaModelConstraintProvider", "IOS", Path("fixture-binary"))
        with self.assertRaisesRegex(RuntimeError, "Unsupported Apple runtime platform"):
            minimum_os_from_build(build_version_fixture(), "MACOS")

    def testFinalInspectionMetadataMustCoverActualMinimumWithoutProductPolicy(self):
        validate_minimum_os_metadata("15.0", "15")
        validate_minimum_os_metadata("26.4", "26.4")  # Truthful, not app26.0 compatibility.
        validate_minimum_os_metadata("15.0", "26.0")  # Existing C API may declare a stricter minimum.
        with self.assertRaisesRegex(RuntimeError, "understates actual"):
            validate_minimum_os_metadata("26.4", "15.0")
        for declared in [None, 26.4, "26.4 beta"]:
            with self.subTest(declared=declared), self.assertRaises(RuntimeError):
                validate_minimum_os_metadata("26.4", declared)

    def testBothSliceNativeProjectionsRecordActualAndDeclaredMinimum(self):
        self.inspectFixture("26.4", succeeds=True)

    def testFinalInspectionRefusesUnderstatedMetadataBeforeExportAndSignatureAcceptance(self):
        self.inspectFixture("15.0", succeeds=False)

    def inspectFixture(self, declared, succeeds):
        # Real inspect() projection with tiny structural files and mocked read
        # tools. This is an offline control, not framework/model acceptance.
        import plistlib
        with tempfile.TemporaryDirectory() as temporary:
            name = "GemmaModelConstraintProvider"
            root = Path(temporary) / (name + ".xcframework")
            root.mkdir()
            entries = []
            for identifier, platform in [("ios-arm64", "IOS"),
                                         ("ios-arm64-simulator", "IOSSIMULATOR")]:
                library = {"LibraryIdentifier": identifier, "LibraryPath": name + ".framework",
                           "SupportedPlatform": "ios", "SupportedArchitectures": ["arm64"]}
                if platform == "IOSSIMULATOR":
                    library["SupportedPlatformVariant"] = "simulator"
                entries.append(library)
                framework = root / identifier / (name + ".framework")
                framework.mkdir(parents=True)
                (framework / name).write_bytes(b"offline structural fixture; not a native binary")
                (framework / "Info.plist").write_bytes(plistlib.dumps({
                    "CFBundleExecutable": name, "CFBundlePackageType": "FMWK",
                    "MinimumOSVersion": declared}))
            (root / "Info.plist").write_bytes(plistlib.dumps({"AvailableLibraries": entries}))
            def native(*args):
                if args[:2] == ("lipo", "-archs"):
                    return "arm64"
                if args[:2] == ("vtool", "-show-build"):
                    platform = "IOSSIMULATOR" if "ios-arm64-simulator" in args[2] else "IOS"
                    return build_version_fixture(platform, "26.4")
                if args[:2] == ("otool", "-D"):
                    return args[2] + ":\n" + install_name(name)
                if args[:3] == ("codesign", "--verify", "--strict"):
                    return ""
                self.fail("Unexpected tool call: " + repr(args))
            with patch("package_apple_runtime.run", side_effect=native) as calls, \
                    patch("package_apple_runtime.dependencies", return_value=[install_name(name)]), \
                    patch("package_apple_runtime.exported_symbols", return_value={
                        "LiteRtLmGemmaModelConstraintProvider_Create",
                        "LiteRtLmGemmaModelConstraintProvider_Destroy",
                        "LiteRtLmGemmaModelConstraintProvider_CreateConstraintFromTools",
                        "LiteRtLmConstraint_Destroy"}) as exports:
                if succeeds:
                    result = inspect({name: root})
                    self.assertEqual(len(result), 2)
                    self.assertEqual({row["platform"] for row in result}, {"IOS", "IOSSIMULATOR"})
                    self.assertTrue(all(row["minimumOS"] == "26.4"
                                        and row["frameworkMinimumOSVersion"] == "26.4" for row in result))
                    self.assertEqual(exports.call_count, 2)
                    self.assertEqual(sum(call.args[:3] == ("codesign", "--verify", "--strict")
                                         for call in calls.call_args_list), 2)
                else:
                    with self.assertRaisesRegex(RuntimeError, "understates actual"):
                        inspect({name: root})
                    exports.assert_not_called()
                    self.assertFalse(any(call.args[0] == "codesign" for call in calls.call_args_list))


class ApplePublicCExportTests(unittest.TestCase):
    def testActualPublicHeaderGraphAndAll213Declarations(self):
        import re
        from package_apple_runtime import ROOT, PUBLIC_C_HEADERS, PUBLIC_C_SENTINELS, public_c_exports
        build = (ROOT / 'swift' / 'BUILD').read_text()
        public = re.search(r'C_LITERT_LM_PUBLIC_HDRS = \[(.*?)\]', build, re.S).group(1)
        self.assertEqual(set(re.findall(r'"//c:([^\"]+)"', public)), set(PUBLIC_C_HEADERS))
        declared = public_c_exports(ROOT / 'c')
        self.assertEqual(len(declared), 213)
        self.assertTrue(PUBLIC_C_SENTINELS <= declared)
        self.assertIn('litert_lm_loaded_file_model_type', declared)

    def testCommentsLiteralsAndContinuedPreprocessorDoNotInventExports(self):
        from package_apple_runtime import public_c_declarations
        text = ('/* LITERT_LM_C_API_EXPORT int litert_lm_fake(void); */\n'
                '// LITERT_LM_C_API_EXPORT int litert_lm_fake2(void);\n'
                '#define LITERT_LM_C_API_EXPORT \\\n __attribute__((visibility("default")))\n'
                '#define OTHER LITERT_LM_C_API_EXPORT void litert_lm_fake3(void);\n'
                'const char* label = "LITERT_LM_C_API_EXPORT void litert_lm_fake4(void);";\n'
                'LITERT_LM_C_API_EXPORT const char* litert_lm_real(void);\n'
                'LITERT_LM_C_API_EXPORT int litert_lm_other(const LiteRtLmInputData* const* inputs, size_t count);\n')
        self.assertEqual(public_c_declarations(text), {'litert_lm_real', 'litert_lm_other'})

    def testEveryMalformedOrUnmatchedMarkerRefuses(self):
        from package_apple_runtime import public_c_declarations
        declarations = [
            '', 'int litert_lm_missing(void)', 'int not_public(void);',
            'int litert_lm_variable;', 'int litert_lm_variable = 0;',
            'int litert_lm_definition(void) { return 0; }',
            'int litert_lm_first(void), litert_lm_second(void);',
            'int litert_lm_callback(void (*callback)(int));',
            'int litert_lm_array(int values[4]);',
            'int litert_lm_default(int value = 2);',
            'int litert_lm_varargs(int value, ...);',
            'int litert_lm_attribute(void) OTHER_ATTRIBUTE;',
            'LITERT_LM_C_API_EXPORT int litert_lm_second(void);',
        ]
        for declaration in declarations:
            with self.subTest(declaration=declaration), self.assertRaises(RuntimeError):
                public_c_declarations('LITERT_LM_C_API_EXPORT ' + declaration)
        for suffix in ['/* unterminated', '"unterminated', '#define UNFINISHED \\']:
            with self.subTest(suffix=suffix), self.assertRaises(RuntimeError):
                public_c_declarations(suffix)

    def testDuplicateNamesRefuseWithinAndAcrossHeaders(self):
        from package_apple_runtime import PUBLIC_C_HEADERS, public_c_declarations, public_c_exports
        declaration = 'LITERT_LM_C_API_EXPORT int litert_lm_same(void);\n'
        with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
            public_c_declarations(declaration + declaration)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for header in PUBLIC_C_HEADERS:
                (root / header).write_text('')
            (root / 'engine.h').write_text(declaration)
            (root / 'conversation.h').write_text(declaration)
            with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
                public_c_exports(root)

    def testHeaderAndSentinelFailuresCannotProducePartialExpectedSet(self):
        from package_apple_runtime import PUBLIC_C_HEADERS, MAX_PUBLIC_HEADER_BYTES, public_c_exports
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for header in PUBLIC_C_HEADERS:
                (root / header).write_text('')
            with self.assertRaisesRegex(RuntimeError, 'sentinels'):
                public_c_exports(root)
            (root / 'engine.h').write_bytes(b' ' * (MAX_PUBLIC_HEADER_BYTES + 1))
            with self.assertRaisesRegex(RuntimeError, 'bound'):
                public_c_exports(root)
            (root / 'engine.h').unlink()
            with self.assertRaises(FileNotFoundError):
                public_c_exports(root)

    def testEveryActualDeclaredExportIsRequiredNotOnlySentinels(self):
        from package_apple_runtime import ROOT, public_c_exports, validate_public_c_exports
        declared = public_c_exports(ROOT / 'c')
        validate_public_c_exports(declared | {'unrelated_private_symbol'}, declared)
        for name in sorted(declared):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, 'Missing public'):
                validate_public_c_exports(declared - {name}, declared)
        with self.assertRaisesRegex(RuntimeError, 'sentinels'):
            validate_public_c_exports(set(), set())

    def testAllFourGemmaCExportsMatchCurrentSource(self):
        import re
        from package_apple_runtime import ROOT, GEMMA_PROVIDER_FUNCTIONS
        source = (ROOT / 'runtime' / 'components' / 'constrained_decoding' /
                  'gemma_model_constraint_provider.h').read_text()
        declarations = re.findall(r'^GEMMA_MODEL_CONSTRAINT_PROVIDER_EXPORT\s+[^;]+;', source, re.M)
        names = {re.search(r'\b(LiteRtLm[A-Za-z0-9_]+)\s*\(', declaration).group(1)
                 for declaration in declarations}
        self.assertEqual(names, GEMMA_PROVIDER_FUNCTIONS)
        self.assertEqual(len(names), 4)

    def testBothCoreSlicesVerifyAllHeadersAndExportsBeforeSignature(self):
        self.inspectFixture()
        self.inspectFixture(missing=('CLiteRTLM', 'litert_lm_loaded_file_model_type'))
        for header in ('api_export.h', 'model_info.h'):
            with self.subTest(header=header):
                self.inspectFixture(changed_header=header)

    def testEachGemmaCExportMissingFromSecondSliceRefuses(self):
        from package_apple_runtime import GEMMA_PROVIDER_FUNCTIONS
        for name in sorted(GEMMA_PROVIDER_FUNCTIONS):
            with self.subTest(name=name):
                self.inspectFixture(missing=('GemmaModelConstraintProvider', name))

    def inspectFixture(self, missing=None, changed_header=None):
        # Structural Python control only: every native command is mocked.
        import plistlib
        import shutil
        from package_apple_runtime import (ROOT, PUBLIC_C_HEADERS, public_c_exports,
                                          GEMMA_PROVIDER_FUNCTIONS, SAMPLER_FUNCTIONS, SLICES)
        core = public_c_exports(ROOT / 'c')
        exports = {'CLiteRTLM': core, 'LiteRtRuntime': {'kLiteRtRuntimeBuiltin'},
                   'GemmaModelConstraintProvider': GEMMA_PROVIDER_FUNCTIONS,
                   'LiteRtMetalAccelerator': {'LiteRtAcceleratorImpl'},
                   'LiteRtTopKMetalSampler': SAMPLER_FUNCTIONS}
        with tempfile.TemporaryDirectory() as temporary:
            frameworks = {}
            for name in ['CLiteRTLM', *LIBRARIES.values()]:
                root = Path(temporary) / (name + '.xcframework')
                root.mkdir()
                entries = []
                for identifier, (_, platform) in SLICES.items():
                    entry = {'LibraryIdentifier': identifier, 'LibraryPath': name + '.framework',
                             'SupportedPlatform': 'ios', 'SupportedArchitectures': ['arm64']}
                    if platform == 'IOSSIMULATOR':
                        entry['SupportedPlatformVariant'] = 'simulator'
                    entries.append(entry)
                    framework = root / identifier / (name + '.framework')
                    framework.mkdir(parents=True)
                    (framework / name).write_bytes(b'offline structural fixture, not native binary')
                    (framework / 'Info.plist').write_bytes(plistlib.dumps({
                        'CFBundleExecutable': name, 'CFBundlePackageType': 'FMWK', 'MinimumOSVersion': '26.4'}))
                    if name == 'CLiteRTLM':
                        (framework / 'Headers').mkdir()
                        for header in PUBLIC_C_HEADERS:
                            shutil.copyfile(ROOT / 'c' / header, framework / 'Headers' / header)
                        if changed_header and platform == 'IOSSIMULATOR':
                            (framework / 'Headers' / changed_header).write_text('changed source bytes\n')
                (root / 'Info.plist').write_bytes(plistlib.dumps({'AvailableLibraries': entries}))
                frameworks[name] = root
            signatures = []
            def native(*args):
                if args[:2] == ('lipo', '-archs'):
                    return 'arm64'
                if args[:2] == ('vtool', '-show-build'):
                    platform = 'IOSSIMULATOR' if 'ios-arm64-simulator' in args[2] else 'IOS'
                    return build_version_fixture(platform)
                if args[:2] == ('otool', '-D'):
                    return args[2] + ':\n' + install_name(Path(args[2]).name)
                if args[:3] == ('codesign', '--verify', '--strict'):
                    signatures.append(Path(args[3]))
                    return ''
                self.fail('Unexpected native call ' + repr(args))
            def linked(binary):
                return [install_name(binary.name), *([install_name(n) for n in LIBRARIES.values()]
                        if binary.name == 'CLiteRTLM' else [])]
            def symbols(binary):
                result = set(exports[binary.name])
                if missing and binary.name == missing[0] and 'ios-arm64-simulator' in binary.parts:
                    result.remove(missing[1])
                return result
            with patch('package_apple_runtime.run', side_effect=native), \
                    patch('package_apple_runtime.dependencies', side_effect=linked), \
                    patch('package_apple_runtime.exported_symbols', side_effect=symbols):
                if missing or changed_header:
                    with self.assertRaisesRegex(RuntimeError, 'Missing public|tool constraint|header differs'):
                        inspect(frameworks)
                    blocked_name = missing[0] if missing else 'CLiteRTLM'
                    self.assertFalse(any(p.name == blocked_name + '.framework'
                                         and 'ios-arm64-simulator' in p.parts for p in signatures))
                else:
                    result = inspect(frameworks)
                    self.assertEqual(len(result), 10)
                    self.assertEqual(len(signatures), 10)
                    self.assertEqual({row['name'] for row in result}, set(frameworks))


if __name__ == "__main__":
    unittest.main()
