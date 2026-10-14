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
        import os
        from package_apple_runtime import create_framework_layout
        for mac_partial in (False, True):
            with self.subTest(mac_partial=mac_partial), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = root / "original.zip"
                archive.write_bytes(b"unit fixture, not an SDK binary")
                destination = root / "package"
                original = RuntimeError("fixture failure")
                with self.assertRaisesRegex(RuntimeError, "fixture failure") as result:
                    with retained_package_work(
                            archive, destination, "fixture-source", preserve_symlinks=mac_partial) as work:
                        (work / "partial-binary").write_bytes(b"exact partial fixture bytes")
                        if mac_partial:
                            # Failure between layout creation and binary copy leaves a real dangling link.
                            create_framework_layout(work / "Runtime.framework", "Runtime", "MACOS")
                        raise original
                self.assertIs(result.exception, original)
                failed = root / "package-failure"
                self.assertEqual((failed / "partial-binary").read_bytes(), b"exact partial fixture bytes")
                report = json.loads((failed / "FAILURE.json").read_text())
                self.assertEqual(report["source"], "fixture-source")
                self.assertEqual(report["sourceArchiveSHA256"],
                    "ebd289a0da2f59788660bb9c1a95437495829198473637ab90f0233be066ade6")
                self.assertFalse(report["artifactAcceptance"])
                self.assertEqual(report["error"], "fixture failure")
                self.assertFalse(destination.exists())
                if mac_partial:
                    framework = failed / "Runtime.framework"
                    self.assertTrue((framework / "Runtime").is_symlink())
                    self.assertFalse((framework / "Runtime").exists())
                    self.assertTrue((framework / "Headers").is_symlink())
                    self.assertEqual(os.readlink(framework / "Versions" / "Current"), "A")
                    self.assertTrue((framework / "Versions" / "A").is_dir())

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
        from package_apple_runtime import GEMMA_PROVIDER_FUNCTIONS, GEMMA_ABI_OWNERS
        for name in sorted(GEMMA_PROVIDER_FUNCTIONS):
            with self.subTest(name=name):
                self.inspectFixture(missing=(GEMMA_ABI_OWNERS[name], name))

    def testEveryGemmaCExportHasExactlyOneDeclaredOwner(self):
        from package_apple_runtime import GEMMA_PROVIDER_FUNCTIONS, GEMMA_ABI_OWNERS
        self.assertEqual(set(GEMMA_ABI_OWNERS), GEMMA_PROVIDER_FUNCTIONS)
        self.assertEqual(len(GEMMA_ABI_OWNERS), 4)
        self.assertEqual({name for name, owner in GEMMA_ABI_OWNERS.items() if owner == 'CLiteRTLM'},
                         {'LiteRtLmConstraint_Destroy'})
        self.assertEqual(sum(owner == 'GemmaModelConstraintProvider' for owner in GEMMA_ABI_OWNERS.values()), 3)

    def testIncompleteOrUnknownGemmaAbiOwnerMapRefuses(self):
        from package_apple_runtime import GEMMA_ABI_OWNERS
        missing = dict(GEMMA_ABI_OWNERS); missing.pop('LiteRtLmConstraint_Destroy')
        wrong = dict(GEMMA_ABI_OWNERS); wrong['LiteRtLmConstraint_Destroy'] = 'UnknownRuntime'
        for owners in [missing, wrong]:
            with self.subTest(owners=owners), patch('package_apple_runtime.GEMMA_ABI_OWNERS', owners), \
                    self.assertRaisesRegex(RuntimeError, 'Invalid native tool constraint provider ABI ownership'):
                inspect({})

    def testEachGemmaCExportMissingFromFirstSliceRefuses(self):
        from package_apple_runtime import GEMMA_PROVIDER_FUNCTIONS, GEMMA_ABI_OWNERS
        for name in sorted(GEMMA_PROVIDER_FUNCTIONS):
            with self.subTest(name=name):
                self.inspectFixture(missing=(GEMMA_ABI_OWNERS[name], name), missing_slice='ios-arm64')

    def testGenericDestroyInProviderCannotReplaceItsRuntimeOwner(self):
        self.inspectFixture(wrong_owner=('LiteRtLmConstraint_Destroy', 'GemmaModelConstraintProvider'))

    def testProviderCreationInRuntimeCannotReplaceItsProviderOwner(self):
        self.inspectFixture(wrong_owner=('LiteRtLmGemmaModelConstraintProvider_Create', 'CLiteRTLM'))

    def inspectFixture(self, missing=None, changed_header=None,
                       missing_slice='ios-arm64-simulator', wrong_owner=None,
                       slices=None, unexpected_dependency=None, actual_platform=None,
                       declared_platform=None):
        # Structural Python control only: every native command is mocked.
        import plistlib
        import shutil
        from package_apple_runtime import (ROOT, PUBLIC_C_HEADERS, public_c_exports,
                                          GEMMA_PROVIDER_FUNCTIONS, GEMMA_ABI_OWNERS, SAMPLER_FUNCTIONS, SLICES,
                                          create_framework_layout)
        selected_slices = SLICES if slices is None else slices
        core = public_c_exports(ROOT / 'c')
        exports = {'CLiteRTLM': set(core), 'LiteRtRuntime': {'kLiteRtRuntimeBuiltin'},
                   'GemmaModelConstraintProvider': set(),
                   'LiteRtMetalAccelerator': {'LiteRtAcceleratorImpl'},
                   'LiteRtTopKMetalSampler': SAMPLER_FUNCTIONS}
        for name, owner in GEMMA_ABI_OWNERS.items():
            exports[owner].add(name)
        if wrong_owner:
            name, owner = wrong_owner
            exports[GEMMA_ABI_OWNERS[name]].remove(name)
            exports[owner].add(name)
        with tempfile.TemporaryDirectory() as temporary:
            frameworks = {}
            for name in ['CLiteRTLM', *LIBRARIES.values()]:
                root = Path(temporary) / (name + '.xcframework')
                root.mkdir()
                entries = []
                for identifier, (_, platform) in selected_slices.items():
                    entry = {'LibraryIdentifier': identifier, 'LibraryPath': name + '.framework',
                             'SupportedPlatform': declared_platform or ('macos' if platform == 'MACOS' else 'ios'),
                             'SupportedArchitectures': ['arm64']}
                    if platform == 'IOSSIMULATOR':
                        entry['SupportedPlatformVariant'] = 'simulator'
                    entries.append(entry)
                    framework = root / identifier / (name + '.framework')
                    framework.mkdir(parents=True)
                    metadata_path = (create_framework_layout(framework, name, platform)
                                     if platform == 'MACOS' else framework / 'Info.plist')
                    (framework / name).write_bytes(b'offline structural fixture, not native binary')
                    metadata_path.write_bytes(plistlib.dumps({
                        'CFBundleExecutable': name, 'CFBundlePackageType': 'FMWK', 'MinimumOSVersion': '26.4'}))
                    if name == 'CLiteRTLM':
                        (framework / 'Headers').mkdir(exist_ok=True)
                        for header in PUBLIC_C_HEADERS:
                            shutil.copyfile(ROOT / 'c' / header, framework / 'Headers' / header)
                        if changed_header and platform in ('IOSSIMULATOR', 'MACOS'):
                            (framework / 'Headers' / changed_header).write_text('changed source bytes\n')
                (root / 'Info.plist').write_bytes(plistlib.dumps({'AvailableLibraries': entries}))
                frameworks[name] = root
            signatures = []
            def native(*args):
                if args[:2] == ('lipo', '-archs'):
                    return 'arm64'
                if args[:2] == ('vtool', '-show-build'):
                    platform = actual_platform or ('MACOS' if 'macos-arm64' in args[2] else
                                                  'IOSSIMULATOR' if 'ios-arm64-simulator' in args[2] else 'IOS')
                    return build_version_fixture(platform)
                if args[:2] == ('otool', '-D'):
                    platform = 'MACOS' if 'macos-arm64' in args[2] else 'IOS'
                    return args[2] + ':\n' + install_name(Path(args[2]).name, platform=platform)
                if args[:3] == ('codesign', '--verify', '--strict'):
                    signatures.append(Path(args[3]))
                    return ''
                self.fail('Unexpected native call ' + repr(args))
            def linked(binary):
                platform = 'MACOS' if 'macos-arm64' in binary.parts else 'IOS'
                return [install_name(binary.name, platform=platform), *([install_name(n, platform=platform) for n in LIBRARIES.values()]
                        if binary.name == 'CLiteRTLM' else []),
                        *([unexpected_dependency] if unexpected_dependency and binary.name == 'LiteRtRuntime' else [])]
            def symbols(binary):
                result = set(exports[binary.name])
                if missing and binary.name == missing[0] and missing_slice in binary.parts:
                    result.remove(missing[1])
                return result
            with patch('package_apple_runtime.run', side_effect=native), \
                    patch('package_apple_runtime.dependencies', side_effect=linked), \
                    patch('package_apple_runtime.exported_symbols', side_effect=symbols):
                if missing or changed_header or wrong_owner or unexpected_dependency or actual_platform or declared_platform:
                    with self.assertRaisesRegex(RuntimeError, 'Missing public|tool constraint|header differs|Unpackaged dependencies|Missing pinned|Wrong/ambiguous|Incorrect declared'):
                        inspect(frameworks, slices=selected_slices)
                    blocked_name = (GEMMA_ABI_OWNERS[wrong_owner[0]] if wrong_owner
                                    else missing[0] if missing else 'LiteRtRuntime' if unexpected_dependency else 'CLiteRTLM')
                    blocked_slice = 'ios-arm64' if wrong_owner else missing_slice
                    self.assertFalse(any(p.name == blocked_name + '.framework'
                                         and blocked_slice in p.parts for p in signatures))
                else:
                    result = inspect(frameworks, slices=selected_slices)
                    self.assertEqual(len(result), len(selected_slices) * 5)
                    self.assertEqual(len(signatures), len(selected_slices) * 5)
                    self.assertEqual({row['name'] for row in result}, set(frameworks))


class MacMetalPackageClosureTests(unittest.TestCase):
    inspectFixture = ApplePublicCExportTests.inspectFixture
    def macFixture(self, **options):
        from package_apple_runtime import MACOS_ARM64_SLICES
        return self.inspectFixture(slices=MACOS_ARM64_SLICES, missing_slice='macos-arm64', **options)

    def testCompleteMacMetalClosureUsesAllFiveFrameworks(self):
        self.macFixture()

    def testEveryMacCHeaderRetainsExactSourceBytes(self):
        from package_apple_runtime import PUBLIC_C_HEADERS
        for header in PUBLIC_C_HEADERS:
            with self.subTest(header=header):
                self.macFixture(changed_header=header)

    def testMacRequiresCounterAndEveryConstraintOwner(self):
        from package_apple_runtime import GEMMA_ABI_OWNERS
        self.macFixture(missing=('CLiteRTLM', 'litert_lm_engine_count_first_input_text_tokens'))
        for symbol, owner in GEMMA_ABI_OWNERS.items():
            with self.subTest(symbol=symbol):
                self.macFixture(missing=(owner, symbol))

    def testMacRequiresActualMetalAcceleratorAndEverySamplerExport(self):
        from package_apple_runtime import SAMPLER_FUNCTIONS
        self.macFixture(missing=('LiteRtMetalAccelerator', 'LiteRtAcceleratorImpl'))
        for symbol in SAMPLER_FUNCTIONS:
            with self.subTest(symbol=symbol):
                self.macFixture(missing=('LiteRtTopKMetalSampler', symbol))

    def testUnknownDawnOrWebGpuDependencyIsNotSilentlyOmitted(self):
        for name in ['libwebgpu_dawn.dylib', 'libLiteRtWebGpuAccelerator.dylib']:
            with self.subTest(dependency=name):
                self.macFixture(unexpected_dependency='@rpath/' + name)

    def testMacCannotDeclareOrContainAnIOSBinary(self):
        self.macFixture(actual_platform='IOS')
        self.macFixture(declared_platform='ios')


class MacFrameworkArchiveTests(unittest.TestCase):
    def createFixture(self, directory):
        import plistlib
        from package_apple_runtime import create_framework_layout
        framework = directory / 'Runtime_mac.xcframework' / 'macos-arm64' / 'Runtime.framework'
        info = create_framework_layout(framework, 'Runtime', 'MACOS')
        (framework / 'Runtime').write_bytes(b'non-native archive transport fixture')
        (framework / 'Runtime').chmod(0o755)
        (framework / 'Headers' / 'Runtime.h').write_text('exact header fixture\n')
        (framework / 'Modules' / 'module.modulemap').write_text('exact module fixture\n')
        info.write_bytes(plistlib.dumps({'CFBundleExecutable': 'Runtime', 'CFBundlePackageType': 'FMWK'}))
        signature = framework / 'Versions' / 'A' / '_CodeSignature'
        signature.mkdir()
        (signature / 'CodeResources').write_bytes(b'not-a-signature transport fixture')
        return framework

    def testVersionedMacBundleAndArchiveRoundTripRetainLinksBytesAndModes(self):
        import os
        import stat
        import zipfile
        from package_apple_runtime import (zip_directory, extract_symlink_archive,
                                          validate_macos_framework_layout)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            framework = self.createFixture(root)
            validate_macos_framework_layout(framework, 'Runtime')
            source = root / 'Runtime_mac.xcframework'
            archive = root / 'Runtime_mac.xcframework.zip'
            zip_directory(source, archive, preserve_symlinks=True)
            with zipfile.ZipFile(archive) as bundle:
                links = [entry for entry in bundle.infolist() if stat.S_ISLNK(entry.external_attr >> 16)]
                self.assertEqual(len(links), 5)
                self.assertTrue(all(entry.create_system == 3 for entry in links))
                self.assertFalse(any('/Headers/Runtime.h' in entry.filename
                                     and '/Versions/A/' not in entry.filename for entry in bundle.infolist()))
            restored = root / 'restored'
            extract_symlink_archive(archive, restored)
            copy = restored / framework.relative_to(root)
            validate_macos_framework_layout(copy, 'Runtime')
            for relative in ['Runtime', 'Headers/Runtime.h', 'Modules/module.modulemap',
                             'Resources/Info.plist', 'Versions/A/_CodeSignature/CodeResources']:
                self.assertEqual((framework / relative).read_bytes(), (copy / relative).read_bytes())
            for relative in ['Runtime', 'Headers', 'Modules', 'Resources', 'Versions/Current']:
                self.assertEqual(os.readlink(framework / relative), os.readlink(copy / relative))
            self.assertEqual((copy / 'Runtime').stat().st_mode & 0o777, 0o755)

    def testFlatOrMissingVersionLinksAreRejected(self):
        from package_apple_runtime import validate_macos_framework_layout
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for removed in ['Runtime', 'Headers', 'Modules', 'Resources', 'Versions/Current']:
                framework = self.createFixture(root / removed.replace('/', '-'))
                path = framework / removed
                path.unlink()
                with self.subTest(link=removed), self.assertRaisesRegex(RuntimeError, 'Invalid Mac framework link'):
                    validate_macos_framework_layout(framework, 'Runtime')
            flat = root / 'Flat.framework'
            flat.mkdir()
            (flat / 'Flat').write_bytes(b'flat is not versioned')
            with self.assertRaisesRegex(RuntimeError, 'Invalid Mac framework link'):
                validate_macos_framework_layout(flat, 'Flat')

    def testEscapingLinksDuplicatePathsAndContentThroughLinksRefuse(self):
        import stat
        import zipfile
        from package_apple_runtime import extract_symlink_archive
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, target in enumerate(['/outside', '../outside']):
                archive = root / f'escape{index}.zip'
                with zipfile.ZipFile(archive, 'w') as bundle:
                    entry = zipfile.ZipInfo('Runtime.framework/Headers')
                    entry.create_system = 3
                    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                    bundle.writestr(entry, target)
                with self.subTest(target=target), self.assertRaisesRegex(RuntimeError, 'Unsafe Mac archive symlink'):
                    extract_symlink_archive(archive, root / f'out{index}')
            archive = root / 'traversal.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                entry = zipfile.ZipInfo('Runtime.framework/Headers')
                entry.create_system = 3
                entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                bundle.writestr(entry, 'Versions/Current/Headers')
                bundle.writestr('Runtime.framework/Headers/data', b'no traversal')
            with self.assertRaisesRegex(RuntimeError, 'traverses a symlink'):
                extract_symlink_archive(archive, root / 'traversal-out')
            import warnings
            archive = root / 'duplicate.zip'
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                with zipfile.ZipFile(archive, 'w') as bundle:
                    bundle.writestr('Runtime.framework/data', b'first')
                    bundle.writestr('Runtime.framework/data', b'second')
            with self.assertRaisesRegex(RuntimeError, 'duplicate Mac archive path'):
                extract_symlink_archive(archive, root / 'duplicate-out')


class MacPinnedRuntimeInputTests(unittest.TestCase):
    def testUnfetchedMacLfsInputRefusesBeforeNativeInspection(self):
        import verify_apple_prebuilt_runtime as verifier
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / 'prebuilt' / 'macos_arm64' / verifier.LIBRARIES[0]
            library.parent.mkdir(parents=True)
            library.write_text('version https://git-lfs.github.com/spec/v1\n')
            with patch.object(verifier, 'ROOT', root), patch.object(verifier, 'output') as native:
                with self.assertRaisesRegex(RuntimeError, 'Unfetched LFS binary'):
                    verifier.verify(platforms={'macos_arm64': 'MACOS'})
                native.assert_not_called()

    def testMacInputsRequireExactCommittedLfsBytesAndActualMacPlatform(self):
        import hashlib
        import verify_apple_prebuilt_runtime as verifier
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / 'prebuilt' / 'macos_arm64'
            inputs.mkdir(parents=True)
            for name in verifier.LIBRARIES:
                (inputs / name).write_bytes(('not-native fixture ' + name).encode())
            committed_hashes = {name: hashlib.sha256((inputs / name).read_bytes()).hexdigest()
                                for name in verifier.LIBRARIES}
            platform = 'MACOS'
            def output(*args):
                if args[0] == 'git':
                    path = root / args[-1].removeprefix('HEAD:')
                    return 'oid sha256:' + committed_hashes[path.name]
                if args[:2] == ('lipo', '-archs'):
                    return 'arm64'
                if args[:2] == ('vtool', '-show-build'):
                    return build_version_fixture(platform, '12.0')
                if args[:2] == ('nm', '-gU'):
                    return '_kLiteRtRuntimeBuiltin'
                self.fail('Unexpected mocked inspection command: ' + repr(args))
            with patch.object(verifier, 'ROOT', root), patch.object(verifier, 'output', side_effect=output):
                rows = verifier.verify(platforms={'macos_arm64': 'MACOS'})
                self.assertEqual(len(rows), 4)
                self.assertEqual({row['platform'] for row in rows}, {'MACOS'})
                platform = 'IOS'
                with self.assertRaisesRegex(RuntimeError, 'Expected MACOS'):
                    verifier.verify(platforms={'macos_arm64': 'MACOS'})
                platform = 'MACOS'
                (inputs / verifier.LIBRARIES[0]).write_bytes(b'corrupt retained bytes')
                with self.assertRaisesRegex(RuntimeError, 'Pinned LFS content differs'):
                    verifier.verify(platforms={'macos_arm64': 'MACOS'})


class MacArm64PackageBoundaryTests(unittest.TestCase):
    def native(self, *args):
        if args[0] == 'git':
            return '' if '--porcelain' in args else 'source-fixture'
        if args[:2] == ('lipo', '-archs'):
            return 'arm64'
        if args[:2] == ('vtool', '-show-build'):
            return build_version_fixture('MACOS', '12.0')
        if args[:2] == ('otool', '-D'):
            return args[2] + ':\n@rpath/libCLiteRTLM_mac.dylib'
        self.fail('Unexpected native command in pure boundary control: ' + repr(args))

    def testMacPlatformRequiresExplicitOptInWithoutChangingIOSDefault(self):
        value = build_version_fixture('MACOS', '12.0')
        self.assertEqual(minimum_os_from_build(value, 'MACOS', supported_platforms=('MACOS',)), '12.0')
        with self.assertRaisesRegex(RuntimeError, 'Unsupported Apple runtime platform'):
            minimum_os_from_build(value, 'MACOS')
        with self.assertRaisesRegex(RuntimeError, 'Wrong/ambiguous Mach-O platform'):
            minimum_os_from_build(build_version_fixture('IOS'), 'MACOS', supported_platforms=('MACOS',))

    def testMacUniversalAndIntelCannotBecomeArm64Qualification(self):
        from package_apple_runtime import package_macos_arm64
        with tempfile.TemporaryDirectory() as temporary:
            for architecture in ('x86_64', 'arm64 x86_64'):
                def native(*args):
                    return architecture if args[:2] == ('lipo', '-archs') else self.native(*args)
                with self.subTest(architecture=architecture), patch('package_apple_runtime.run', side_effect=native):
                    with self.assertRaisesRegex(RuntimeError, 'exactly arm64'):
                        package_macos_arm64(Path('fixture-library'), Path(temporary) / 'output')

    def testMacUnexpectedRuntimeClosureRefusesBeforeAnyPackageIsCreated(self):
        from package_apple_runtime import package_macos_arm64
        with tempfile.TemporaryDirectory() as temporary, \
                patch('package_apple_runtime.run', side_effect=self.native), \
                patch('package_apple_runtime.dependencies', return_value=[
                    '@rpath/libCLiteRTLM_mac.dylib', '@rpath/libLiteRt.dylib']):
            output = Path(temporary) / 'output'
            with self.assertRaisesRegex(RuntimeError, 'Unpackaged dependencies'):
                package_macos_arm64(Path('fixture-library'), output)
            self.assertFalse(output.exists())

    def testMacMissingCanonicalExportCannotReachSigningOrAcceptance(self):
        from package_apple_runtime import ROOT, public_c_exports, package_macos_arm64
        exports = public_c_exports(ROOT / 'c') - {'litert_lm_engine_count_first_input_text_tokens'}
        with tempfile.TemporaryDirectory() as temporary, \
                patch('package_apple_runtime.run', side_effect=self.native), \
                patch('package_apple_runtime.dependencies', return_value=['@rpath/libCLiteRTLM_mac.dylib']), \
                patch('package_apple_runtime.exported_symbols', return_value=exports):
            output = Path(temporary) / 'output'
            with self.assertRaisesRegex(RuntimeError, 'Missing public C API exports'):
                package_macos_arm64(Path('fixture-library'), output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
