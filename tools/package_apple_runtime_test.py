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
            minimum_os_from_build(build_version_fixture(), "WATCHOS")

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

    def testMacArm64WholeSDKReadsBackEveryPublicCExportAndDependencyFramework(self):
        self.inspectFixture(macos_arm64=True)
        self.inspectFixture(macos_arm64=True, missing_slice='macos-arm64',
                            missing=('CLiteRTLM', 'litert_lm_engine_count_first_input_text_tokens'))
        self.inspectFixture(macos_arm64=True, changed_header='conversation.h')
        self.inspectFixture(macos_arm64=True, missing_slice='macos-arm64',
                            missing=('GemmaModelConstraintProvider', 'LiteRtLmGemmaModelConstraintProvider_Create'))

    def inspectFixture(self, missing=None, changed_header=None,
                       missing_slice='ios-arm64-simulator', wrong_owner=None, macos_arm64=False):
        # Structural Python control only: every native command is mocked.
        import plistlib
        import shutil
        from package_apple_runtime import (ROOT, PUBLIC_C_HEADERS, public_c_exports,
                                          GEMMA_PROVIDER_FUNCTIONS, GEMMA_ABI_OWNERS, SAMPLER_FUNCTIONS, SLICES, MAC_SLICES, MAC_LIBRARIES)
        slices = MAC_SLICES if macos_arm64 else SLICES
        libraries = MAC_LIBRARIES if macos_arm64 else LIBRARIES
        core = public_c_exports(ROOT / 'c')
        exports = {'CLiteRTLM': set(core), 'LiteRtRuntime': {'kLiteRtRuntimeBuiltin'},
                   'GemmaModelConstraintProvider': set(),
                   'LiteRtMetalAccelerator': {'LiteRtAcceleratorImpl'},
                   'LiteRtTopKMetalSampler': SAMPLER_FUNCTIONS}
        for name in libraries.values():
            exports.setdefault(name, set())
        for name, owner in GEMMA_ABI_OWNERS.items():
            exports[owner].add(name)
        if wrong_owner:
            name, owner = wrong_owner
            exports[GEMMA_ABI_OWNERS[name]].remove(name)
            exports[owner].add(name)
        with tempfile.TemporaryDirectory() as temporary:
            frameworks = {}
            for name in ['CLiteRTLM', *libraries.values()]:
                root = Path(temporary) / (name + '.xcframework')
                root.mkdir()
                entries = []
                for identifier, (_, platform) in slices.items():
                    entry = {'LibraryIdentifier': identifier, 'LibraryPath': name + '.framework',
                             'SupportedPlatform': 'macos' if macos_arm64 else 'ios', 'SupportedArchitectures': ['arm64']}
                    if platform == 'IOSSIMULATOR':
                        entry['SupportedPlatformVariant'] = 'simulator'
                    entries.append(entry)
                    framework = root / identifier / (name + '.framework')
                    framework.mkdir(parents=True)
                    storage = framework / 'Versions/A' if macos_arm64 else framework
                    storage.mkdir(parents=True, exist_ok=True)
                    (storage / name).write_bytes(b'offline structural fixture, not native binary')
                    if macos_arm64:
                        for sub in ['Headers', 'Modules', 'Resources']: (storage / sub).mkdir()
                        (framework / 'Versions/Current').symlink_to('A')
                        for sub in [name, 'Headers', 'Modules', 'Resources']:
                            (framework / sub).symlink_to('Versions/Current/' + sub)
                    info_path = storage / 'Resources/Info.plist' if macos_arm64 else storage / 'Info.plist'
                    info_path.write_bytes(plistlib.dumps({
                        'CFBundleExecutable': name, 'CFBundlePackageType': 'FMWK', 'MinimumOSVersion': '26.4',
                        **({'LSMinimumSystemVersion': '26.4'} if macos_arm64 else {})}))
                    if name == 'CLiteRTLM':
                        if not macos_arm64: (framework / 'Headers').mkdir()
                        for header in PUBLIC_C_HEADERS:
                            shutil.copyfile(ROOT / 'c' / header, framework / 'Headers' / header)
                        if changed_header and (macos_arm64 or platform == 'IOSSIMULATOR'):
                            (framework / 'Headers' / changed_header).write_text('changed source bytes\n')
                (root / 'Info.plist').write_bytes(plistlib.dumps({'AvailableLibraries': entries}))
                frameworks[name] = root
            signatures = []
            def native(*args):
                if args[:2] == ('lipo', '-archs'):
                    return 'arm64'
                if args[:2] == ('vtool', '-show-build'):
                    platform = 'MACOS' if macos_arm64 else 'IOSSIMULATOR' if 'ios-arm64-simulator' in args[2] else 'IOS'
                    return build_version_fixture(platform)
                if args[:2] == ('otool', '-D'):
                    return args[2] + ':\n' + install_name(Path(args[2]).name, macos=macos_arm64)
                if args[:3] == ('codesign', '--verify', '--strict'):
                    signatures.append(Path(args[3]))
                    return ''
                self.fail('Unexpected native call ' + repr(args))
            def linked(binary):
                return [install_name(binary.name, macos=macos_arm64), *([install_name(n, macos=macos_arm64) for n in LIBRARIES.values()]
                        if binary.name == 'CLiteRTLM' else [])]
            def symbols(binary):
                result = set(exports[binary.name])
                if missing and binary.name == missing[0] and missing_slice in binary.parts:
                    result.remove(missing[1])
                return result
            with patch('package_apple_runtime.run', side_effect=native), \
                    patch('package_apple_runtime.dependencies', side_effect=linked), \
                    patch('package_apple_runtime.exported_symbols', side_effect=symbols):
                if missing or changed_header or wrong_owner:
                    with self.assertRaisesRegex(RuntimeError, 'Missing public|tool constraint|header differs'):
                        inspect(frameworks, macos_arm64=macos_arm64)
                    blocked_name = (GEMMA_ABI_OWNERS[wrong_owner[0]] if wrong_owner
                                    else missing[0] if missing else 'CLiteRTLM')
                    blocked_slice = 'macos-arm64' if macos_arm64 else 'ios-arm64' if wrong_owner else missing_slice
                    self.assertFalse(any(p.name == blocked_name + '.framework'
                                         and blocked_slice in p.parts for p in signatures))
                else:
                    result = inspect(frameworks, macos_arm64=macos_arm64)
                    expected_count = 8 if macos_arm64 else 10
                    self.assertEqual(len(result), expected_count)
                    self.assertEqual(len(signatures), expected_count)
                    self.assertEqual({row['name'] for row in result}, set(frameworks))
                    if macos_arm64:
                        self.assertTrue(all(row['bundleLayout'] == 'Versions/A' and len(row['symbolicLinks']) == 5 for row in result))


class MacArm64DevelopmentPackageTests(unittest.TestCase):
    def testMacMetadataRetainsRealMinimumAndRejectsIOS(self):
        with patch('package_apple_runtime.run', return_value=build_version_fixture('MACOS', '12.0')):
            result = framework_metadata('CLiteRTLM', 'MACOS', Path('fixture'))
            self.assertEqual(result['MinimumOSVersion'], '12.0')
            self.assertEqual(result['CFBundleSupportedPlatforms'], ['MacOSX'])
        with self.assertRaisesRegex(RuntimeError, 'Wrong/ambiguous'):
            minimum_os_from_build(build_version_fixture('IOS'), 'MACOS')

    def testDeclaredBazelOutputRejectsAmbiguousMissingAndEscapingPaths(self):
        from package_apple_runtime import resolve_macos_bazel_output
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); listing = root / 'outputs.txt'
            binary = root / 'bazel-out/apple/bin/swift/libCLiteRTLM.dylib'
            binary.parent.mkdir(parents=True); binary.write_bytes(b'offline fixture only')
            with patch('package_apple_runtime.ROOT', root):
                listing.write_text(str(binary.relative_to(root)) + '\n')
                self.assertEqual(resolve_macos_bazel_output(listing), binary)
                for value in ['', '/other.dylib', 'bazel-out/../other.dylib',
                              'bazel-out/missing.dylib', 'bazel-out/file.a',
                              'bazel-out/a.dylib\nbazel-out/b.dylib']:
                    listing.write_text(value)
                    with self.subTest(value=value), self.assertRaises(RuntimeError):
                        resolve_macos_bazel_output(listing)

    def testMacDependenciesAreExactCommittedLFSPayloadsNotAnotherSDK(self):
        from package_apple_runtime import pinned_macos_libraries, digest
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); binary = root / 'prebuilt/macos_arm64/libLiteRt.dylib'
            binary.parent.mkdir(parents=True); binary.write_bytes(b'offline fixture only')
            pointer = ('version https://git-lfs.github.com/spec/v1\noid sha256:' + digest(binary)
                       + '\nsize ' + str(binary.stat().st_size))
            def native(*args):
                if args[0] == 'git': return pointer
                if args[0] == 'lipo': return 'arm64'
                if args[0] == 'vtool': return build_version_fixture('MACOS', '12.0')
                self.fail('Unexpected native call ' + repr(args))
            with patch('package_apple_runtime.ROOT', root), \
                    patch('package_apple_runtime.MAC_LIBRARIES', {'libLiteRt.dylib': 'LiteRtRuntime'}), \
                    patch('package_apple_runtime.run', side_effect=native):
                rows = pinned_macos_libraries()
                self.assertEqual(rows[0]['sha256'], digest(binary))
                for change in ['content', 'size', 'architecture', 'platform', 'symlink']:
                    with self.subTest(change=change):
                        if change == 'content': binary.write_bytes(b'changed fixture only')
                        elif change == 'size': binary.write_bytes(b'offline fixture only' + b'!')
                        elif change == 'symlink':
                            binary.unlink(); binary.symlink_to(root / 'foreign-file')
                        if change in {'architecture', 'platform'}:
                            def wrong(*args):
                                if change == 'architecture' and args[0] == 'lipo': return 'arm64 x86_64'
                                if change == 'platform' and args[0] == 'vtool': return build_version_fixture('IOS')
                                return native(*args)
                            with patch('package_apple_runtime.run', side_effect=wrong), self.assertRaises(RuntimeError):
                                pinned_macos_libraries()
                        else:
                            with self.assertRaises(RuntimeError): pinned_macos_libraries()
                        if change != 'symlink': binary.write_bytes(b'offline fixture only')

    def testMacReadbackCannotAdmitUnpackagedOrForeignRuntimeDependencies(self):
        allowed = {install_name('LiteRtRuntime'), install_name('GemmaModelConstraintProvider')}
        for dependency in ['@rpath/libLiteRt.dylib', '@rpath/Foreign.framework/Foreign',
                           '/Users/runner/old-runtime.dylib', '@loader_path/foreign.dylib']:
            with self.subTest(dependency=dependency), self.assertRaises(RuntimeError):
                validate_dependencies('CLiteRTLM_mac', [dependency], allowed)



class MacVersionedFrameworkTests(unittest.TestCase):
    def fixture(self, root):
        from package_apple_runtime import create_macos_framework
        binary = root / "input"
        binary.write_bytes(b"offline-layout-control-not-a-native-library")
        framework = create_macos_framework(root, "LiteRtRuntime", binary, {
            "CFBundleExecutable": "LiteRtRuntime", "CFBundlePackageType": "FMWK",
            "MinimumOSVersion": "14.0", "LSMinimumSystemVersion": "14.0"})
        xc = root / "LiteRtRuntime.xcframework"
        (xc / "macos-arm64").mkdir(parents=True)
        framework.rename(xc / "macos-arm64/LiteRtRuntime.framework")
        (xc / "Info.plist").write_bytes(b"offline-xcframework-layout-control")
        return xc, xc / "macos-arm64/LiteRtRuntime.framework"

    def testRealVersionedLayoutAndSafeArchiveRoundtrip(self):
        from package_apple_runtime import validate_macos_layout, macos_links, zip_directory, extract_macos_archive
        import zipfile, stat
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); xc, framework = self.fixture(root)
            self.assertEqual(validate_macos_layout(framework, "LiteRtRuntime"), macos_links("LiteRtRuntime"))
            self.assertFalse((framework / "Info.plist").exists())
            archive = root / "framework.zip"; zip_directory(xc, archive, macos=True)
            with zipfile.ZipFile(archive) as z:
                links = {i.filename: z.read(i).decode() for i in z.infolist() if stat.S_ISLNK(i.external_attr >> 16)}
                self.assertEqual(len(links), 5)
                self.assertNotIn("LiteRtRuntime.xcframework/macos-arm64/LiteRtRuntime.framework/Resources/Info.plist", z.namelist())
            restored = extract_macos_archive(archive, root / "restored", "LiteRtRuntime", versioned=True)
            restored_framework = restored / "macos-arm64/LiteRtRuntime.framework"
            self.assertEqual((restored_framework / "LiteRtRuntime").read_bytes(), (framework / "LiteRtRuntime").read_bytes())
            self.assertEqual((restored_framework / "Resources/Info.plist").read_bytes(), (framework / "Resources/Info.plist").read_bytes())
            self.assertEqual(validate_macos_layout(restored_framework, "LiteRtRuntime"), macos_links("LiteRtRuntime"))

    def testFlatBrokenEscapingAndExtraLinkLayoutsRefuse(self):
        from package_apple_runtime import validate_macos_layout
        for boundary in ["flatInfo", "broken", "escape", "extra"]:
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as t:
                xc, framework = self.fixture(Path(t))
                if boundary == "flatInfo": (framework / "Info.plist").write_bytes(b"wrong-flat-info")
                elif boundary == "extra": (framework / "Other").symlink_to("Versions/A")
                else:
                    (framework / "Resources").unlink()
                    (framework / "Resources").symlink_to("Versions/Current/Absent" if boundary == "broken" else "/tmp")
                with self.assertRaises((RuntimeError, FileNotFoundError)):
                    validate_macos_layout(framework, "LiteRtRuntime")

    def testZipTraversalUnknownLinksDuplicateAndAliasWritesRefuse(self):
        from package_apple_runtime import extract_macos_archive, zip_directory
        import zipfile, stat, warnings
        for boundary in ["traversal", "unknownLink", "duplicate", "aliasWrite", "missingLink"]:
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as t:
                root = Path(t); xc, framework = self.fixture(root)
                good = root / "good.zip"; zip_directory(xc, good, macos=True)
                bad = root / "bad.zip"
                with zipfile.ZipFile(good) as src, zipfile.ZipFile(bad, "x") as dst:
                    for i in src.infolist():
                        if boundary == "missingLink" and i.filename.endswith("/Versions/Current"): continue
                        dst.writestr(i, src.read(i))
                    prefix = "LiteRtRuntime.xcframework/macos-arm64/LiteRtRuntime.framework/"
                    if boundary == "traversal": dst.writestr("../escape", b"bad")
                    elif boundary == "unknownLink":
                        i = zipfile.ZipInfo(prefix + "Other"); i.create_system = 3
                        i.external_attr = (stat.S_IFLNK | 0o777) << 16; dst.writestr(i, b"../../outside")
                    elif boundary == "duplicate":
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore"); dst.writestr(src.infolist()[0], src.read(src.infolist()[0]))
                    elif boundary == "aliasWrite": dst.writestr(prefix + "Resources/extra", b"bad")
                with self.assertRaises(RuntimeError):
                    extract_macos_archive(bad, root / "out", "LiteRtRuntime", versioned=True)
                self.assertFalse((root / "escape").exists())

    def testIOSDefaultInstallAndFlatZipRemainUnchanged(self):
        from package_apple_runtime import zip_directory
        import zipfile
        self.assertEqual(install_name("CLiteRTLM"), "@rpath/CLiteRTLM.framework/CLiteRTLM")
        self.assertEqual(install_name("CLiteRTLM", macos=True), "@rpath/CLiteRTLM.framework/Versions/A/CLiteRTLM")
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); directory = root / "IOS.xcframework"; directory.mkdir()
            (directory / "Info.plist").write_bytes(b"original-ios-metadata")
            archive = root / "ios.zip"; zip_directory(directory, archive)
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.namelist(), ["IOS.xcframework/Info.plist"])
                self.assertEqual(z.read(z.namelist()[0]), b"original-ios-metadata")

    def testMacRewritesRawAndAlreadyPackagedMatchingDependencies(self):
        from package_apple_runtime import MAC_LIBRARIES
        old = ["@rpath/libLiteRt.dylib", install_name("GemmaModelConstraintProvider"), install_name("WebGpuDawn")]
        with patch("package_apple_runtime.dependencies", return_value=old), patch("package_apple_runtime.run") as native:
            rewrite_linkage(Path("fixture"), "CLiteRTLM", MAC_LIBRARIES, macos=True)
        self.assertEqual(native.call_args_list[0].args[2], install_name("CLiteRTLM", macos=True))
        self.assertEqual([x.args[3] for x in native.call_args_list[1:]], [install_name(n, macos=True) for n in ["LiteRtRuntime", "GemmaModelConstraintProvider", "WebGpuDawn"]])

    def testMacMetadataDeclaresBothMinimumKeysFromActualBuild(self):
        with patch("package_apple_runtime.run", return_value=build_version_fixture("MACOS", "14.0")):
            metadata = framework_metadata("CLiteRTLM", "MACOS", Path("fixture"))
        self.assertEqual(metadata["LSMinimumSystemVersion"], "14.0")
        self.assertEqual(metadata["MinimumOSVersion"], "14.0")

    def testRepackageSourceSeparatesCompilerAndPackagingAndRejectsNativeChanges(self):
        import copy
        from package_apple_runtime import (MAC_LIBRARIES, validate_repackage_source,
            REPACKAGE_COMPILED_SOURCE, REPACKAGE_MANIFEST_SHA256)
        names = ["CLiteRTLM", *MAC_LIBRARIES.values()]
        original = {"source": REPACKAGE_COMPILED_SOURCE, "archives": dict.fromkeys(names),
            "frameworks": [{"name": n} for n in names], "architectures": ["arm64"],
            "upstreamVersion": "0.18.0", "intelQualified": False, "productionDefaultChanged": False}
        validate_repackage_source(original, REPACKAGE_MANIFEST_SHA256, "b" * 40, ["tools/package_apple_runtime.py"])
        for boundary in ["source", "digest", "nativeChange", "intel", "missing"]:
            with self.subTest(boundary=boundary):
                value = copy.deepcopy(original); digest = REPACKAGE_MANIFEST_SHA256; changed = []
                if boundary == "source": value["source"] = "c" * 40
                elif boundary == "digest": digest = "0" * 64
                elif boundary == "nativeChange": changed = ["c/conversation.cc"]
                elif boundary == "intel": value["intelQualified"] = True
                else: value["archives"].pop("CLiteRTLM")
                with self.assertRaises(RuntimeError): validate_repackage_source(value, digest, "b" * 40, changed)


    def testRetainedMacFailurePreservesExactVersionedLinks(self):
        from package_apple_runtime import validate_macos_layout, macos_links
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); source = root / "source.zip"; source.write_bytes(b"offline-source")
            destination = root / "package"
            with self.assertRaisesRegex(RuntimeError, "retained-link-fixture"):
                with retained_package_work(source, destination, "fixture", preserve_symlinks=True) as work:
                    self.fixture(work)
                    raise RuntimeError("retained-link-fixture")
            failed = root / "package-failure/LiteRtRuntime.xcframework/macos-arm64/LiteRtRuntime.framework"
            self.assertEqual(validate_macos_layout(failed, "LiteRtRuntime"), macos_links("LiteRtRuntime"))
            self.assertTrue((failed / "Resources").is_symlink())
            self.assertTrue((failed / "Versions/Current").is_symlink())
            self.assertEqual((failed / "LiteRtRuntime").read_bytes(), b"offline-layout-control-not-a-native-library")
            self.assertFalse(json.loads((root / "package-failure/FAILURE.json").read_text())["artifactAcceptance"])


class MacNativeContentPreservationTests(unittest.TestCase):
    def binary(self, *, versioned=False, corrupt=False, uuid=b"x" * 16):
        import struct
        name = install_name("CLiteRTLM", macos=versioned).encode() + b"\0"
        length = (24 + len(name) + 7) & ~7
        dylib = struct.pack("<6I", 0xd, length, 24, 0, 65536, 65536) + name
        dylib += b"\0" * (length - len(dylib))
        text = bytearray(152); struct.pack_into("<II", text, 0, 0x19, 152); text[8:14] = b"__TEXT"
        struct.pack_into("<I", text, 64, 1); struct.pack_into("<Q", text, 72 + 40, 32); struct.pack_into("<I", text, 72 + 48, 512)
        link = bytearray(72); struct.pack_into("<II", link, 0, 0x19, 72); link[8:18] = b"__LINKEDIT"
        struct.pack_into("<Q", link, 32, 8192 if versioned else 4096); struct.pack_into("<Q", link, 40, 544); struct.pack_into("<Q", link, 48, 128)
        commands = bytes(text) + bytes(link) + struct.pack("<II", 0x1b, 24) + uuid + dylib + struct.pack("<4I", 0x1d, 16, 640, 32)
        header = struct.pack("<8I", 0xfeedfacf, 0x100000c, 0, 6, 5, len(commands), 0, 0)
        data = bytearray(header + commands); data += b"\0" * (672 - len(data)); data[512:544] = bytes(range(32))
        data[640:] = b"S" * 32
        if corrupt: data[513] ^= 1
        return bytes(data)

    def testOnlyVersionedLinkageSignatureAllocationMayChange(self):
        from package_apple_runtime import require_same_native_content
        self.assertEqual(len(require_same_native_content(self.binary(), self.binary(versioned=True), "CLiteRTLM")), 64)

    def testNativePayloadUUIDOrArchitectureChangesAreRejected(self):
        from package_apple_runtime import require_same_native_content
        for value in [self.binary(versioned=True, corrupt=True), self.binary(versioned=True, uuid=b"y" * 16),
                      b"wrong-architecture" + self.binary()]:
            with self.subTest(prefix=value[:4]), self.assertRaises(RuntimeError):
                require_same_native_content(self.binary(), value, "CLiteRTLM")


if __name__ == "__main__":
    unittest.main()
