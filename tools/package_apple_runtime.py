#!/usr/bin/env python3
"""Package the pinned iOS runtime dependency graph as explicit XCFrameworks.

Each runtime has its own framework bundle, install name, and SwiftPM binary
artifact. No nested framework or standalone dylib is placed in an iOS app.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import plistlib
import re
import shutil
import shlex
import subprocess
import stat
import tempfile
import zipfile

from verify_apple_prebuilt_runtime import ROOT, verify

LIBRARIES = {
    "libLiteRt.dylib": "LiteRtRuntime",
    "libGemmaModelConstraintProvider.dylib": "GemmaModelConstraintProvider",
    "libLiteRtMetalAccelerator.dylib": "LiteRtMetalAccelerator",
    "libLiteRtTopKMetalSampler.dylib": "LiteRtTopKMetalSampler",
}
SLICES = {"ios-arm64": ("ios_arm64", "IOS"),
          "ios-arm64-simulator": ("ios_sim_arm64", "IOSSIMULATOR")}
MACOS_ARM64_SLICES = {"macos-arm64": ("macos_arm64", "MACOS")}
PUBLIC_C_HEADERS = ("api_export.h", "conversation.h", "embedding_engine.h", "engine.h",
                    "error_reporter.h", "experimental.h", "model_info.h")
PUBLIC_C_SENTINELS = {
    "litert_lm_engine_count_first_input_text_tokens",
    "litert_lm_conversation_send_message_stream",
    "litert_lm_embedding_engine_create",
    "litert_lm_embedding_engine_compute_embedding",
}
GEMMA_PROVIDER_FUNCTIONS = {
    "LiteRtLmGemmaModelConstraintProvider_Create",
    "LiteRtLmGemmaModelConstraintProvider_Destroy",
    "LiteRtLmGemmaModelConstraintProvider_CreateConstraintFromTools",
    "LiteRtLmConstraint_Destroy",
}
# All four declarations remain required. The immutable provider owns its three
# factory/provider entry points; the source-built runtime owns generic deletion.
GEMMA_ABI_OWNERS = {
    "LiteRtLmGemmaModelConstraintProvider_Create": "GemmaModelConstraintProvider",
    "LiteRtLmGemmaModelConstraintProvider_Destroy": "GemmaModelConstraintProvider",
    "LiteRtLmGemmaModelConstraintProvider_CreateConstraintFromTools": "GemmaModelConstraintProvider",
    "LiteRtLmConstraint_Destroy": "CLiteRTLM",
}
MAX_PUBLIC_HEADER_BYTES = 1024 * 1024
_C_TYPE = (r"(?:const\s+)?(?:void|bool|char|int|float|double|size_t|u?int(?:8|16|32|64)_t|"
           r"LiteRtLm[A-Za-z0-9_]+)(?:\s*\*\s*(?:const\b\s*)?)*")
_C_DECLARATION = re.compile(
    rf"({_C_TYPE})\s+(litert_lm_[a-z0-9_]+)\s*\(([^()]*)\)\s*;\Z")
_C_PARAMETER = re.compile(rf"{_C_TYPE}\s+[A-Za-z_][A-Za-z0-9_]*\Z")


SAMPLER_FUNCTIONS = {
    "LiteRtTopKMetalSampler_Create", "LiteRtTopKMetalSampler_Destroy",
    "LiteRtTopKMetalSampler_SampleToIdAndScoreBuffer", "LiteRtTopKMetalSampler_UpdateConfig",
    "LiteRtTopKMetalSampler_CanHandleInput", "LiteRtTopKMetalSampler_HandlesInput",
    "LiteRtTopKMetalSampler_SetInferenceFuncAndInputTensors",
}


def run(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as error:
        # check_output captures the diagnostic; its default traceback omits it.
        raise RuntimeError(
            f"Command failed ({error.returncode}): {shlex.join(args)}\n{error.output}"
        ) from error


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def public_c_declarations(text):
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_PUBLIC_HEADER_BYTES:
        raise RuntimeError("Public C header exceeds its source bound")
    # Remove non-code without interpreting conditional compilation. All declared
    # public branches must be represented in the packaged export set.
    code = []
    index = 0
    while index < len(text):
        if text.startswith("//", index):
            end = text.find("\n", index + 2)
            end = len(text) if end < 0 else end
            code.append(" " * (end - index))
        elif text.startswith("/*", index):
            closing = text.find("*/", index + 2)
            if closing < 0:
                raise RuntimeError("Unterminated public C header comment")
            end = closing + 2
            code.append("".join("\n" if c == "\n" else " " for c in text[index:end]))
        elif text[index] in {"\"", "'"}:
            quote = text[index]
            end = index + 1
            while end < len(text) and text[end] != quote:
                if text[end] == "\\":
                    end += 1
                end += 1
            if end >= len(text):
                raise RuntimeError("Unterminated public C header literal")
            end += 1
            code.append("".join("\n" if c == "\n" else " " for c in text[index:end]))
        else:
            code.append(text[index])
            end = index + 1
        index = end
    lines = []
    continued = False
    for line in "".join(code).splitlines(keepends=True):
        if continued or line.lstrip().startswith("#"):
            continued = line.rstrip("\r\n").endswith("\\")
            lines.append("\n" if line.endswith("\n") else "")
        else:
            lines.append(line)
    if continued:
        raise RuntimeError("Unterminated public C preprocessor directive")
    source = "".join(lines)
    markers = list(re.finditer(r"\bLITERT_LM_C_API_EXPORT\b", source))
    names = set()
    for index, marker in enumerate(markers):
        end = source.find(";", marker.end())
        if end < 0 or (index + 1 < len(markers) and markers[index + 1].start() < end):
            raise RuntimeError("Unmatched public C export marker")
        declaration = source[marker.end():end + 1].strip()
        match = _C_DECLARATION.fullmatch(declaration)
        if match is None:
            raise RuntimeError("Unsupported public C export declaration")
        parameters = match[3].strip()
        if parameters not in {"", "void"} and any(
                _C_PARAMETER.fullmatch(parameter.strip()) is None
                for parameter in parameters.split(",")):
            raise RuntimeError("Unsupported public C export parameters")
        if match[2] in names:
            raise RuntimeError("Duplicate public C export declaration: " + match[2])
        names.add(match[2])
    return names


def public_c_exports(directory):
    names = set()
    for header in PUBLIC_C_HEADERS:
        with (directory / header).open("rb") as stream:
            value = stream.read(MAX_PUBLIC_HEADER_BYTES + 1)
        if len(value) > MAX_PUBLIC_HEADER_BYTES:
            raise RuntimeError("Public C header exceeds its source bound: " + header)
        declared = public_c_declarations(value.decode("utf-8"))
        duplicates = names & declared
        if duplicates:
            raise RuntimeError("Duplicate public C exports across headers: " + repr(sorted(duplicates)))
        names.update(declared)
    if not PUBLIC_C_SENTINELS <= names:
        raise RuntimeError("Required public C API sentinels are missing")
    return names


def validate_public_c_exports(symbols, expected):
    if not PUBLIC_C_SENTINELS <= expected:
        raise RuntimeError("Required public C API sentinels are missing")
    missing = expected - symbols
    if missing:
        raise RuntimeError("Missing public C API exports: " + repr(sorted(missing)))


def install_name(name, *, platform="IOS"):
    version = "Versions/A/" if platform == "MACOS" else ""
    return f"@rpath/{name}.framework/{version}{name}"


def dependencies(binary):
    return [line.strip().split(" (compatibility")[0]
            for line in run("otool", "-L", str(binary)).splitlines()[1:] if line.strip()]


def exported_symbols(binary):
    return {line.split()[-1].removeprefix("_")
            for line in run("nm", "-gU", str(binary)).splitlines() if line.split()}


def validate_dependencies(name, linked, allowed, *, platform="IOS"):
    missing = [dependency for dependency in linked
               if not dependency.startswith(("/usr/lib/", "/System/Library/")) and dependency not in allowed]
    if missing:
        raise RuntimeError(f"Unpackaged dependencies for {name}: {missing}")
    if name == "CLiteRTLM":
        required = {install_name(dependency, platform=platform) for dependency in LIBRARIES.values()}
        if not required <= set(linked):
            raise RuntimeError("Native iOS runtime APIs are not all strong dependencies")


def minimum_os_components(value):
    if not isinstance(value, str) or re.fullmatch(
            r"[1-9][0-9]*(?:\.(?:0|[1-9][0-9]*)){0,2}", value) is None:
        raise RuntimeError(f"Malformed minimum OS version: {value!r}")
    parts = [int(component) for component in value.split(".")]
    parts += [0] * (3 - len(parts))
    if parts[0] > 65535 or any(component > 255 for component in parts[1:]):
        raise RuntimeError(f"Minimum OS version exceeds Mach-O component bounds: {value}")
    return tuple(parts)


def minimum_os_from_build(build, platform, *, supported_platforms=("IOS", "IOSSIMULATOR")):
    if platform not in supported_platforms:
        raise RuntimeError(f"Unsupported Apple runtime platform: {platform}")
    blocks = re.findall(r"^\s*Load command [0-9]+[ \t]*$", build, re.M)
    commands = re.findall(r"^\s*cmd\s+(\S+)\s*$", build, re.M)
    platforms = re.findall(r"^\s*platform\s+(\S+)\s*$", build, re.M)
    minimums = re.findall(r"^\s*minos\s+(\S+)\s*$", build, re.M)
    if len(blocks) != 1 or commands != ["LC_BUILD_VERSION"]:
        raise RuntimeError("Ambiguous/missing Mach-O build version command")
    if platforms != [platform] or len(re.findall(r"^\s*platform\b", build, re.M)) != 1:
        raise RuntimeError("Wrong/ambiguous Mach-O platform")
    if len(minimums) != 1 or len(re.findall(r"^\s*minos\b", build, re.M)) != 1:
        raise RuntimeError("Ambiguous/missing Mach-O minimum OS")
    minimum_os_components(minimums[0])
    return minimums[0]


def validate_minimum_os_metadata(actual, declared):
    if minimum_os_components(declared) < minimum_os_components(actual):
        raise RuntimeError(f"Framework minimum OS {declared} understates actual Mach-O minimum {actual}")


def upstream_release_version():
    release = json.loads((ROOT / "UPSTREAM_RELEASE.json").read_text())
    tag = release.get("tag")
    commit = release.get("commit")
    if not isinstance(tag, str) or re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag) is None:
        raise RuntimeError("Invalid stable upstream release identity")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("Invalid upstream source identity")
    return tag.removeprefix("v")


def framework_metadata(name, platform, binary):
    minimum = minimum_os_from_build(
        run("vtool", "-show-build", str(binary)), platform,
        supported_platforms=("MACOS",) if platform == "MACOS" else ("IOS", "IOSSIMULATOR"))
    return {"CFBundleExecutable": name, "CFBundleIdentifier": f"com.guideai.litert.{name}",
            "CFBundleName": name, "CFBundlePackageType": "FMWK", "CFBundleVersion": "1",
            "CFBundleShortVersionString": upstream_release_version(), "MinimumOSVersion": minimum,
            "CFBundleSupportedPlatforms": ["MacOSX" if platform == "MACOS" else
                                          "iPhoneSimulator" if platform == "IOSSIMULATOR" else "iPhoneOS"]}


def rewrite_linkage(binary, name, *, platform="IOS"):
    # Rewrite the signed copy, then replace its signature when the completed
    # framework is sealed. Early removal can leave signature-alignment padding
    # at the end of __LINKEDIT that older install_name_tool versions reject.
    run("install_name_tool", "-id", install_name(name, platform=platform), str(binary))
    for dependency in dependencies(binary):
        old = dependency.removeprefix("@rpath/")
        if dependency.startswith("@rpath/") and old in LIBRARIES:
            run("install_name_tool", "-change", dependency, install_name(LIBRARIES[old], platform=platform), str(binary))


def create_framework_layout(framework, name, platform):
    if platform != "MACOS":
        (framework / "Headers").mkdir(parents=True)
        (framework / "Modules").mkdir()
        return framework / "Info.plist"
    version = framework / "Versions" / "A"
    for directory in ("Headers", "Modules", "Resources"):
        (version / directory).mkdir(parents=True, exist_ok=True)
    (framework / "Versions" / "Current").symlink_to("A", target_is_directory=True)
    for entry in (name, "Headers", "Modules", "Resources"):
        (framework / entry).symlink_to("Versions/Current/" + entry,
                                      target_is_directory=entry != name)
    return version / "Resources" / "Info.plist"


def validate_macos_framework_layout(framework, name):
    expected = {name: "Versions/Current/" + name,
                "Headers": "Versions/Current/Headers", "Modules": "Versions/Current/Modules",
                "Resources": "Versions/Current/Resources", "Versions/Current": "A"}
    for path, target in expected.items():
        link = framework / path
        if not link.is_symlink() or os.readlink(link) != target:
            raise RuntimeError(f"Invalid Mac framework link: {link}")
    version = framework / "Versions" / "A"
    if not version.is_dir() or version.is_symlink():
        raise RuntimeError(f"Invalid Mac framework version: {version}")
    for directory in ("Headers", "Modules", "Resources"):
        path = version / directory
        if not path.is_dir() or path.is_symlink():
            raise RuntimeError(f"Invalid Mac framework directory: {path}")
    for path in (version / name, version / "Resources" / "Info.plist"):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Invalid Mac framework content: {path}")


def zip_directory(directory, archive, *, preserve_symlinks=False):
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(directory.parent)
            if preserve_symlinks and path.is_symlink():
                entry = zipfile.ZipInfo(str(relative))
                entry.create_system = 3
                entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                bundle.writestr(entry, os.readlink(path).encode("utf-8"))
            elif path.is_file():
                bundle.write(path, relative)
            elif preserve_symlinks and path.is_dir():
                bundle.write(path, relative)


def extract_symlink_archive(archive, destination):
    """Restore our Mac archive for structural and codesign round-trip checks."""
    with zipfile.ZipFile(archive) as bundle:
        entries = {}
        links = {}
        for entry in bundle.infolist():
            path = PurePosixPath(entry.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in entry.filename
                    or not path.parts or path in entries):
                raise RuntimeError("Unsafe or duplicate Mac archive path")
            entries[path] = entry
            if stat.S_ISLNK(entry.external_attr >> 16):
                target = bundle.read(entry).decode("utf-8")
                relative = PurePosixPath(target)
                if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                        or "\\" in target):
                    raise RuntimeError("Unsafe Mac archive symlink")
                links[path] = target
        if any(parent in links for path in entries for parent in path.parents):
            raise RuntimeError("Mac archive content traverses a symlink")
        for path, entry in entries.items():
            if path in links:
                continue
            output = destination.joinpath(*path.parts)
            if entry.is_dir():
                output.mkdir(parents=True, exist_ok=True)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(entry) as source, output.open("xb") as target:
                    shutil.copyfileobj(source, target)
                mode = (entry.external_attr >> 16) & 0o777
                if mode:
                    output.chmod(mode)
        for path, target in links.items():
            output = destination.joinpath(*path.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.symlink_to(target)


def inspect(frameworks, *, slices=SLICES):
    if (set(GEMMA_ABI_OWNERS) != GEMMA_PROVIDER_FUNCTIONS
            or set(GEMMA_ABI_OWNERS.values()) != {"CLiteRTLM", "GemmaModelConstraintProvider"}):
        raise RuntimeError("Invalid native tool constraint provider ABI ownership")
    results = []
    expected_c_exports = public_c_exports(ROOT / "c") if "CLiteRTLM" in frameworks else set()
    for identifier, (_, platform) in slices.items():
        allowed = {install_name(name, platform=platform) for name in frameworks}
        for name, xcframework in frameworks.items():
            info = plistlib.loads((xcframework / "Info.plist").read_bytes())
            entries = info["AvailableLibraries"]
            if len(entries) != len(slices) or {entry["LibraryIdentifier"] for entry in entries} != set(slices):
                raise RuntimeError(f"Incorrect slice coverage in {xcframework}")
            entry = next(entry for entry in entries if entry["LibraryIdentifier"] == identifier)
            expected_variant = "simulator" if platform == "IOSSIMULATOR" else None
            if (entry["SupportedArchitectures"] != ["arm64"] or entry["SupportedPlatform"] != ("macos" if platform == "MACOS" else "ios")
                    or entry.get("SupportedPlatformVariant") != expected_variant):
                raise RuntimeError(f"Incorrect declared platform in {xcframework}/{identifier}")
            framework = xcframework / identifier / entry["LibraryPath"]
            binary = framework / name
            if platform == "MACOS":
                validate_macos_framework_layout(framework, name)
            metadata_path = framework / ("Resources/Info.plist" if platform == "MACOS" else "Info.plist")
            metadata = plistlib.loads(metadata_path.read_bytes())
            if metadata.get("CFBundleExecutable") != name or metadata.get("CFBundlePackageType") != "FMWK":
                raise RuntimeError(f"Invalid framework metadata at {framework}")
            if run("lipo", "-archs", str(binary)) != "arm64":
                raise RuntimeError(f"Wrong architecture at {binary}")
            minimum = minimum_os_from_build(
                run("vtool", "-show-build", str(binary)), platform,
                supported_platforms=("MACOS",) if platform == "MACOS" else ("IOS", "IOSSIMULATOR"))
            validate_minimum_os_metadata(minimum, metadata.get("MinimumOSVersion"))
            linked = dependencies(binary)
            own_id = run("otool", "-D", str(binary)).splitlines()[1:]
            if own_id != [install_name(name, platform=platform)]:
                raise RuntimeError(f"Incorrect install name at {binary}: {own_id}")
            validate_dependencies(name, linked, allowed, platform=platform)
            symbols = exported_symbols(binary)
            required_gemma_exports = {symbol for symbol, owner in GEMMA_ABI_OWNERS.items() if owner == name}
            if not required_gemma_exports <= symbols:
                raise RuntimeError(f"Missing native tool constraint provider ABI at {binary}: "
                                   + repr(sorted(required_gemma_exports - symbols)))
            if name == "CLiteRTLM":
                for header in PUBLIC_C_HEADERS:
                    if digest(framework / "Headers" / header) != digest(ROOT / "c" / header):
                        raise RuntimeError(f"C header differs from source at {framework}/{header}")
                validate_public_c_exports(symbols, expected_c_exports)
            elif name == "LiteRtRuntime" and "kLiteRtRuntimeBuiltin" not in symbols:
                raise RuntimeError(f"Missing native runtime ABI at {binary}")
            elif name == "LiteRtMetalAccelerator" and "LiteRtAcceleratorImpl" not in symbols:
                raise RuntimeError(f"Missing pinned Metal accelerator definition at {binary}")
            elif name == "LiteRtTopKMetalSampler" and not SAMPLER_FUNCTIONS <= symbols:
                raise RuntimeError(f"Missing pinned Metal sampler functions at {binary}")
            run("codesign", "--verify", "--strict", str(framework))
            if list(framework.rglob("*.dylib")) or list(framework.glob("Frameworks/*")):
                raise RuntimeError(f"Unsupported nested runtime payload at {framework}")
            results.append({"name": name, "slice": identifier, "platform": platform,
                            "sha256": digest(binary), "dependencies": linked,
                            "minimumOS": minimum, "frameworkMinimumOSVersion": metadata["MinimumOSVersion"],
                            "requiredGemmaABIExports": sorted(required_gemma_exports)})
    return results


@contextmanager
def retained_package_work(source_archive, destination, source, *, preserve_symlinks=False):
    failure_directory = destination.with_name(destination.name + "-failure")
    if failure_directory.exists():
        raise RuntimeError("Never replace retained package failure evidence")
    with tempfile.TemporaryDirectory(prefix="litert-apple-package-") as temporary:
        work = Path(temporary)
        try:
            yield work
        except Exception as error:
            # Preserve the binaries at the actual failing operation. The original
            # Bazel archive is uploaded separately, before any Mach-O edits.
            shutil.copytree(work, failure_directory, symlinks=preserve_symlinks)
            (failure_directory / "FAILURE.json").write_text(json.dumps({
                "source": source, "sourceArchiveSHA256": digest(source_archive),
                "artifactAcceptance": False, "modelQueries": 0,
                "errorType": type(error).__name__, "error": str(error),
            }, indent=2) + "\n")
            raise


def package(source_archive, destination, *, macos_metal_arm64=False):
    if destination.exists():
        raise RuntimeError("Never replace a retained package")
    slices = MACOS_ARM64_SLICES if macos_metal_arm64 else SLICES
    if macos_metal_arm64:
        verify(platforms={"macos_arm64": "MACOS"})
    else:
        verify()
    source = run("git", "-C", str(ROOT), "rev-parse", "HEAD")
    if run("git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Package only committed source")
    with retained_package_work(
            source_archive, destination, source, preserve_symlinks=macos_metal_arm64) as work:
        capi_name = "CLiteRTLM_mac" if macos_metal_arm64 else "CLiteRTLM"
        capi = work / f"{capi_name}.xcframework"
        if macos_metal_arm64:
            if run("lipo", "-archs", str(source_archive)) != "arm64":
                raise RuntimeError("Mac Metal package requires exactly arm64")
            if run("otool", "-D", str(source_archive)).splitlines()[1:] != ["@rpath/libCLiteRTLM_mac.dylib"]:
                raise RuntimeError("Unexpected source Mac C API install name")
            framework = work / "macos-input" / "CLiteRTLM.framework"
            metadata_path = create_framework_layout(framework, "CLiteRTLM", "MACOS")
            shutil.copy2(source_archive, framework / "CLiteRTLM")
            for header in PUBLIC_C_HEADERS:
                shutil.copy2(ROOT / "c" / header, framework / "Headers" / header)
            (framework / "Modules" / "module.modulemap").write_text(
                'framework module CLiteRTLM {\n' + ''.join(
                    f'  header "{header}"\n' for header in PUBLIC_C_HEADERS) + '  export *\n}\n')
            metadata_path.write_bytes(plistlib.dumps(
                framework_metadata("CLiteRTLM", "MACOS", framework / "CLiteRTLM")))
            run("xcodebuild", "-create-xcframework", "-framework", str(framework), "-output", str(capi))
        else:
            with zipfile.ZipFile(source_archive) as bundle:
                if any(Path(member.filename).is_absolute() or ".." in Path(member.filename).parts
                       for member in bundle.infolist()):
                    raise RuntimeError("Unsafe framework archive path")
                bundle.extractall(work)
        frameworks = {"CLiteRTLM": capi}
        for identifier, (directory, platform) in slices.items():
            framework = capi / identifier / "CLiteRTLM.framework"
            rewrite_linkage(framework / "CLiteRTLM", "CLiteRTLM", platform=platform)
            run("codesign", "--force", "--sign", "-", str(framework))
            for library, name in LIBRARIES.items():
                framework = work / identifier / f"{name}.framework"
                metadata_path = create_framework_layout(framework, name, platform)
                shutil.copy2(ROOT / "prebuilt" / directory / library, framework / name)
                metadata_path.write_bytes(plistlib.dumps(
                    framework_metadata(name, platform, framework / name)))
                (framework / "Headers" / "RuntimeDependency.h").write_text("// APIs are bound internally by CLiteRTLM.\n")
                (framework / "Modules" / "module.modulemap").write_text(
                    f'framework module {name} {{\n  umbrella header "RuntimeDependency.h"\n  export *\n}}\n')
                rewrite_linkage(framework / name, name, platform=platform)
                run("codesign", "--force", "--sign", "-", str(framework))
        for name in LIBRARIES.values():
            target_name = name + ("_mac" if macos_metal_arm64 else "")
            target = work / f"{target_name}.xcframework"
            command = ["xcodebuild", "-create-xcframework"]
            for identifier in slices:
                command += ["-framework", str(work / identifier / f"{name}.framework")]
            run(*command, "-output", str(target))
            frameworks[name] = target
        results = inspect(frameworks, slices=slices)
        destination.mkdir()
        archives = {}
        roundtrip_frameworks = {}
        for name, framework in frameworks.items():
            if macos_metal_arm64:
                run("codesign", "--force", "--sign", "-", str(framework))
                run("codesign", "--verify", "--strict", str(framework))
            archive_name = name + ("_mac" if macos_metal_arm64 else "")
            archive = destination / f"{archive_name}.xcframework.zip"
            zip_directory(framework, archive, preserve_symlinks=macos_metal_arm64)
            if macos_metal_arm64:
                restored = work / "zip-roundtrip" / name
                extract_symlink_archive(archive, restored)
                restored_framework = restored / framework.name
                run("codesign", "--verify", "--strict", str(restored_framework))
                roundtrip_frameworks[name] = restored_framework
            archives[name] = {"file": archive.name, "sha256": digest(archive)}
        if macos_metal_arm64 and inspect(roundtrip_frameworks, slices=slices) != results:
            raise RuntimeError("Mac archive round-trip changed the inspected runtime")
        (destination / "SOURCE_COMMIT").write_text(source + "\n")
        (destination / "SHA256SUMS").write_text("".join(
            f'{entry["sha256"]}  {entry["file"]}\n' for entry in archives.values()))
        report = {"source": source, "modelQueries": 0, "phoneAcceptance": False,
                  "packagedDependencyClosure": True, "frameworks": results, "archives": archives,
                  "gemmaABIExportOwners": GEMMA_ABI_OWNERS}
        if macos_metal_arm64:
            report.update({"platform": "MACOS", "architectures": ["arm64"],
                           "macAppAcceptance": False, "canonicalProcessingQualification": False,
                           "capacityQualification": False, "gpuRuntimeQualification": False})
        (destination / "APPLE_RUNTIME_MANIFEST.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


def package_macos_arm64(source_library, destination):
    """Wrap the existing Mac C API product; certify bytes/ABI, never capacity."""
    if destination.exists():
        raise RuntimeError("Never replace a retained package")
    source = run("git", "-C", str(ROOT), "rev-parse", "HEAD")
    if run("git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Package only committed source")
    if run("lipo", "-archs", str(source_library)) != "arm64":
        raise RuntimeError("Mac qualification requires exactly arm64, not a universal claim")
    minimum = minimum_os_from_build(run("vtool", "-show-build", str(source_library)), "MACOS", supported_platforms=("MACOS",))
    own_id = "@rpath/libCLiteRTLM_mac.dylib"
    linked = dependencies(source_library)
    if run("otool", "-D", str(source_library)).splitlines()[1:] != [own_id]:
        raise RuntimeError("Unexpected Mac C API install name")
    # An unexpected non-system dependency is a packaging failure. Do not ship
    # a partial dynamic runtime, omit a dependency, or silently switch linkage.
    validate_dependencies("CLiteRTLM_mac", linked, {own_id})
    expected = public_c_exports(ROOT / "c")
    validate_public_c_exports(exported_symbols(source_library), expected)
    with tempfile.TemporaryDirectory(prefix="litert-mac-arm64-package-") as temporary:
        work = Path(temporary)
        binary = work / "libCLiteRTLM_mac.dylib"
        shutil.copy2(source_library, binary)
        headers = work / "Headers"
        headers.mkdir()
        header_hashes = {}
        for header in PUBLIC_C_HEADERS:
            shutil.copy2(ROOT / "c" / header, headers / header)
            header_hashes[header] = digest(headers / header)
        module_map = 'module CLiteRTLM {\n' + ''.join(
            f'  header "{header}"\n' for header in PUBLIC_C_HEADERS) + '  export *\n}\n'
        (headers / "module.modulemap").write_text(module_map)
        run("codesign", "--force", "--sign", "-", str(binary))
        run("codesign", "--verify", "--strict", str(binary))
        xcframework = work / "CLiteRTLM_mac.xcframework"
        run("xcodebuild", "-create-xcframework", "-library", str(binary),
            "-headers", str(headers), "-output", str(xcframework))
        info = plistlib.loads((xcframework / "Info.plist").read_bytes())
        entries = info.get("AvailableLibraries", [])
        if len(entries) != 1:
            raise RuntimeError("Mac qualification requires exactly one library slice")
        entry = entries[0]
        if (entry.get("LibraryIdentifier") != "macos-arm64" or
                entry.get("SupportedPlatform") != "macos" or
                entry.get("SupportedArchitectures") != ["arm64"] or
                "SupportedPlatformVariant" in entry or
                entry.get("LibraryPath") != binary.name or
                entry.get("HeadersPath") != "Headers"):
            raise RuntimeError("Incorrect declared Mac arm64 slice")
        library = xcframework / "macos-arm64" / binary.name
        copied_headers = library.parent / "Headers"
        if digest(library) != digest(binary):
            raise RuntimeError("XCFramework changed the signed Mac library")
        for header in PUBLIC_C_HEADERS:
            if digest(copied_headers / header) != header_hashes[header]:
                raise RuntimeError("Mac public header differs from exact source")
        if (copied_headers / "module.modulemap").read_text() != module_map:
            raise RuntimeError("Mac module map changed")
        if (run("lipo", "-archs", str(library)) != "arm64" or
                minimum_os_from_build(run("vtool", "-show-build", str(library)), "MACOS", supported_platforms=("MACOS",)) != minimum or
                dependencies(library) != linked):
            raise RuntimeError("Mac packaged binary platform or dependencies changed")
        validate_public_c_exports(exported_symbols(library), expected)
        run("codesign", "--verify", "--strict", str(library))
        run("codesign", "--force", "--sign", "-", str(xcframework))
        run("codesign", "--verify", "--strict", str(xcframework))
        destination.mkdir()
        archive = destination / "CLiteRTLM_mac.xcframework.zip"
        zip_directory(xcframework, archive)
        report = {
            "source": source, "platform": "MACOS", "architectures": ["arm64"],
            "minimumOS": minimum, "sourceLibrarySHA256": digest(source_library),
            "binarySHA256": digest(library), "dependencies": linked,
            "publicHeaderSHA256": header_hashes,
            "publicCExports": sorted(expected),
            "archiveSHA256": digest(archive), "modelQueries": 0,
            "macAppAcceptance": False, "canonicalProcessingQualification": False,
            "capacityQualification": False,
        }
        (destination / "SOURCE_COMMIT").write_text(source + "\n")
        (destination / "SHA256SUMS").write_text(f'{digest(archive)}  {archive.name}\n')
        (destination / "APPLE_RUNTIME_MANIFEST.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_archive", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--macos-arm64", action="store_true",
                        help="Package an existing same-source Mac arm64 C API dylib")
    parser.add_argument("--macos-metal-arm64", action="store_true",
                        help="Package same-source Mac arm64 C API with exact pinned Metal runtime closure")
    arguments = parser.parse_args()
    if arguments.macos_arm64 and arguments.macos_metal_arm64:
        parser.error("Choose exactly one Mac packaging mode")
    if arguments.macos_metal_arm64:
        package(arguments.source_archive, arguments.destination, macos_metal_arm64=True)
    elif arguments.macos_arm64:
        package_macos_arm64(arguments.source_archive, arguments.destination)
    else:
        package(arguments.source_archive, arguments.destination)
