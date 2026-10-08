#!/usr/bin/env python3
"""Package the pinned iOS runtime dependency graph as explicit XCFrameworks.

Each runtime has its own framework bundle, install name, and SwiftPM binary
artifact. No nested framework or standalone dylib is placed in an iOS app.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import plistlib
import re
import shutil
import shlex
import subprocess
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


def install_name(name):
    return f"@rpath/{name}.framework/{name}"


def dependencies(binary):
    return [line.strip().split(" (compatibility")[0]
            for line in run("otool", "-L", str(binary)).splitlines()[1:] if line.strip()]


def exported_symbols(binary):
    return {line.split()[-1].removeprefix("_")
            for line in run("nm", "-gU", str(binary)).splitlines() if line.split()}


def validate_dependencies(name, linked, allowed):
    missing = [dependency for dependency in linked
               if not dependency.startswith(("/usr/lib/", "/System/Library/")) and dependency not in allowed]
    if missing:
        raise RuntimeError(f"Unpackaged dependencies for {name}: {missing}")
    if name == "CLiteRTLM":
        required = {install_name(dependency) for dependency in LIBRARIES.values()}
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


def minimum_os_from_build(build, platform):
    if platform not in {"IOS", "IOSSIMULATOR"}:
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
    minimum = minimum_os_from_build(run("vtool", "-show-build", str(binary)), platform)
    return {"CFBundleExecutable": name, "CFBundleIdentifier": f"com.guideai.litert.{name}",
            "CFBundleName": name, "CFBundlePackageType": "FMWK", "CFBundleVersion": "1",
            "CFBundleShortVersionString": upstream_release_version(), "MinimumOSVersion": minimum,
            "CFBundleSupportedPlatforms": ["iPhoneSimulator" if platform == "IOSSIMULATOR" else "iPhoneOS"]}


def rewrite_linkage(binary, name):
    # Rewrite the signed copy, then replace its signature when the completed
    # framework is sealed. Early removal can leave signature-alignment padding
    # at the end of __LINKEDIT that older install_name_tool versions reject.
    run("install_name_tool", "-id", install_name(name), str(binary))
    for dependency in dependencies(binary):
        old = dependency.removeprefix("@rpath/")
        if dependency.startswith("@rpath/") and old in LIBRARIES:
            run("install_name_tool", "-change", dependency, install_name(LIBRARIES[old]), str(binary))


def zip_directory(directory, archive):
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(directory.parent))


def inspect(frameworks):
    results = []
    allowed = {install_name(name) for name in frameworks}
    for identifier, (_, platform) in SLICES.items():
        for name, xcframework in frameworks.items():
            info = plistlib.loads((xcframework / "Info.plist").read_bytes())
            entries = info["AvailableLibraries"]
            if len(entries) != 2 or {entry["LibraryIdentifier"] for entry in entries} != set(SLICES):
                raise RuntimeError(f"Incorrect slice coverage in {xcframework}")
            entry = next(entry for entry in entries if entry["LibraryIdentifier"] == identifier)
            expected_variant = "simulator" if platform == "IOSSIMULATOR" else None
            if (entry["SupportedArchitectures"] != ["arm64"] or entry["SupportedPlatform"] != "ios"
                    or entry.get("SupportedPlatformVariant") != expected_variant):
                raise RuntimeError(f"Incorrect declared platform in {xcframework}/{identifier}")
            framework = xcframework / identifier / entry["LibraryPath"]
            binary = framework / name
            metadata = plistlib.loads((framework / "Info.plist").read_bytes())
            if metadata.get("CFBundleExecutable") != name or metadata.get("CFBundlePackageType") != "FMWK":
                raise RuntimeError(f"Invalid framework metadata at {framework}")
            if run("lipo", "-archs", str(binary)) != "arm64":
                raise RuntimeError(f"Wrong architecture at {binary}")
            minimum = minimum_os_from_build(run("vtool", "-show-build", str(binary)), platform)
            validate_minimum_os_metadata(minimum, metadata.get("MinimumOSVersion"))
            linked = dependencies(binary)
            own_id = run("otool", "-D", str(binary)).splitlines()[1:]
            if own_id != [install_name(name)]:
                raise RuntimeError(f"Incorrect install name at {binary}: {own_id}")
            validate_dependencies(name, linked, allowed)
            symbols = exported_symbols(binary)
            if name == "CLiteRTLM":
                for header in ("engine.h", "conversation.h", "embedding_engine.h", "capabilities.h"):
                    if digest(framework / "Headers" / header) != digest(ROOT / "c" / header):
                        raise RuntimeError(f"C header differs from source at {framework}/{header}")
                if "litert_lm_conversation_send_message_stream" not in symbols:
                    raise RuntimeError(f"Missing native conversation API at {binary}")
            elif name == "LiteRtRuntime" and "kLiteRtRuntimeBuiltin" not in symbols:
                raise RuntimeError(f"Missing native runtime ABI at {binary}")
            elif name == "LiteRtMetalAccelerator" and "LiteRtAcceleratorImpl" not in symbols:
                raise RuntimeError(f"Missing pinned Metal accelerator definition at {binary}")
            elif name == "LiteRtTopKMetalSampler" and not SAMPLER_FUNCTIONS <= symbols:
                raise RuntimeError(f"Missing pinned Metal sampler functions at {binary}")
            elif name == "GemmaModelConstraintProvider" and "LiteRtLmGemmaModelConstraintProvider_Create" not in symbols:
                raise RuntimeError(f"Missing native tool constraint provider at {binary}")
            run("codesign", "--verify", "--strict", str(framework))
            if list(framework.rglob("*.dylib")) or list(framework.glob("Frameworks/*")):
                raise RuntimeError(f"Unsupported nested runtime payload at {framework}")
            results.append({"name": name, "slice": identifier, "platform": platform,
                            "sha256": digest(binary), "dependencies": linked,
                            "minimumOS": minimum, "frameworkMinimumOSVersion": metadata["MinimumOSVersion"]})
    return results


@contextmanager
def retained_package_work(source_archive, destination, source):
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
            shutil.copytree(work, failure_directory)
            (failure_directory / "FAILURE.json").write_text(json.dumps({
                "source": source, "sourceArchiveSHA256": digest(source_archive),
                "artifactAcceptance": False, "modelQueries": 0,
                "errorType": type(error).__name__, "error": str(error),
            }, indent=2) + "\n")
            raise


def package(source_archive, destination):
    if destination.exists():
        raise RuntimeError("Never replace a retained package")
    verify()
    source = run("git", "-C", str(ROOT), "rev-parse", "HEAD")
    if run("git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Package only committed source")
    with retained_package_work(source_archive, destination, source) as work:
        with zipfile.ZipFile(source_archive) as bundle:
            if any(Path(member.filename).is_absolute() or ".." in Path(member.filename).parts
                   for member in bundle.infolist()):
                raise RuntimeError("Unsafe framework archive path")
            bundle.extractall(work)
        capi = work / "CLiteRTLM.xcframework"
        frameworks = {"CLiteRTLM": capi}
        for identifier, (directory, platform) in SLICES.items():
            framework = capi / identifier / "CLiteRTLM.framework"
            rewrite_linkage(framework / "CLiteRTLM", "CLiteRTLM")
            run("codesign", "--force", "--sign", "-", str(framework))
            for library, name in LIBRARIES.items():
                framework = work / identifier / f"{name}.framework"
                (framework / "Headers").mkdir(parents=True)
                (framework / "Modules").mkdir()
                shutil.copy2(ROOT / "prebuilt" / directory / library, framework / name)
                (framework / "Info.plist").write_bytes(plistlib.dumps(
                    framework_metadata(name, platform, framework / name)))
                (framework / "Headers" / "RuntimeDependency.h").write_text("// APIs are bound internally by CLiteRTLM.\n")
                (framework / "Modules" / "module.modulemap").write_text(
                    f'framework module {name} {{\n  umbrella header "RuntimeDependency.h"\n  export *\n}}\n')
                rewrite_linkage(framework / name, name)
                run("codesign", "--force", "--sign", "-", str(framework))
        for name in LIBRARIES.values():
            target = work / f"{name}.xcframework"
            command = ["xcodebuild", "-create-xcframework"]
            for identifier in SLICES:
                command += ["-framework", str(work / identifier / f"{name}.framework")]
            run(*command, "-output", str(target))
            frameworks[name] = target
        results = inspect(frameworks)
        destination.mkdir()
        archives = {}
        for name, framework in frameworks.items():
            archive = destination / f"{name}.xcframework.zip"
            zip_directory(framework, archive)
            archives[name] = {"file": archive.name, "sha256": digest(archive)}
        (destination / "SOURCE_COMMIT").write_text(source + "\n")
        (destination / "SHA256SUMS").write_text("".join(
            f'{entry["sha256"]}  {entry["file"]}\n' for entry in archives.values()))
        report = {"source": source, "modelQueries": 0, "phoneAcceptance": False,
                  "packagedDependencyClosure": True, "frameworks": results, "archives": archives}
        (destination / "APPLE_RUNTIME_MANIFEST.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_archive", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    package(arguments.source_archive, arguments.destination)
