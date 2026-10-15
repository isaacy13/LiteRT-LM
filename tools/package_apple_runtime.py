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
import stat
import struct
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
MAC_LIBRARIES = {**LIBRARIES,
    "libLiteRtTopKWebGpuSampler.dylib": "LiteRtTopKWebGpuSampler",
    "libLiteRtWebGpuAccelerator.dylib": "LiteRtWebGpuAccelerator",
    "libwebgpu_dawn.dylib": "WebGpuDawn",
}
MAC_SLICES = {"macos-arm64": ("macos_arm64", "MACOS")}

SLICES = {"ios-arm64": ("ios_arm64", "IOS"),
          "ios-arm64-simulator": ("ios_sim_arm64", "IOSSIMULATOR")}
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


def install_name(name, *, macos=False):
    return f"@rpath/{name}.framework/" + ("Versions/A/" if macos else "") + name


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
    if platform not in {"IOS", "IOSSIMULATOR", "MACOS"}:
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
            "CFBundleSupportedPlatforms": [{"IOS": "iPhoneOS", "IOSSIMULATOR": "iPhoneSimulator", "MACOS": "MacOSX"}[platform]],
            **({"LSMinimumSystemVersion": minimum} if platform == "MACOS" else {})}


def rewrite_linkage(binary, name, libraries=LIBRARIES, *, macos=False):
    # Rewrite the signed copy, then replace its signature when the completed
    # framework is sealed. Early removal can leave signature-alignment padding
    # at the end of __LINKEDIT that older install_name_tool versions reject.
    run("install_name_tool", "-id", install_name(name, macos=macos), str(binary))
    for dependency in dependencies(binary):
        old = dependency.removeprefix("@rpath/")
        if dependency.startswith("@rpath/") and old in libraries:
            run("install_name_tool", "-change", dependency, install_name(libraries[old], macos=macos), str(binary))
        elif macos:
            names = [n for n in ["CLiteRTLM", *libraries.values()]
                     if dependency == install_name(n)]
            if names:
                run("install_name_tool", "-change", dependency,
                    install_name(names[0], macos=True), str(binary))


def macos_links(name):
    return {name: "Versions/Current/" + name, "Headers": "Versions/Current/Headers",
            "Modules": "Versions/Current/Modules", "Resources": "Versions/Current/Resources",
            "Versions/Current": "A"}


def validate_macos_layout(framework, name):
    expected = macos_links(name)
    links = {p.relative_to(framework).as_posix(): p.readlink().as_posix()
             for p in framework.rglob("*") if p.is_symlink()}
    if links != expected or {p.name for p in framework.iterdir()} != {name, "Headers", "Modules", "Resources", "Versions"}:
        raise RuntimeError("Incorrect versioned Mac framework links/layout")
    if {p.name for p in (framework / "Versions").iterdir()} != {"A", "Current"}:
        raise RuntimeError("Unexpected Mac framework version")
    version = framework / "Versions/A"
    if not (version / name).is_file() or (version / name).is_symlink():
        raise RuntimeError("Missing real versioned Mac executable")
    if not (version / "Resources/Info.plist").is_file():
        raise RuntimeError("Missing versioned Mac Info.plist")
    for relative, target in expected.items():
        path = framework / relative
        if not path.resolve(strict=True).is_relative_to(version.resolve(strict=True)):
            raise RuntimeError("Escaping/broken Mac framework link")
    return links


def create_macos_framework(work, name, original, metadata):
    framework = work / "macos-arm64" / (name + ".framework")
    version = framework / "Versions/A"
    for directory in ["Headers", "Modules", "Resources"]:
        (version / directory).mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, version / name)
    (version / "Resources/Info.plist").write_bytes(plistlib.dumps(metadata))
    if name == "CLiteRTLM":
        for header in PUBLIC_C_HEADERS:
            shutil.copy2(ROOT / "c" / header, version / "Headers" / header)
        headers = "".join('  header "' + h + '"\n' for h in PUBLIC_C_HEADERS)
    else:
        (version / "Headers/RuntimeDependency.h").write_text("// APIs are bound internally by CLiteRTLM.\n")
        headers = '  umbrella header "RuntimeDependency.h"\n'
    (version / "Modules/module.modulemap").write_text(
        "framework module " + name + " {\n" + headers + "  export *\n}\n")
    for relative, target in macos_links(name).items():
        (framework / relative).symlink_to(target)
    validate_macos_layout(framework, name)
    return framework


def native_content_identity(data, name):
    """Compare all native payload and load commands except explicit linkage/signing fields."""
    if len(data) < 32 or struct.unpack_from("<II", data) != (0xfeedfacf, 0x100000c):
        raise RuntimeError("Expected thin arm64 Mach-O")
    header = struct.unpack_from("<8I", data)
    count, size = header[4:6]
    offset = 32; commands = []; first_section = len(data); signature = None; linkedit = None
    mapping = {install_name(n): install_name(n, macos=True) for n in ["CLiteRTLM", *MAC_LIBRARIES.values()]}
    mapping.update({"@rpath/" + lib: install_name(n, macos=True) for lib, n in MAC_LIBRARIES.items()})
    for _ in range(count):
        command, length = struct.unpack_from("<II", data, offset)
        if length < 8 or offset + length > 32 + size or 32 + size > len(data):
            raise RuntimeError("Malformed native load command")
        raw = bytearray(data[offset:offset + length])
        if command == 0x1d:
            if signature is not None: raise RuntimeError("Duplicate native signature")
            signature = struct.unpack_from("<II", raw, 8)
        elif command in {0xc, 0xd, 0x80000018, 0x8000001f, 0x20, 0x80000023}:
            start = struct.unpack_from("<I", raw, 8)[0]
            pieces = bytes(raw[start:]).split(b"\0", 1)
            if start != 24 or len(pieces) != 2 or any(pieces[1]):
                raise RuntimeError("Unexpected native linkage command layout")
            linked = pieces[0].decode("utf-8")
            linked = install_name(name, macos=True) if command == 0xd else mapping.get(linked, linked)
            commands.append((command, linked, struct.unpack_from("<III", raw, 12)))
        else:
            if command == 0x19:
                sections = struct.unpack_from("<I", raw, 64)[0]
                for index in range(sections):
                    section = 72 + index * 80
                    position = struct.unpack_from("<I", raw, section + 48)[0]
                    section_size = struct.unpack_from("<Q", raw, section + 40)[0]
                    if position and section_size: first_section = min(first_section, position)
                if raw[8:24].rstrip(b"\0") == b"__LINKEDIT":
                    if linkedit is not None: raise RuntimeError("Duplicate native LINKEDIT")
                    linkedit = struct.unpack_from("<QQQ", raw, 32)
                    struct.pack_into("<Q", raw, 32, 0)  # signature allocation changes vmsize
                    struct.pack_into("<Q", raw, 48, 0)  # and final filesize only
            commands.append((command, bytes(raw).hex()))
        offset += length
    if offset != 32 + size or signature is None or signature[0] + signature[1] != len(data):
        raise RuntimeError("Expected terminal native code signature")
    if linkedit is None or linkedit[1] + linkedit[2] != len(data) or linkedit[0] < linkedit[2]:
        raise RuntimeError("Native LINKEDIT must end with the signature and cover its file bytes")
    if first_section < offset or first_section > signature[0] or any(data[offset:first_section]):
        raise RuntimeError("Unexpected native header padding/layout")
    return {"header": header[:4] + header[6:], "commands": commands,
            "payloadSHA256": hashlib.sha256(data[first_section:signature[0]]).hexdigest(),
            "payloadBytes": signature[0] - first_section, "firstSectionOffset": first_section}


def require_same_native_content(before, after, name):
    original = native_content_identity(before, name)
    if native_content_identity(after, name) != original:
        raise RuntimeError("Native code/data or unrelated load metadata changed: " + name)
    return hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()


def make_macos_xcframework(work, name, original, metadata):
    """The single native packaging path shared by full builds and exact repackaging."""
    framework = create_macos_framework(work, name, original, metadata)
    binary = framework / name
    rewrite_linkage(binary, name, MAC_LIBRARIES, macos=True)
    run("codesign", "--force", "--sign", "-", str(framework))
    content = require_same_native_content(original.read_bytes(), binary.read_bytes(), name)
    target = work / (name + ".xcframework")
    run("xcodebuild", "-create-xcframework", "-framework", str(framework), "-output", str(target))
    validate_macos_layout(target / "macos-arm64" / (name + ".framework"), name)
    return target, content


def zip_directory(directory, archive, *, macos=False):
    if macos:
        for framework in directory.glob("macos-arm64/*.framework"):
            validate_macos_layout(framework, framework.stem)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(directory.rglob("*")):
            if macos and path.is_symlink():
                info = zipfile.ZipInfo(path.relative_to(directory.parent).as_posix())
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                bundle.writestr(info, path.readlink().as_posix().encode("utf-8"))
            elif path.is_file():
                bundle.write(path, path.relative_to(directory.parent))


def extract_macos_archive(archive, destination, name, *, versioned):
    """Only the exact framework link graph is permitted; no generic ZIP symlinks."""
    prefix = name + ".xcframework/"
    result = destination / (name + ".xcframework")
    if result.exists() or result.is_symlink():
        raise RuntimeError("Never overwrite a Mac archive extraction")
    if destination.is_symlink():
        raise RuntimeError("Mac archive extraction root cannot be a symlink")
    expected_links = {prefix + "macos-arm64/" + name + ".framework/" + p: t
                      for p, t in macos_links(name).items()} if versioned else {}
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if sum(e.file_size for e in entries) > 256 * 1024 * 1024:
            raise RuntimeError("Mac archive exceeds finite size bound")
        names = [e.filename.rstrip("/") for e in entries]
        if len(set(names)) != len(names):
            raise RuntimeError("Duplicate Mac archive entry")
        links = {}
        for entry in entries:
            path = Path(entry.filename)
            if (path.is_absolute() or not entry.filename.startswith(prefix)
                    or any(p in {"..", "."} for p in entry.filename.split("/") if p)
                    or "\\" in entry.filename):
                raise RuntimeError("Unsafe Mac archive path")
            mode = entry.external_attr >> 16
            if stat.S_ISLNK(mode):
                target = bundle.read(entry).decode("utf-8")
                if expected_links.get(entry.filename) != target:
                    raise RuntimeError("Unexpected/escaping Mac archive symlink")
                links[entry.filename] = target
            elif stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise RuntimeError("Unsupported Mac archive member type")
        if links != expected_links:
            raise RuntimeError("Missing Mac archive symlinks")
        for entry in entries:
            if entry.filename in links:
                continue
            if any(entry.filename.startswith(link + "/") for link in links):
                raise RuntimeError("Mac archive writes through symlink")
            path = destination / entry.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            if entry.is_dir(): path.mkdir(exist_ok=True)
            else:
                with path.open("xb") as output: output.write(bundle.read(entry))
                path.chmod((entry.external_attr >> 16) & 0o777 or 0o644)
        for relative, target in links.items():
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(target)
    if versioned:
        validate_macos_layout(result / "macos-arm64" / (name + ".framework"), name)
    return result


def write_macos_archives(frameworks, destination, work):
    archives = {}
    for name, framework in frameworks.items():
        archive = destination / (("CLiteRTLM_mac" if name == "CLiteRTLM" else name) + ".xcframework.zip")
        zip_directory(framework, archive, macos=True)
        roundtrip = extract_macos_archive(archive, work / "roundtrip", name, versioned=True)
        def identity(root):
            return {p.relative_to(root).as_posix(): {"link": p.readlink().as_posix()} if p.is_symlink()
                    else {"sha256": digest(p), "bytes": p.stat().st_size}
                    for p in root.rglob("*") if p.is_symlink() or p.is_file()}
        if identity(roundtrip) != identity(framework):
            raise RuntimeError("Versioned archive roundtrip differs")
        archives[name] = {"file": archive.name, "sha256": digest(archive), "bytes": archive.stat().st_size}
    return archives


def inspect(frameworks, *, macos_arm64=False):
    slices = MAC_SLICES if macos_arm64 else SLICES
    if (set(GEMMA_ABI_OWNERS) != GEMMA_PROVIDER_FUNCTIONS
            or set(GEMMA_ABI_OWNERS.values()) != {"CLiteRTLM", "GemmaModelConstraintProvider"}):
        raise RuntimeError("Invalid native tool constraint provider ABI ownership")
    results = []
    allowed = {install_name(name, macos=macos_arm64) for name in frameworks}
    expected_c_exports = public_c_exports(ROOT / "c") if "CLiteRTLM" in frameworks else set()
    for identifier, (_, platform) in slices.items():
        for name, xcframework in frameworks.items():
            info = plistlib.loads((xcframework / "Info.plist").read_bytes())
            entries = info["AvailableLibraries"]
            if len(entries) != len(slices) or {entry["LibraryIdentifier"] for entry in entries} != set(slices):
                raise RuntimeError(f"Incorrect slice coverage in {xcframework}")
            entry = next(entry for entry in entries if entry["LibraryIdentifier"] == identifier)
            expected_variant = "simulator" if platform == "IOSSIMULATOR" else None
            if (entry["SupportedArchitectures"] != ["arm64"] or entry["SupportedPlatform"] != ("macos" if macos_arm64 else "ios")
                    or entry.get("SupportedPlatformVariant") != expected_variant):
                raise RuntimeError(f"Incorrect declared platform in {xcframework}/{identifier}")
            framework = xcframework / identifier / entry["LibraryPath"]
            binary = framework / name
            links = validate_macos_layout(framework, name) if macos_arm64 else None
            metadata = plistlib.loads((framework / ("Resources/Info.plist" if macos_arm64 else "Info.plist")).read_bytes())
            if metadata.get("CFBundleExecutable") != name or metadata.get("CFBundlePackageType") != "FMWK":
                raise RuntimeError(f"Invalid framework metadata at {framework}")
            if run("lipo", "-archs", str(binary)) != "arm64":
                raise RuntimeError(f"Wrong architecture at {binary}")
            minimum = minimum_os_from_build(run("vtool", "-show-build", str(binary)), platform)
            validate_minimum_os_metadata(minimum, metadata.get("MinimumOSVersion"))
            if macos_arm64 and metadata.get("LSMinimumSystemVersion") != metadata.get("MinimumOSVersion"):
                raise RuntimeError("Incoherent Mac minimum OS metadata")
            linked = dependencies(binary)
            own_id = run("otool", "-D", str(binary)).splitlines()[1:]
            if own_id != [install_name(name, macos=macos_arm64)]:
                raise RuntimeError(f"Incorrect install name at {binary}: {own_id}")
            validate_dependencies("CLiteRTLM_mac" if macos_arm64 and name == "CLiteRTLM" else name, linked, allowed)
            if macos_arm64 and name == "CLiteRTLM":
                required = {install_name(n, macos=True) for n in ("LiteRtRuntime", "GemmaModelConstraintProvider")}
                if not required <= set(linked):
                    raise RuntimeError("Mac core must strongly link its matching runtime and Gemma provider")
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
                            "requiredGemmaABIExports": sorted(required_gemma_exports),
                            **({"definedExports": sorted(symbols), "bundleLayout": "Versions/A",
                                "symbolicLinks": links} if macos_arm64 else {})})
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
                  "packagedDependencyClosure": True, "frameworks": results, "archives": archives,
                  "gemmaABIExportOwners": GEMMA_ABI_OWNERS}
        (destination / "APPLE_RUNTIME_MANIFEST.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


def pinned_macos_libraries():
    """Read exact fork LFS identities; no other SDK's libraries are admissible."""
    results = []
    for library in MAC_LIBRARIES:
        relative = "prebuilt/macos_arm64/" + library
        path = ROOT / relative
        pointer = run("git", "-C", str(ROOT), "show", "HEAD:" + relative)
        match = re.fullmatch(r"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([1-9][0-9]*)", pointer)
        if match is None or not path.is_file() or path.is_symlink():
            raise RuntimeError("Missing/invalid committed Mac LFS object: " + relative)
        if path.stat().st_size != int(match[2]) or digest(path) != match[1]:
            raise RuntimeError("Pinned Mac LFS content differs: " + relative)
        if run("lipo", "-archs", str(path)) != "arm64":
            raise RuntimeError("Mac development dependency must be thin arm64: " + relative)
        minimum = minimum_os_from_build(run("vtool", "-show-build", str(path)), "MACOS")
        results.append({"library": relative, "sha256": match[1], "bytes": int(match[2]), "minimumOS": minimum})
    return results


def resolve_macos_bazel_output(output_list):
    # cquery --output=files resolves the existing rule's actual declared output.
    # Never guess an output name or search for another build's binary.
    rows = output_list.read_text().splitlines()
    if len(rows) != 1:
        raise RuntimeError("Expected exactly one declared Mac dylib output")
    relative = Path(rows[0])
    if (relative.is_absolute() or ".." in relative.parts or not relative.parts
            or relative.parts[0] != "bazel-out" or relative.suffix != ".dylib"):
        raise RuntimeError("Unexpected declared Mac dylib output path")
    path = ROOT / relative
    if not path.is_file():
        raise RuntimeError("Declared Mac dylib output is missing")
    return path


def package_macos_arm64(output_list, destination):
    """Development artifact only. Never replaces the official universal package."""
    if destination.exists():
        raise RuntimeError("Never replace a retained package")
    source_binary = resolve_macos_bazel_output(output_list)
    source = run("git", "-C", str(ROOT), "rev-parse", "HEAD")
    if run("git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Package only committed source")
    if upstream_release_version() != "0.18.0":
        raise RuntimeError("Mac development producer is scoped to v0.18.0")
    pins = pinned_macos_libraries()
    original_sha = digest(source_binary)
    with retained_package_work(source_binary, destination, source, preserve_symlinks=True) as work:
        binaries = {"CLiteRTLM": source_binary, **{name: ROOT / "prebuilt/macos_arm64" / library
                     for library, name in MAC_LIBRARIES.items()}}
        frameworks = {}
        content_proofs = {}
        for name, original in binaries.items():
            frameworks[name], content_proofs[name] = make_macos_xcframework(
                work, name, original, framework_metadata(name, "MACOS", original))
        results = inspect(frameworks, macos_arm64=True)
        # Original compiler output and LFS objects must survive packaging unchanged.
        if digest(source_binary) != original_sha or pinned_macos_libraries() != pins:
            raise RuntimeError("Mac build or pinned dependencies changed during packaging")
        destination.mkdir()
        archives = write_macos_archives(frameworks, destination, work)
        report = {"source": source, "upstreamVersion": "0.18.0", "developmentOnly": True,
                  "architectures": ["arm64"], "intelQualified": False,
                  "productionDefaultChanged": False, "modelQueries": 0,
                  "appAcceptance": False, "modelAcceptance": False,
                  "target": "//swift:CLiteRTLM_dylib", "declaredOutput": source_binary.relative_to(ROOT).as_posix(),
                  "sourceBinarySHA256": original_sha, "prebuiltLFSObjects": pins,
                  "packagedDependencyClosure": True, "frameworks": results, "archives": archives,
                  "gemmaABIExportOwners": GEMMA_ABI_OWNERS,
                  "nativeContentIdentitySHA256": content_proofs}
        (destination / "SOURCE_COMMIT").write_text(source + "\n")
        (destination / "SHA256SUMS").write_text("".join(
            f'{v["sha256"]}  {v["file"]}\n' for v in archives.values()))
        (destination / "APPLE_RUNTIME_MANIFEST.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


REPACKAGE_MANIFEST_SHA256 = "cc6bd22c595b75cc8b05281602c4707795431739a7b4005d2c7fdd6411d6430b"
REPACKAGE_COMPILED_SOURCE = "796b2281a5848221a3710cd528b23b1b5ac66b34"


def validate_repackage_source(manifest, manifest_sha, revision, changed):
    allowed = {"tools/package_apple_runtime.py", "tools/package_apple_runtime_test.py"}
    if (manifest_sha != REPACKAGE_MANIFEST_SHA256 or manifest.get("source") != REPACKAGE_COMPILED_SOURCE
            or set(changed) - allowed or re.fullmatch(r"[0-9a-f]{40}", revision) is None):
        raise RuntimeError("Repackage requires exact compiled source and packaging-only revision")
    expected = {"CLiteRTLM", *MAC_LIBRARIES.values()}
    if (set(manifest["archives"]) != expected or {r["name"] for r in manifest["frameworks"]} != expected
            or manifest["architectures"] != ["arm64"] or manifest["upstreamVersion"] != "0.18.0"
            or manifest["intelQualified"] or manifest["productionDefaultChanged"]):
        raise RuntimeError("Incorrect original whole-SDK repackage closure")


def repackage_macos_arm64(artifact, destination):
    """Repackage one exact compiled artifact; never invokes Bazel or substitutes binaries."""
    if destination.exists(): raise RuntimeError("Never replace a retained package")
    manifest_path = artifact / "APPLE_RUNTIME_MANIFEST.json"
    original_manifest = json.loads(manifest_path.read_bytes())
    revision = run("git", "-C", str(ROOT), "rev-parse", "HEAD")
    if run("git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Package only committed source")
    changed = run("git", "-C", str(ROOT), "diff", "--name-only", REPACKAGE_COMPILED_SOURCE, revision).splitlines()
    validate_repackage_source(original_manifest, digest(manifest_path), revision, changed)
    input_archives = {}
    for name, row in original_manifest["archives"].items():
        archive = artifact / row["file"]
        if archive.is_symlink() or not archive.is_file() or archive.stat().st_size != row["bytes"] or digest(archive) != row["sha256"]:
            raise RuntimeError("Original compiled archive changed: " + name)
        input_archives[name] = archive
    with retained_package_work(manifest_path, destination, revision, preserve_symlinks=True) as work:
        frameworks = {}; content_proofs = {}; input_binaries = {}
        for row in original_manifest["frameworks"]:
            name = row["name"]
            extracted = extract_macos_archive(input_archives[name], work / "inputs", name, versioned=False)
            original_framework = extracted / "macos-arm64" / (name + ".framework")
            original = original_framework / name
            if digest(original) != row["sha256"]:
                raise RuntimeError("Original compiled binary changed: " + name)
            run("codesign", "--verify", "--strict", str(original_framework))
            input_binaries[name] = digest(original)
            metadata = plistlib.loads((original_framework / "Info.plist").read_bytes())
            metadata["LSMinimumSystemVersion"] = metadata["MinimumOSVersion"]
            frameworks[name], content_proofs[name] = make_macos_xcframework(work, name, original, metadata)
        results = inspect(frameworks, macos_arm64=True)
        for row in results:
            previous = next(r for r in original_manifest["frameworks"] if r["name"] == row["name"])
            if row["definedExports"] != previous["definedExports"] or row["minimumOS"] != previous["minimumOS"]:
                raise RuntimeError("Repackaging changed native API or platform")
        if digest(manifest_path) != REPACKAGE_MANIFEST_SHA256 or any(
                digest(path) != original_manifest["archives"][name]["sha256"] for name, path in input_archives.items()):
            raise RuntimeError("Original artifact changed during packaging")
        destination.mkdir(); archives = write_macos_archives(frameworks, destination, work)
        report = {**original_manifest, "source": REPACKAGE_COMPILED_SOURCE,
                  "compiledSource": REPACKAGE_COMPILED_SOURCE, "packagingRevision": revision,
                  "compilerOutputRebuilt": False, "repackagedFromManifestSHA256": REPACKAGE_MANIFEST_SHA256,
                  "originalArchives": original_manifest["archives"], "originalBinarySHA256": input_binaries,
                  "nativeContentIdentitySHA256": content_proofs, "frameworks": results, "archives": archives,
                  "appAcceptance": False, "modelAcceptance": False, "nativeHostTestsRerun": False,
                  "originalCompilerSourceHostEvidence": {"workflowRunID": 38696736097,
                      "testExecutions": 341, "newPackageAcceptanceTransferred": False}}
        (destination / "SOURCE_COMMIT").write_text(REPACKAGE_COMPILED_SOURCE + "\n")
        (destination / "PACKAGING_COMMIT").write_text(revision + "\n")
        (destination / "SHA256SUMS").write_text("".join(f'{r["sha256"]}  {r["file"]}\n' for r in archives.values()))
        (destination / "APPLE_RUNTIME_MANIFEST.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macos-arm64", action="store_true", help="Source-built arm64 development SDK; input is cquery output file list")
    parser.add_argument("--repackage-macos-artifact", action="store_true", help="Repackage the exact796b compiled whole SDK as versioned Mac frameworks; no compiler")
    parser.add_argument("source_archive", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    if arguments.macos_arm64 and arguments.repackage_macos_artifact:
        parser.error("Select full Mac build packaging or exact artifact repackaging")
    if arguments.repackage_macos_artifact:
        repackage_macos_arm64(arguments.source_archive, arguments.destination)
    elif arguments.macos_arm64:
        package_macos_arm64(arguments.source_archive, arguments.destination)
    else:
        package(arguments.source_archive, arguments.destination)
