#!/usr/bin/env python3
"""Build an Ambarella CVFlow/FlexiDAG from an ONNX model.

The Ambarella toolchain is shipped in the CVTools container rather than as a
Python wheel in this repository.  This helper creates an ADK work directory,
configures graph surgery and CVFlow, and invokes the ADK ``cvb`` target.  It
also works without a container when the CVTools tools are installed locally.

The generated files are copied to the requested output directory.  The helper
does not implement a second quantizer: Ambarella's DRA/CNNGen quantization is
performed by ``prepare.py`` as part of the CVFlow build.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}


def _root_from_config(config_path):
    path = Path(config_path).resolve()
    # configs/<platform>/compile.yaml -> mode -> modes -> repository root
    return path.parents[3]


def _load_config(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _resolve(root, value):
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _model_candidates(root, mode, configured, prefer_headcut):
    if configured and str(configured).lower() != "auto":
        return [_resolve(root, configured)]
    model_dir = root / "modes" / mode / "model"
    candidates = sorted(model_dir.glob("*.onnx"))
    if prefer_headcut:
        headcut = [p for p in candidates if "_headcut_raw" in p.name]
        if headcut:
            candidates = headcut
    return candidates


def _load_onnx(path):
    try:
        import onnx
    except ImportError as error:
        raise SystemExit(
            "onnx is required to inspect the Ambarella model; run this "
            "through STATLAS_PYTHON or install onnx") from error
    return onnx.load(str(path), load_external_data=False)


def _shape(value):
    result = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.dim_value:
            result.append(int(dim.dim_value))
        elif dim.dim_param:
            raise SystemExit("dynamic ONNX dimensions are not supported: {}".format(
                value.name))
        else:
            raise SystemExit("unknown ONNX dimension in input {}".format(value.name))
    return result


def _select_model(root, mode, config):
    candidates = _model_candidates(
        root, mode, config.get("model"),
        bool(config.get("prefer_headcut", True)))
    candidates = [path for path in candidates if path.is_file()]
    if len(candidates) != 1:
        names = ", ".join(str(p) for p in candidates) or "none"
        raise SystemExit(
            "Ambarella requires exactly one configured ONNX model; found: {}. "
            "Set model: in configs/ambarella/compile.yaml.".format(names))
    return candidates[0]


def _prepare_headcut_model(root, mode, model, prefer_headcut, dry_run):
    """Resolve or create the head-cut model requested by the config.

    ``prefer_headcut`` applies to explicitly configured models as well as
    ``model: auto``. Models without a supported YOLO head are left unchanged.
    """
    if not prefer_headcut or "_headcut_raw" in model.stem:
        return model
    headcut = model.with_name(model.stem + "_headcut_raw.onnx")
    if headcut.is_file():
        print("Ambarella head-cut model:", headcut)
        return headcut

    script = root / "common" / "tools" / "cut_yolo_head.py"
    if not script.is_file():
        raise SystemExit("head-cut helper not found: {}".format(script))
    print("Ambarella head-cut: generating {} from {}".format(headcut, model))
    if dry_run:
        # The source model is still needed for dry-run graph inspection.
        return model
    command = [sys.executable, str(script), "--input_model", str(model),
               "--output_model", str(headcut)]
    subprocess.run(command, cwd=str(root), check=True)
    if headcut.is_file():
        return headcut
    print("Ambarella head-cut: no supported head found; using original model")
    return model


def _run(command, cwd=None, env=None, dry_run=False):
    print("+", " ".join(shlex.quote(str(item)) for item in command), flush=True)
    if not dry_run:
        subprocess.run([str(item) for item in command], cwd=str(cwd) if cwd else None,
                       env=env, check=True)


def _container_command(container, command, cwd, env):
    assignments = []
    # Do not forward the host's PATH wholesale.  In this setup it contains
    # the host Anaconda ``python3`` ahead of the container's Python, which
    # lacks the CVTools dependencies (onnx/termcolor).  Keep the container
    # system paths and the CVTools wrappers instead.
    container_path = ":".join(filter(None, [
        "/opt/cvtools/cv7/local/bin",
        "/opt/cvtools/cv7/tv2/exe",
        "/opt/cvtools/cv72/local/bin",
        "/opt/cvtools/cv72/tv2/exe",
        "/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin",
        "/sbin", "/bin",
    ]))
    env = dict(env)
    env["PATH"] = container_path
    for key in ("PROJECT", "PATH", "PYTHONPATH", "AMBARELLA_ADK_PATH"):
        value = env.get(key)
        if value:
            assignments.append("{}={}".format(key, shlex.quote(value)))
    # RunContainer.sh starts an interactive shell, but ``podman exec`` starts
    # a fresh non-interactive shell.  ADK needs the chip-specific variables
    # (AMBA_ROOT, TV2_CONFIG, and tool wrappers) supplied by env_set.sh.
    chip = str(env.get("PROJECT", "cv7"))
    init = "source /opt/cvtools/env/env_set.sh {}".format(
        shlex.quote(chip))
    shell = "{} && cd {} && {} && {}".format(
        init, shlex.quote(str(cwd)), " ".join(assignments) or ":",
        " ".join(shlex.quote(str(item)) for item in command))
    return ["podman", "exec", "-i", container, "bash", "-lc", shell]


def _require_running_container(container):
    """Fail early with a useful message when the configured container is stopped."""
    try:
        result = subprocess.run(
            ["podman", "container", "inspect", "--format",
             "{{.State.Status}}", container],
            capture_output=True, text=True, check=False)
    except FileNotFoundError as error:
        raise SystemExit(
            "podman is required for Ambarella container builds; install "
            "Podman or unset AMBARELLA_CONTAINER to use local CVTools") from error
    if result.returncode != 0:
        detail = result.stderr.strip()
        suffix = ": {}".format(detail) if detail else ""
        raise SystemExit(
            "Ambarella container '{}' was not found or cannot be inspected{}. "
            "Run ./setup_conda_envs.sh --ambarella-only and source env.sh.".format(
                container, suffix))
    state = result.stdout.strip()
    if state != "running":
        raise SystemExit(
            "Ambarella container '{}' is {}, not running. Run "
            "./setup_conda_envs.sh --ambarella-only, then source env.sh.".format(
                container, state or "unknown"))


def _copy_template(template, workdir, dry_run):
    if dry_run:
        print("template:", template)
        return
    if not template.is_dir():
        raise SystemExit(
            "Ambarella template directory not found: {}. Set "
            "AMBARELLA_TEMPLATE_DIR or template_dir in compile.yaml.".format(
                template))
    if workdir.exists():
        shutil.rmtree(workdir)
    # Keep the template model catalog: its Makefile invokes model_info.py
    # while parsing defaults, even when all effective model variables are
    # supplied on the command line.  The actual requested ONNX is copied
    # below and remains the source used by CVFlow.
    shutil.copytree(template, workdir, ignore=shutil.ignore_patterns(
        ".git", "out", "eval", "test_images"))


def _sample_calibration_images(dataset, workdir, input_cfg, dry_run):
    images = sorted(path for path in dataset.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    sample_count = int(input_cfg.get("sample_count", 0) or 0)
    if sample_count < 0:
        raise SystemExit("Ambarella input.sample_count must be non-negative")
    if sample_count > 0:
        images = images[:sample_count]
    if not images:
        raise SystemExit(
            "No Ambarella calibration images found under {}".format(dataset))

    print("Ambarella calibration: {} image(s) from {}".format(
        len(images), dataset))
    if sample_count == 0:
        return dataset

    sampled_dir = workdir / "calibration_images"
    if not dry_run:
        sampled_dir.mkdir(parents=True, exist_ok=True)
        for image in images:
            shutil.copy2(image, sampled_dir / image.name)
    return sampled_dir


def _descriptor(path, network, inputs, outputs, dra_mode, dra_options):
    begin = {}
    for item in inputs:
        begin[item["name"]] = {
            "shape": item["shape"],
            "dtype": item.get("dtype", "0,0,0,0"),
            "scale": [float(item.get("scale", 255.0))],
            "file": item["file"],
            "extn": ".bin",
        }
    end = {name: {"channels_last": False} for name in outputs}
    data = {
        network: {
            "type": "VP",
            "begin": begin,
            "end": end,
            "attr": {
                "cnngen_flags": "-dra mode={} -c {} ".format(
                    dra_mode, dra_options),
                "vas_flags": "-auto",
            },
        }
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _network_name(model, configured):
    value = str(configured or model.stem)
    value = value.replace(".", "_").replace("-", "_")
    # ADK limits USR_NETWORK to 22 characters.
    return value[:22]


def convert(config_path, mode, workspace, output_dir, container=None, dry_run=False):
    config_path = Path(config_path).resolve()
    root = Path(workspace).resolve()
    config = _load_config(config_path)
    model = _select_model(root, mode, config)
    model = _prepare_headcut_model(
        root, mode, model, bool(config.get("prefer_headcut", True)), dry_run)
    graph = _load_onnx(model)
    inputs = list(graph.graph.input)
    outputs = list(graph.graph.output)
    if not inputs or not outputs:
        raise SystemExit("ONNX model must have at least one input and output")

    input_cfg = config.get("input", {}) or {}
    dataset = _resolve(root, input_cfg.get(
        "images", "modes/{}/datasets/calibration/images".format(mode)))
    if not dataset.is_dir():
        raise SystemExit("Ambarella calibration/test image directory not found: {}".format(dataset))
    shapes = [_shape(item) for item in inputs]
    input_names = [item.name for item in inputs]
    output_names = [item.name for item in outputs]
    network = _network_name(model, config.get("network"))
    workdir = root / "modes" / mode / "outputs" / "compile" / "ambarella" / "work"
    requested_output = _resolve(root, output_dir or config.get(
        "output_dir", "modes/{}/outputs/compile/ambarella".format(mode)))
    template_value = os.environ.get(
        "AMBARELLA_TEMPLATE_DIR", config.get(
            "template_dir", "/home/falcon2/my_model_build"))
    template = Path(template_value)
    adk = os.environ.get("AMBARELLA_ADK_PATH", config.get(
        "adk_path", "/opt/cvtools/sample_nn_diag/diags/onnx/yolo_v8_s_ox/../../../../adk"))
    framework = config.get(
        "framework_dir", "/home/falcon2/amba/linux/ambalinux_sdk/pkg/ambacv")
    dra = config.get("dra", {}) or {}
    dra_mode = int(dra.get("mode", 2))
    if dra_mode not in (1, 2, 3):
        raise SystemExit(
            "Ambarella dra.mode must be 1, 2, or 3; got {} in {}".format(
                dra_mode, config_path))
    dra_options = str(dra.get("options", "act-force-fx16,coeff-force-fx16"))
    model_prune = config.get("model_prune", None)
    if model_prune is not None:
        model_prune = float(model_prune)
        if not 0.0 <= model_prune <= 1.0:
            raise SystemExit(
                "Ambarella model_prune must be between 0 and 1; got {} in {}"
                .format(model_prune, config_path))
    io_cfg = config.get("io", {}) or {}
    io_mode = int(io_cfg.get("mode", 2))
    if io_mode not in (0, 1, 2, 3):
        raise SystemExit(
            "Ambarella io.mode must be 0 (raw), 1 (picinfo), 2 (plain), "
            "or 3 (header); got {} in {}".format(io_mode, config_path))
    # CV7 is built with CVAPI v7 below. ADK explicitly rejects recv-raw
    # (SUPERDAG_TYPE=5) for this API; use plain or picinfo instead.
    if io_mode == 0:
        raise SystemExit(
            "Ambarella io.mode=0 (recv-raw) is not supported with CVAPI v7 "
            "on CV7. Use io.mode=1 (picinfo) or io.mode=2 (plain).")
    # ADK uses SUPERDAG_TYPE for the base input packing and HEADER_FLAG to
    # select header-v3 packing.  Header mode keeps the selected base type.
    superdag_type = {0: 5, 1: 6, 2: 7, 3: 7}[io_mode]
    header_flag = 1 if io_mode == 3 else 0
    # Picinfo mode requires IDSP pyramid/ROI parameters in the ADK task
    # composer.  9999 requests the centered ROI, matching ADK defaults.
    roi_scale = input_cfg.get("roi_scale", 0)
    roi_x = input_cfg.get("roi_x", 9999)
    roi_y = input_cfg.get("roi_y", 9999)
    yuv420_flag = 1 if input_cfg.get("yuv420", False) else 0
    # Let ADK generate the descriptor when YUV420 conversion is enabled so
    # it can expose the required Y and UV primary input buffers.
    use_adk_descriptor = bool(yuv420_flag)

    print("Ambarella model:", model)
    print("Ambarella inputs:", ", ".join(
        "{}={}".format(name, shape) for name, shape in zip(input_names, shapes)))
    print("Ambarella outputs:", ", ".join(output_names))
    print("Ambarella FlexiDAG I/O mode:", io_mode)
    print("Ambarella workdir:", workdir)
    print("Ambarella output:", requested_output)
    if container:
        print("Ambarella container:", container)
    elif not Path(adk).exists() and not dry_run:
        raise SystemExit(
            "Ambarella CVTools are not available on the host. Start the "
            "Ambarella container and set AMBARELLA_CONTAINER to its name "
            "(for example container_20260901_180716), or install CVTools "
            "locally.")

    _copy_template(template, workdir, dry_run)
    calibration_dataset = _sample_calibration_images(
        dataset, workdir, input_cfg, dry_run)
    if not dry_run:
        # compile_preproc removes work/input before copying user materials.
        # Keep both the source ONNX and descriptor outside that directory so
        # they survive the ADK cleanup and can be copied/consumed by ADK.
        source_dir = workdir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        model_copy = source_dir / model.name
        shutil.copy2(model, model_copy)
        # ADK's generated Makefile accepts command-line overrides, but the
        # graph descriptor and model must be present at the same path visible
        # inside the container.
        descriptor = workdir / (network + "_desc.json")
        if not use_adk_descriptor:
            descriptor_inputs = []
            for name, shape in zip(input_names, shapes):
                descriptor_inputs.append({
                    "name": name,
                    "shape": shape,
                    "file": "{}_dra_image_list.txt".format(name),
                    "scale": input_cfg.get("scale", 255.0),
                })
            _descriptor(descriptor, network, descriptor_inputs, output_names,
                        dra_mode, dra_options)
    else:
        descriptor = workdir / (network + "_desc.json")

    env = os.environ.copy()
    env["PROJECT"] = str(config.get("project", "cv7"))
    env["AMBARELLA_ADK_PATH"] = str(adk)
    env["PATH"] = ":".join(filter(None, [
        os.environ.get("AMBARELLA_TOOL_LOCAL_BIN", "/opt/cvtools/cv72/local/bin"),
        os.environ.get("AMBARELLA_TOOL_BIN", "/opt/cvtools/cv72/tv2/exe"),
        "/opt/cvtools/cv7/local/bin",
        "/opt/cvtools/cv7/tv2/exe", env.get("PATH", "")]))

    input_dirs = " ".join(str(calibration_dataset) for _ in inputs)
    input_dims = " ".join(",".join(str(dim) for dim in shape) for shape in shapes)
    input_layers = " ".join(input_names)
    output_layers = " ".join(output_names)
    model_inside = workdir / "source" / model.name
    make_vars = [
        "PROJECT={}".format(env["PROJECT"]),
        "USR_ADK_PATH={}".format(adk),
        "USR_NETWORK={}".format(network),
        "USR_PARSER_ENV=onnx",
        "USR_ONNX_FILE={}".format(model_inside),
        "USR_GRAPH_SURGERY_FLAG=1",
        "USR_GRAPH_SURGERY_OPT=-t CVFlow",
        "USR_GRAPH_SURGERY_INPUT_LAYER={}".format(input_layers),
        "USR_GRAPH_SURGERY_INPUT_DIM={}".format(input_dims),
        "USR_GRAPH_SURGERY_OUT_LAYER={}".format(output_layers),
        "USR_TEST_OUT_LAYER={}".format(output_layers),
        "USR_TEST_INPUT_LAYER={}".format(input_layers),
        "USR_TEST_INPUT_DIR={}".format(input_dirs),
        "USR_TEST_INPUT_DRA_DIR={}".format(input_dirs),
        "USR_TEST_INPUT_DIM={}".format(input_dims),
        "USR_TEST_INPUT_CF={}".format(input_cfg.get("color_format", 1)),
        "USR_TEST_INPUT_SCALE={}".format(input_cfg.get("scale", 255.0)),
        "USR_INPUT_YUV420_FLAG={}".format(yuv420_flag),
        "USR_DRA_MODE={}".format(dra_mode),
        "USR_DRA_OPT={}".format(dra_options),
        "USR_BUB_FWK_DIR={}".format(framework),
        "USR_BUB_FWK_CVAPI_VER=7",
        "USR_SUPERDAG_TYPE={}".format(superdag_type),
        "USR_HEADER_FLAG={}".format(header_flag),
    ]
    if model_prune is not None:
        make_vars.append("USR_MODEL_PRUNE={}".format(model_prune))
    if not use_adk_descriptor:
        make_vars.append("USR_CVB_JSON_FILE={}".format(descriptor))
    if io_mode == 1:
        make_vars.extend([
            "USR_IDSP_ROI_SCALE={}".format(roi_scale),
            "USR_IDSP_ROI_X={}".format(roi_x),
            "USR_IDSP_ROI_Y={}".format(roi_y),
        ])
    make_command = ["make", "-C", workdir] + make_vars + ["cvb"]
    command = _container_command(container, make_command, workdir, env) if container else make_command
    if container and not dry_run:
        _require_running_container(container)
    _run(command, cwd=root if not container else None, env=env if not container else None,
         dry_run=dry_run)
    if dry_run:
        return

    requested_output.mkdir(parents=True, exist_ok=True)
    build_output = workdir / "out" / "deploy" / "build_output"
    artifacts = []
    for source in sorted(build_output.rglob("*")):
        if source.is_file() and (source.suffix in {".bin", ".sh", ".pdf", ".mnft", ".tbar"}
                                 or source.name.endswith(".ckpt.onnx")):
            target = requested_output / source.name
            shutil.copy2(source, target)
            artifacts.append(target)
    for source in sorted((workdir / "out" / "prepare").glob("*.ambapb.ckpt.onnx")):
        target = requested_output / source.name
        shutil.copy2(source, target)
        artifacts.append(target)
    metadata = workdir / "input" / "metadata_info.json"
    if metadata.is_file():
        target = requested_output / metadata.name
        shutil.copy2(metadata, target)
        artifacts.append(target)
    if not artifacts:
        raise SystemExit(
            "Ambarella build completed but no deploy artifacts were found under {}".format(
                build_output))
    print("Ambarella artifacts:")
    for artifact in artifacts:
        print("  {}".format(artifact))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--container", default=os.environ.get("AMBARELLA_CONTAINER"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        convert(args.config, args.mode, args.workspace, args.output_dir,
                args.container, args.dry_run)
    except subprocess.CalledProcessError as error:
        raise SystemExit("Ambarella command failed with exit code {}".format(
            error.returncode)) from error


if __name__ == "__main__":
    main()
