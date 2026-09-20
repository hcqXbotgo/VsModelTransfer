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
import re
import shlex
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}
ADK_SAMPLE_PROJECT = "/opt/cvtools/sample_nn_diag/diags/onnx/yolo_v8_s_ox"


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


def _bootstrap_template_from_container(container, workdir, adk, project, dry_run):
    """Generate a configured ADK project from the example in CVTools."""
    print("Ambarella template: generating from container example", ADK_SAMPLE_PROJECT)
    if dry_run:
        return
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    env = {"PROJECT": str(project)}
    command = _container_command(container, [
        "cp", "-a", ADK_SAMPLE_PROJECT + "/.", str(workdir)], workdir, env)
    _require_running_container(container)
    _run(command)

    makefile = workdir / "Makefile.in"
    if not makefile.is_file():
        raise SystemExit("Ambarella example did not provide Makefile.in: {}"
                         .format(ADK_SAMPLE_PROJECT))
    # configure.ac reads the ADK directory from Makefile.in.  The vendor
    # example's relative path stops working when copied into a mode workdir.
    relative_adk = os.path.relpath(str(adk), str(workdir))
    source = makefile.read_text(encoding="utf-8")
    source, count = re.subn(
        r"(?m)^(USR_ADK_PATH\s*=\s*).*$",
        lambda match: match.group(1) + "@srcdir@/" + relative_adk,
        source, count=1)
    if count != 1:
        raise SystemExit("Ambarella example has no USR_ADK_PATH in Makefile.in")
    # The bundled example declares two Y/UV test inputs with += and a
    # preproc.json for those inputs. ADK reads dimensions from that JSON even
    # when make overrides the layer names and dimensions on the command line.
    # Its bundled CVFlow descriptor also refers to old sample image lists;
    # leave it empty so ADK generates one for the configured ONNX and YUV flag.
    input_fields = (
        "USR_TEST_INPUT_DRA_DIR", "USR_TEST_INPUT_LAYER", "USR_TEST_INPUT_DIR",
        "USR_TEST_INPUT_CF", "USR_TEST_INPUT_DIM", "USR_TEST_OUT_LAYER",
        "USR_JSON_PREPOST_PROC_FILE", "USR_CVB_JSON_FILE")
    for field in input_fields:
        source, count = re.subn(
            r"(?m)^({}\s*)(\+?=)[^\n]*\n?".format(field),
            lambda match: (match.group(1) + "=\n")
            if match.group(2) == "=" else "",
            source)
        if count == 0:
            raise SystemExit("Ambarella example has no {} in Makefile.in"
                             .format(field))
    makefile.write_text(source, encoding="utf-8")

    command = _container_command(container, ["bash", "-lc",
        "autoconf && ./configure"], workdir, env)
    _run(command)
    for name in ("Makefile", "Makefile.command", "Makefile.internal_cfg",
                 "Makefile.combo"):
        if not (workdir / name).is_file():
            raise SystemExit("Ambarella template generation did not create {}"
                             .format(workdir / name))


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


def _yuv_resize_descriptor(path, network, input_name, outputs, source_size,
                           model_size, scale, dra_mode, dra_options):
    """Describe NV12 resize, color conversion, and the network as one graph."""
    source_width, source_height = source_size
    model_width, model_height = model_size
    source_y = input_name + "_y"
    source_uv = input_name + "_uv"
    resized_y = input_name + "_resized_y"
    resized_uv = input_name + "_resized_uv"
    rgb = input_name + "_RGB"
    scale_name = input_name + "_data_scale"

    data = {
        input_name + "_y_resize": {
            "type": "SGL::Resize",
            "begin": {
                source_y: {
                    "shape": [1, 1, source_height, source_width],
                    "dtype": "0,0,0,0",
                    "dram_format": 0,
                },
            },
            "end": {resized_y: {}},
            "attr": {"h": model_height, "w": model_width},
        },
        input_name + "_uv_resize": {
            "type": "SGL::Resize",
            "begin": {
                source_uv: {
                    "shape": [1, 2, source_height // 2, source_width // 2],
                    "dtype": "0,0,0,0",
                    "dram_format": 1,
                    "file_dram_format": 1,
                },
            },
            "end": {resized_uv: {}},
            "attr": {"h": model_height // 2, "w": model_width // 2},
        },
        input_name + "_cc_node": {
            "type": "SGL.E::ColorConvert",
            "begin": {
                resized_y: {
                    "shape": [1, 1, model_height, model_width],
                    "dtype": "0,0,0,0",
                },
                resized_uv: {
                    "shape": [1, 2, model_height // 2, model_width // 2],
                    "dtype": "0,0,0,0",
                },
            },
            "end": {rgb: {}},
            "attr": {
                "code": "yuv420_to_rgb_nv12",
                "data_format": "0,0,0,0",
            },
        },
        input_name + "_scale_node": {
            "type": "SGL::Div",
            "begin": {
                rgb: {},
                scale_name: {
                    "shape": [1, 1, 1, 1],
                    "init": [float(scale)],
                    "dtype": "1,2,0,7",
                },
            },
            "end": {input_name: {}},
        },
        network: {
            "type": "VP",
            "begin": {input_name: {}},
            "end": {
                name: {"channels_last": False}
                for name in outputs
            },
            "attr": {
                "cnngen_flags": "-dra mode={} -c {} ".format(
                    dra_mode, dra_options),
                "vas_flags": "-auto",
            },
        },
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _replace_once(source, old, new, description):
    count = source.count(old)
    if count != 1:
        raise SystemExit(
            "Ambarella dynamic resize expected one {}, found {}".format(
                description, count))
    return source.replace(old, new, 1)


def _patch_dynamic_resize_vas(path, input_name, model_size):
    """Replace CNNGen's fixed SGL resize with runtime-controlled VP resize."""
    model_width, model_height = model_size
    source_y = input_name + "_y"
    source_uv = input_name + "_uv"
    resized_y = input_name + "_resized_y"
    resized_uv = input_name + "_resized_uv"
    dzoom_y = input_name + "_dzoom_y"
    dzoom_uv = input_name + "_dzoom_uv"
    source = path.read_text(encoding="utf-8")
    if ("VP_variableresamp({}, {}".format(source_y, dzoom_y) in source and
            "VP_variableresamp({}, {}".format(source_uv, dzoom_uv) in source):
        return

    first_resamp = source.index("    VP_resamp({}".format(source_y))
    source = source[:first_resamp] + (
        "    VP_input({}, uint32_t, vector(1, 1, 1, 4));\n\n".format(
            dzoom_y)) + source[first_resamp:]
    second_resamp = source.index("    VP_resamp({}".format(source_uv))
    source = source[:second_resamp] + (
        "    VP_input({}, uint32_t, vector(1, 1, 1, 4));\n\n".format(
            dzoom_uv)) + source[second_resamp:]

    def replace_resamp(text, image, dzoom, output, out_width, out_height):
        pattern = re.compile(
            r"    VP_resamp\(" + re.escape(image) +
            r",\s*\n.*?\n\s*\);", re.DOTALL)
        replacement = (
            "    VP_variableresamp({image}, {dzoom},\n"
            "              VP_tensor({output}, u8(0), "
            "vector(1, {channels}, {height}, {width}), __sparsity = 0.00),\n"
            "              replicate_w = false,\n"
            "              replicate_h = false,\n"
            "              resamp_mode = 0,\n"
            "              out_w = {width},\n"
            "              out_h = {height}\n"
            "             );"
        ).format(image=image, dzoom=dzoom, output=output,
                 channels=1 if image == source_y else 2,
                 width=out_width, height=out_height)
        text, count = pattern.subn(replacement, text, count=1)
        if count != 1:
            raise SystemExit(
                "Ambarella dynamic resize could not find VP_resamp({}) in {}"
                .format(image, path))
        return text

    source = replace_resamp(
        source, source_y, dzoom_y, resized_y, model_width, model_height)
    source = replace_resamp(
        source, source_uv, dzoom_uv, resized_uv,
        model_width // 2, model_height // 2)
    path.write_text(source, encoding="utf-8")


def _patch_dynamic_resize_cvtask(c_path, h_path, task_name, model_size,
                                 max_size):
    """Make the generated PicInfo task derive resize parameters per frame."""
    model_width, model_height = model_size
    max_width, max_height = max_size
    header = h_path.read_text(encoding="utf-8")
    if "    uint32_t roi_width;" not in header:
        header = _replace_once(
            header,
            "    uint32_t dpitch_m1_img_uv;\n\n    uint32_t source_vin;",
            "    uint32_t dpitch_m1_img_uv;\n"
            "    uint32_t roi_width;\n"
            "    uint32_t roi_height;\n\n"
            "    uint32_t source_vin;",
            "img_ctx pitch fields")
    h_path.write_text(header, encoding="utf-8")

    source = c_path.read_text(encoding="utf-8")
    if "    uint32_t dzoom_y[32]" not in source:
        source = _replace_once(
            source,
            "struct DRAM_temporary_scratchpad {{\n"
            "    {0}_dram_t {1}_dram;\n"
            "}};".format(task_name, task_name),
            "struct DRAM_temporary_scratchpad {{\n"
            "    uint32_t dzoom_y[32];\n"
            "    uint32_t dzoom_uv[32];\n"
            "    {0}_dram_t {1}_dram;\n"
            "}};".format(task_name, task_name),
            "DRAM scratchpad declaration")
    if "    uint32_t dzoom[4];" not in source:
        source = _replace_once(
            source,
            "    amba_roi_config_t p_cfg_msg;\n};",
            "    amba_roi_config_t p_cfg_msg;\n"
            "    uint32_t dzoom[4];\n};",
            "CMEM scratchpad declaration")
    feedback_desc = re.compile(
        r"    \.num_feedback = 2,\n"
        r"    \.feedback\[0\] = \{\n.*?"
        r"    \.feedback\[1\] = \{\n.*?\n    \},\n",
        re.DOTALL)
    source, feedback_count = feedback_desc.subn(
        "    .num_feedback = 0,\n", source, count=1)
    if feedback_count != 1 and "    .num_feedback = 0," not in source:
        raise SystemExit(
            "Ambarella dynamic resize could not internalize feedback ports in {}"
            .format(c_path))

    feedback_start = source.index("    // Feedback buffers")
    feedback_end = source.index("    // Output buffers", feedback_start)
    dynamic_block = """    // Runtime resize parameters use VP 19.13 phase increments. Both
    // NV12 planes use the same ratio because UV is half-sized in both axes.
    cmem->dzoom[0] = (cmem->img_ctx.roi_width << 13) / {model_width}U;
    cmem->dzoom[1] = (cmem->img_ctx.roi_height << 13) / {model_height}U;
    cmem->dzoom[2] = 0U;
    cmem->dzoom[3] = 0U;
    status = vishw_dma_cmem_to_dxofsaddr(
        status,
        pCVTaskEnv->DRAM_temporary_scratchpad_dxaddr,
        visorc_offsetof(struct DRAM_temporary_scratchpad, dzoom_y),
        cmem->dzoom,
        sizeof(cmem->dzoom)
    );
    status = vishw_dma_cmem_to_dxofsaddr(
        status,
        pCVTaskEnv->DRAM_temporary_scratchpad_dxaddr,
        visorc_offsetof(struct DRAM_temporary_scratchpad, dzoom_uv),
        cmem->dzoom,
        sizeof(cmem->dzoom)
    );
    if (is_err(status)) {{
        P_DEBUG(cis_id, "   > Error: dynamic resize parameter DMA failed (%u).\\n",
                status, 0);
        return ERRCODE_GENERIC;
    }}
    {task_name}_r_args->images_dzoom_y_dxaddr =
        pCVTaskEnv->DRAM_temporary_scratchpad_dxaddr;
    {task_name}_r_args->images_dzoom_y_dxofs =
        visorc_offsetof(struct DRAM_temporary_scratchpad, dzoom_y);
    {task_name}_r_args->images_dzoom_uv_dxaddr =
        pCVTaskEnv->DRAM_temporary_scratchpad_dxaddr;
    {task_name}_r_args->images_dzoom_uv_dxofs =
        visorc_offsetof(struct DRAM_temporary_scratchpad, dzoom_uv);

""".format(task_name=task_name, model_width=model_width,
           model_height=model_height)
    if "Runtime resize parameters use VP 19.13" not in source:
        source = source[:feedback_start] + dynamic_block + source[feedback_end:]

    roi_pattern = re.compile(
        r"static errcode_enum_t " + re.escape(task_name) +
        r"_roi_handling\(\n.*?\n}\n\nstatic errcode_enum_t " +
        re.escape(task_name) + r"_picinfo_store", re.DOTALL)
    roi_function = """static errcode_enum_t {task_name}_roi_handling(
    img_ctx_t     *img_ctx,
    cv_pic_info_t *pic_info
) {{
    uint32_t roi_w, roi_h, luma, chroma, pitch;
    int32_t roi_start_col, roi_start_row;
    uint32_t p_scale = img_ctx->idsp_pyramid_scale;

    roi_w = pic_info->pyramid.half_octave[p_scale].roi_width_m1 + 1U;
    roi_h = pic_info->pyramid.half_octave[p_scale].roi_height_m1 + 1U;
    roi_start_col = pic_info->pyramid.half_octave[p_scale].roi_start_col;
    roi_start_row = pic_info->pyramid.half_octave[p_scale].roi_start_row;
    pitch = pic_info->pyramid.half_octave[p_scale].ctrl.roi_pitch;
    if ((roi_w < 2U) || (roi_h < 2U) ||
        (roi_w > {max_width}U) || (roi_h > {max_height}U) ||
        ((roi_w & 1U) != 0U) || ((roi_h & 1U) != 0U) ||
        (roi_start_col < 0) || (roi_start_row < 0) ||
        ((roi_start_col & 1) != 0) || ((roi_start_row & 1) != 0) ||
        (pitch < (roi_w + (uint32_t)roi_start_col))) {{
        cvtask_printf(LVL_CRITICAL,
            "dynamic resize invalid ROI %ux%u start=(%d,%d) pitch=%u\\n",
            roi_w, roi_h, roi_start_col, roi_start_row, pitch);
        return ERRCODE_BAD_PARAMETER;
    }}

    if (img_ctx->source_vin == 1U) {{
        luma = pic_info->rpLumaLeft[p_scale];
        chroma = pic_info->rpChromaLeft[p_scale];
    }} else {{
        luma = pic_info->rpLumaRight[p_scale];
        chroma = pic_info->rpChromaRight[p_scale];
    }}
    img_ctx->daddr_img_y_base_dxofs += luma;
    img_ctx->daddr_img_y_base_dxofs += (uint32_t)roi_start_row * pitch;
    img_ctx->daddr_img_y_base_dxofs += (uint32_t)roi_start_col;
    img_ctx->daddr_img_uv_base_dxofs += chroma;
    img_ctx->daddr_img_uv_base_dxofs += (uint32_t)roi_start_row * pitch / 2U;
    img_ctx->daddr_img_uv_base_dxofs += (uint32_t)roi_start_col;
    img_ctx->dpitch_m1_img_y = pitch - 1U;
    img_ctx->dpitch_m1_img_uv = pitch - 1U;
    img_ctx->roi_width = roi_w;
    img_ctx->roi_height = roi_h;
    cvtask_printf(LVL_DEBUG,
        "dynamic resize PicInfo ROI %ux%u start=(%d,%d) pitch=%u\\n",
        roi_w, roi_h, roi_start_col, roi_start_row, pitch);
    return ERRCODE_NONE;
}}

static errcode_enum_t {task_name}_picinfo_store""".format(
        task_name=task_name, max_width=max_width, max_height=max_height)
    if "dynamic resize invalid ROI" not in source:
        source, count = roi_pattern.subn(lambda _match: roi_function,
                                         source, count=1)
        if count != 1:
            raise SystemExit(
                "Ambarella dynamic resize could not patch PicInfo ROI handling in {}"
                .format(c_path))
    c_path.write_text(source, encoding="utf-8")


def _patch_picinfo_logical_metadata(path, model_size):
    """Expose the post-resize model shape while retaining max input capacity."""
    model_width, model_height = model_size
    data = bytearray(path.read_bytes())
    # cvflow_flexidag_metadata_t starts with version/mode, then the input
    # count and 128 fixed-size cvflow_md_buffer_info_t records (48 bytes each).
    if len(data) < 108:
        raise SystemExit("Ambarella IO metadata is truncated: {}".format(path))
    _version, mode, input_count = struct.unpack_from("<3I", data, 0)
    if mode != 1 or input_count < 2:
        raise SystemExit(
            "Ambarella dynamic resize requires PicInfo Y/UV metadata in {}"
            .format(path))
    input_info_offset = 12
    input_info_size = 48
    y_dims = list(struct.unpack_from("<5I", data, input_info_offset))
    uv_offset = input_info_offset + input_info_size
    uv_dims = list(struct.unpack_from("<5I", data, uv_offset))
    y_dims[3:5] = [model_height, model_width]
    uv_dims[3:5] = [model_height // 2, model_width // 2]
    struct.pack_into("<5I", data, input_info_offset, *y_dims)
    struct.pack_into("<5I", data, uv_offset, *uv_dims)
    path.write_bytes(data)


def _patch_metadata_info_json(path, input_name, model_size, max_size):
    if not path.is_file():
        return
    model_width, model_height = model_size
    max_width, max_height = max_size
    data = json.loads(path.read_text(encoding="utf-8"))
    inputs = data.get("input", {})
    y_info = inputs.get(input_name + "_y")
    uv_info = inputs.get(input_name + "_uv")
    if not isinstance(y_info, dict) or not isinstance(uv_info, dict):
        raise SystemExit(
            "Ambarella metadata_info.json has no dynamic resize Y/UV inputs: {}"
            .format(path))
    y_info["dim"] = "1,1,1,{},{}".format(model_height, model_width)
    y_info["max_dim"] = "1,1,1,{},{}".format(max_height, max_width)
    uv_info["dim"] = "1,1,2,{},{}".format(
        model_height // 2, model_width // 2)
    uv_info["max_dim"] = "1,1,2,{},{}".format(
        max_height // 2, max_width // 2)
    path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")


def _enable_dynamic_hardware_resize(workdir, network, input_name, model_size,
                                    max_size, container, env):
    """Regenerate and package one FlexiDAG with per-frame NV12 resizing."""
    frag_root = (workdir / "out" / "prepare" / "cnngen_out" / network /
                 "frag-out")
    metanodes = sorted(path for path in frag_root.iterdir()
                       if path.is_dir() and path.name.endswith("_prim_nvp0"))
    if len(metanodes) != 1:
        raise SystemExit(
            "Ambarella dynamic resize requires exactly one NVP metanode; "
            "found {} under {}".format(len(metanodes), frag_root))
    source_node = metanodes[0]
    vas_path = source_node / (source_node.name + ".vas")
    _patch_dynamic_resize_vas(vas_path, input_name, model_size)

    vas_command = ["vas", vas_path.name, "-auto", "-nvp"]
    command = (_container_command(container, vas_command, source_node, env)
               if container else vas_command)
    _run(command, cwd=None if container else source_node,
         env=None if container else env)

    deploy_node = (workdir / "input" / "deploy_input" / "metanodes" /
                   source_node.name)
    vas_output = source_node / "vas_output"
    runtime_suffixes = {".dagbin", ".ddi", ".dvi", ".h", ".summary",
                        ".vdg", ".vlist"}
    for generated in vas_output.iterdir():
        if generated.is_file() and generated.suffix in runtime_suffixes:
            shutil.copy2(generated, deploy_node / generated.name)

    deploy_dir = workdir / "out" / "deploy"
    task_dir = deploy_dir / "cvtasks" / "task_0"
    autogen_script = deploy_dir / "autogen_cmd.sh"
    command = shlex.split(autogen_script.read_text(encoding="utf-8"))
    if task_dir.exists():
        shutil.rmtree(task_dir)
    feedback_dir = deploy_dir / "dynamic_resize_feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    initial = struct.pack("<4I", 1 << 13, 1 << 13, 0, 0)
    y_init = feedback_dir / "dzoom_y.bin"
    uv_init = feedback_dir / "dzoom_uv.bin"
    y_init.write_bytes(initial)
    uv_init.write_bytes(initial)
    command.extend([
        "--feedbacktasks", input_name + "_dzoom_y=DYNAMIC_RESIZE_Y",
        "--feedbacktasks", input_name + "_dzoom_uv=DYNAMIC_RESIZE_UV",
        "--feedbackinit", "DYNAMIC_RESIZE_Y=0:{}".format(y_init),
        "--feedbackinit", "DYNAMIC_RESIZE_UV=0:{}".format(uv_init),
    ])
    wrapped = (_container_command(container, command, workdir, env)
               if container else command)
    _run(wrapped, cwd=None if container else workdir,
         env=None if container else env)

    _patch_dynamic_resize_cvtask(
        task_dir / "task_0_cvtask.c", task_dir / "task_0_cvtask.h",
        "task_0", model_size, max_size)
    task_manifest = task_dir / "task_0_cvtask.mnft"
    manifest_lines = task_manifest.read_text(encoding="utf-8").splitlines()
    task_manifest.write_text(
        "\n".join(line for line in manifest_lines
                  if "DYNAMIC_RESIZE_" not in line) + "\n",
        encoding="utf-8")
    metadata_paths = [
        workdir / "input" / "deploy_io_metadata.bin",
        deploy_dir / "build_input" / "metadata" / "deploy_io_metadata.bin",
    ]
    for metadata_path in metadata_paths:
        if not metadata_path.is_file():
            raise SystemExit(
                "Ambarella IO metadata not found: {}".format(metadata_path))
        _patch_picinfo_logical_metadata(metadata_path, model_size)
    _patch_metadata_info_json(
        workdir / "input" / "metadata_info.json", input_name,
        model_size, max_size)
    build_script = deploy_dir / "build_cmd.sh"
    build_lines = [line.strip() for line in
                   build_script.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    if len(build_lines) < 2 or not build_lines[1].startswith("remoteconfig "):
        raise SystemExit(
            "Ambarella deploy script has no remoteconfig command: {}"
            .format(build_script))
    build_output = deploy_dir / "build_output"
    remoteconfig_command = shlex.split(build_lines[1])
    wrapped = (_container_command(
        container, remoteconfig_command, build_output, env)
        if container else remoteconfig_command)
    _run(wrapped, cwd=None if container else build_output,
         env=None if container else env)
    build_command = ["make", "-C", str(build_output), "build"]
    wrapped = (_container_command(container, build_command, workdir, env)
               if container else build_command)
    _run(wrapped, cwd=None if container else workdir,
         env=None if container else env)
    generated_flexibin = build_output / "flexidag0" / "flexibin0.bin"
    packaged_flexibin = build_output / "flexibin" / "flexibin0.bin"
    if not generated_flexibin.is_file():
        raise SystemExit(
            "Ambarella dynamic FlexiBin was not generated: {}".format(
                generated_flexibin))
    packaged_flexibin.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated_flexibin, packaged_flexibin)


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
        "AMBARELLA_TEMPLATE_DIR", config.get("template_dir", "auto"))
    template = (None if not template_value or str(template_value).lower() == "auto"
                else _resolve(root, template_value))
    adk = os.environ.get("AMBARELLA_ADK_PATH", config.get(
        "adk_path", "/opt/cvtools/sample_nn_diag/diags/onnx/yolo_v8_s_ox/../../../../adk"))
    framework = config.get("framework_dir")
    if not framework:
        raise SystemExit(
            "Ambarella framework_dir is required in compile.yaml; set it "
            "to the CVAPI framework path accessible inside the container.")
    framework = _resolve(root, framework)
    if not framework.is_dir():
        raise SystemExit(
            "Ambarella framework_dir not found on the host: {}. Place the "
            "CVAPI framework there and mount the repository into the "
            "container at the same path.".format(framework))
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
    output_cfg = config.get("output", {}) or {}
    output_dpa = output_cfg.get("dram_pitch_alignment", None)
    if output_dpa is not None:
        # ADK expects one DPA value for every primary output.  A scalar is a
        # convenient shorthand for applying the same alignment to all outputs;
        # an explicit list is also accepted for per-output control.
        if isinstance(output_dpa, (list, tuple)):
            output_dpa = [int(value) for value in output_dpa]
        else:
            output_dpa = [int(output_dpa)] * len(output_names)
        if len(output_dpa) != len(output_names):
            raise SystemExit(
                "Ambarella output.dram_pitch_alignment must contain one "
                "value per output ({} expected, {} got) in {}"
                .format(len(output_names), len(output_dpa), config_path))
        if any(value not in (0, 1, 2, 3) for value in output_dpa):
            raise SystemExit(
                "Ambarella output.dram_pitch_alignment values must be 0 "
                "(auto), 1 (contiguous), 2 (32-byte), or 3 (64-byte); got {} "
                "in {}".format(output_dpa, config_path))
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
    resize_cfg = input_cfg.get("hardware_resize", {}) or {}
    resize_enabled = bool(resize_cfg.get("enabled", False))
    resize_dynamic = bool(resize_cfg.get("dynamic", False))
    source_width = int(resize_cfg.get("max_width", 0) or 0)
    source_height = int(resize_cfg.get("max_height", 0) or 0)
    if resize_enabled:
        if not yuv420_flag:
            raise SystemExit(
                "Ambarella input.hardware_resize requires input.yuv420=true")
        if len(inputs) != 1 or len(shapes[0]) != 4 or shapes[0][1] != 3:
            raise SystemExit(
                "Ambarella input.hardware_resize requires one NCHW RGB model input")
        if (source_width <= 0 or source_height <= 0 or
                (source_width & 1) != 0 or (source_height & 1) != 0):
            raise SystemExit(
                "Ambarella input.hardware_resize max_width/max_height "
                "must be positive even values")
        if not resize_dynamic:
            raise SystemExit(
                "Ambarella input.hardware_resize currently requires dynamic=true")
        model_height = int(shapes[0][2])
        model_width = int(shapes[0][3])
        if (model_width & 1) != 0 or (model_height & 1) != 0:
            raise SystemExit(
                "Ambarella input.hardware_resize requires an even-sized model input")
    # Let ADK generate the descriptor when YUV420 conversion is enabled so
    # it can expose the required Y and UV primary input buffers.
    use_adk_descriptor = bool(yuv420_flag and not resize_enabled)

    print("Ambarella model:", model)
    print("Ambarella inputs:", ", ".join(
        "{}={}".format(name, shape) for name, shape in zip(input_names, shapes)))
    print("Ambarella outputs:", ", ".join(output_names))
    print("Ambarella FlexiDAG I/O mode:", io_mode)
    if resize_enabled:
        print("Ambarella dynamic hardware resize: <= {}x{} -> {}x{} "
              "(single graph)".format(
            source_width, source_height, model_width, model_height))
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

    if template is None and container:
        _bootstrap_template_from_container(
            container, workdir, adk, config.get("project", "cv7"), dry_run)
    elif template is None:
        raise SystemExit(
            "Ambarella template_dir=auto requires a container. Set "
            "AMBARELLA_CONTAINER or specify a local template_dir.")
    else:
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
        if resize_enabled:
            _yuv_resize_descriptor(
                descriptor, network, input_names[0], output_names,
                (source_width, source_height), (model_width, model_height),
                input_cfg.get("scale", 255.0), dra_mode, dra_options)
        elif not use_adk_descriptor:
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
    test_input_dirs = input_dirs
    test_input_dims = input_dims
    test_input_layers = input_layers
    test_input_cf = str(input_cfg.get("color_format", 1))
    test_input_scale = str(input_cfg.get("scale", 255.0))
    effective_yuv420_flag = str(yuv420_flag)
    if resize_enabled:
        test_input_dirs = "{} {}".format(calibration_dataset, calibration_dataset)
        test_input_dims = "1,1,{},{} 1,2,{},{}".format(
            source_height, source_width, source_height // 2, source_width // 2)
        test_input_layers = "{}_y {}_uv".format(input_names[0], input_names[0])
        # Match CVTools' native Y and interleaved-UV input formats. The custom
        # CVB descriptor performs scaling and NV12 conversion itself.
        test_input_cf = "4 7"
        test_input_scale = ""
        effective_yuv420_flag = ""
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
        "USR_TEST_INPUT_LAYER={}".format(test_input_layers),
        "USR_TEST_INPUT_DIR={}".format(test_input_dirs),
        "USR_TEST_INPUT_DRA_DIR={}".format(test_input_dirs),
        "USR_TEST_INPUT_DIM={}".format(test_input_dims),
        "USR_TEST_INPUT_CF={}".format(test_input_cf),
        "USR_TEST_INPUT_SCALE={}".format(test_input_scale),
        "USR_INPUT_YUV420_FLAG={}".format(effective_yuv420_flag),
        "USR_DRA_MODE={}".format(dra_mode),
        "USR_DRA_OPT={}".format(dra_options),
        "USR_BUB_FWK_DIR={}".format(framework),
        "USR_BUB_FWK_CVAPI_VER=7",
        "USR_SUPERDAG_TYPE={}".format(superdag_type),
        "USR_HEADER_FLAG={}".format(header_flag),
    ]
    if model_prune is not None:
        make_vars.append("USR_MODEL_PRUNE={}".format(model_prune))
    if output_dpa is not None:
        make_vars.append("USR_TEST_FORCE_OUT_DPA={}".format(
            " ".join(str(value) for value in output_dpa)))
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
    if resize_enabled:
        _enable_dynamic_hardware_resize(
            workdir, network, input_names[0], (model_width, model_height),
            (source_width, source_height), container, env)

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
