"""Utility Evaluation stages: decryption, poisoning gate, YOLO training, mAP and FID."""

from __future__ import annotations

from r26_pipeline.stages.base import Context, Stage, opt

SYNTHETIC_RUN = "synthetic_baseline"
REAL_RUN = "kitti_baseline"


def _root(ctx: Context) -> tuple[str, ...]:
    return ("--root", str(ctx.ws.utility), "--decrypted-dir", str(ctx.ws.decrypted))


def intake(ctx: Context) -> list[Stage]:
    ws = ctx.ws
    return [
        Stage(
            "data.decrypt", "utility",
            "Decrypt the delivered dataset once, verifying every file against its manifest",
            module="scripts.decrypt_all_images",
            args=("--root", str(ws.utility), "--encrypted-dir", str(ctx.encrypted),
                  "--decrypted-dir", str(ws.decrypted), "--towns", *ctx.towns, "--strict"),
            inputs=(ctx.encrypted,),
            outputs=(ws.decrypted,),
            clean=(ws.decrypted,),
        ),
    ]


def poisoning(ctx: Context) -> list[Stage]:
    ws = ctx.ws
    return [
        Stage(
            "utility.detect_poisoning", "utility",
            "Poisoning gate: structural, batch and cross-town label checks",
            module="scripts.detect_poisoning",
            args=_root(ctx),
            inputs=(ws.decrypted,),
            outputs=(ws.utility_results / "poisoning_audit.json", ws.clean_labels),
            contracts=((ws.utility_results / "poisoning_audit.json", "poisoning_audit"),),
            needs=("compliance.verify_manifest",),
        ),
    ]


def evaluation(ctx: Context) -> list[Stage]:
    ws, u = ctx.ws, ctx.cfg.utility
    root = _root(ctx)
    stages = [
        Stage(
            "utility.extract_gt", "utility", "Join clean labels with frames and class masks",
            module="scripts.extract_ground_truth", args=root,
            inputs=(ws.clean_labels,), outputs=(ws.utility / "ground_truth_extraction",),
            clean=(ws.utility / "ground_truth_extraction",),
            needs=("utility.detect_poisoning",),
        ),
        Stage(
            "utility.verify_alignment", "utility", "Check every box sits on its object in the class mask",
            module="scripts.verify_label_alignment", args=root,
            inputs=(ws.utility / "ground_truth_extraction",),
            outputs=(ws.utility / "label_alignment_report" / "overall_summary.txt",),
            needs=("utility.extract_gt",),
        ),
        Stage(
            "utility.analyze_misalignment", "utility", "Explain each misaligned box",
            module="scripts.analyze_misalignment", args=root,
            inputs=(ws.utility / "ground_truth_extraction",),
            outputs=(ws.utility / "label_alignment_report" / "misalignment_breakdown_summary.txt",),
            needs=("utility.extract_gt",),
        ),
        Stage(
            "utility.build_dataset", "utility", "Build the YOLO dataset from clean labels only",
            module="scripts.convert_to_yolo",
            args=(*root, "--labels", "clean", "--split-mode", u.split_mode, "--overwrite"),
            inputs=(ws.decrypted, ws.clean_labels), outputs=(ws.utility / "yolo_dataset" / "data.yaml",),
            needs=("utility.detect_poisoning",),
        ),
        Stage(
            "utility.build_kitti", "utility", "Convert KITTI for the real-data baseline",
            module="scripts.kitti_to_yolo",
            args=("--root", str(ws.utility), "--kitti-dir", str(ctx.cfg.inputs.kitti), "--overwrite"),
            inputs=tuple(x for x in (ctx.cfg.inputs.kitti,) if x),
            outputs=(ws.utility / "kitti_dataset" / "data.yaml",),
            skip_reason=ctx.no_kitti,
        ),
    ]

    train = ["--model", u.model, "--epochs", str(u.epochs), "--imgsz", str(u.imgsz),
             "--batch", str(u.batch), "--workers", str(u.workers),
             "--patience", str(u.patience), *opt("--device", u.device)]
    stages += [
        Stage(
            "utility.train_synthetic", "utility", "Train YOLOv8 on the synthetic data",
            module="scripts.train_yolo",
            args=("--root", str(ws.utility), "--dataset", "synthetic", "--name", SYNTHETIC_RUN, *train),
            inputs=(ws.utility / "yolo_dataset",),
            outputs=(ws.utility_weights(SYNTHETIC_RUN),),
            clean=(ws.utility / "runs" / SYNTHETIC_RUN,),
            needs=("utility.build_dataset",),
        ),
        Stage(
            "utility.train_kitti", "utility", "Train YOLOv8 on KITTI with identical settings",
            module="scripts.train_yolo",
            args=("--root", str(ws.utility), "--dataset", "kitti", "--name", REAL_RUN, *train),
            inputs=(ws.utility / "kitti_dataset",),
            outputs=(ws.utility_weights(REAL_RUN),),
            clean=(ws.utility / "runs" / REAL_RUN,),
            needs=("utility.build_kitti",),
            skip_reason=ctx.no_kitti,
        ),
        Stage(
            "utility.cross_eval", "utility", "The four train and test domain experiments",
            module="scripts.cross_evaluation",
            args=("--root", str(ws.utility), *opt("--device", u.device)),
            inputs=tuple(ws.utility_weights(r) for r in (SYNTHETIC_RUN, REAL_RUN)
                         if r == SYNTHETIC_RUN or not ctx.no_kitti),
            outputs=(ws.utility_results / "evaluations.json",),
            needs=("utility.train_synthetic",),
        ),
        Stage(
            "utility.size_map", "utility", "AP by object size",
            module="scripts.size_breakdown_map",
            args=("--root", str(ws.utility), "--name", SYNTHETIC_RUN, *opt("--device", u.device)),
            inputs=(ws.utility_weights(SYNTHETIC_RUN),),
            outputs=(ws.utility_results / f"size_breakdown_{SYNTHETIC_RUN}_on_synthetic.json",),
            needs=("utility.train_synthetic",),
        ),
        Stage(
            "utility.fid", "utility", "FID between KITTI and synthetic test images",
            module="scripts.compute_fid",
            args=("--root", str(ws.utility), "--limit", str(u.fid_limit), "--crop", u.fid_crop),
            inputs=(ws.utility / "yolo_dataset", ws.utility / "kitti_dataset"),
            outputs=(ws.utility_results / "fid.json",),
            needs=("utility.build_dataset", "utility.build_kitti"),
            skip_reason=ctx.no_kitti,
        ),
        Stage(
            "utility.report", "utility", "Utility retention and the utility findings",
            module="scripts.build_report",
            args=("--root", str(ws.utility)),
            inputs=(ws.utility_results / "evaluations.json", ws.utility_results / "poisoning_audit.json",
                    ws.utility_results / "fid.json"),
            outputs=(ws.utility_results / "utility_results.json",),
            contracts=((ws.utility_results / "utility_results.json", "utility_results"),),
            needs=("utility.cross_eval",),
        ),
    ]
    return stages
