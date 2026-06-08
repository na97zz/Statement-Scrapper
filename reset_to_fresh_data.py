from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


RAW_BANK_FOLDERS = [
    "BOC",
    "BOE",
    "BOJ",
    "ECB",
    "FED",
    "NORGES_BANK",
    "RBA",
    "RBNZ",
    "RIKSBANK",
    "SNB",
]

GENERATED_SCRIPTS = [
    "meeting_linkage.py",
    "hawkometer_scorer.py",
    "hawkometer_analytics.py",
    "hawkometer_backtest.py",
    "policy_decisions_builder.py",
    "policy_decisions_qc.py",
    "policy_decisions_qc_detailed.py",
]

GENERATED_FOLDERS = [
    "processed/meeting_linkage",
    "processed/hawkometer",
    "processed/hawkometer_analytics",
    "processed/hawkometer_backtest",
    "processed/policy_decisions",
    "data/external",
]

GENERATED_FILE_NAMES = [
    "updated_master_index.csv",
    "meeting_index.csv",
    "meeting_documents.csv",
    "linkage_summary.csv",
    "document_scores.csv",
    "speaker_rolling_scores.csv",
    "bank_committee_scores.csv",
    "meeting_hawkometer_scores.csv",
    "hawkometer_summary.csv",
    "hawkometer.json",
    "bank_trends.csv",
    "speaker_shift_alerts.csv",
    "meeting_cycle_analysis.csv",
    "cross_bank_ranking.csv",
    "document_type_breakdown.csv",
    "hawkometer_dashboard_data.json",
    "analytics_summary.csv",
    "policy_decisions.csv",
    "policy_decisions_qa.csv",
    "policy_decisions_review.csv",
    "policy_decisions_suggested_corrections.csv",
    "policy_decisions_overrides_template.csv",
    "policy_decisions_review_detailed.csv",
    "policy_decisions_review_detailed.xlsx",
    "policy_decisions_final.csv",
    "policy_decisions_qc_summary.csv",
    "policy_decisions_suggested_corrections.csv",
    "policy_decisions_overrides_from_detailed_template.csv",
    "policy_decisions_review_detailed_summary.csv",
    "policy_decisions_overrides.csv",
    "README_policy_decisions.md",
    "README_policy_decisions_qc.md",
    "README_policy_decisions_review_detailed.md",
]

CLEAN_FOLDERS = [
    "processed",
    "data",
    "data/external",
    "scripts",
    "configs",
    "logs",
]


@dataclass
class ResetPlan:
    timestamp: str
    archive_dir: Path
    raw_preserved: list[str] = field(default_factory=list)
    files_to_move: list[Path] = field(default_factory=list)
    folders_to_move: list[Path] = field(default_factory=list)
    files_not_found: list[str] = field(default_factory=list)
    folders_not_found: list[str] = field(default_factory=list)
    moved_files: list[Path] = field(default_factory=list)
    moved_folders: list[Path] = field(default_factory=list)
    blocked_items: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset project to fresh raw-data state.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions only. This is the default.")
    parser.add_argument("--confirm-reset", action="store_true", help="Actually move generated files/folders to an archive.")
    return parser.parse_args()


def check_raw_folders(root: Path) -> list[str]:
    missing = []
    for bank in RAW_BANK_FOLDERS:
        path = root / "data" / bank
        if not path.is_dir():
            missing.append(str(path))
    return missing


def unique_existing(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    output: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        output.append(path)
    return output


def build_plan(root: Path) -> ResetPlan:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan = ResetPlan(timestamp=timestamp, archive_dir=root / f"_archive_reset_{timestamp}")
    plan.raw_preserved = [str(root / "data" / bank) for bank in RAW_BANK_FOLDERS]

    for script in GENERATED_SCRIPTS:
        path = root / script
        if path.exists():
            plan.files_to_move.append(path)
        else:
            plan.files_not_found.append(script)

    for folder in GENERATED_FOLDERS:
        path = root / folder
        if path.exists():
            plan.folders_to_move.append(path)
        else:
            plan.folders_not_found.append(folder)

    search_roots = [root / "processed", root / "data" / "external"]
    for filename in GENERATED_FILE_NAMES:
        found = []
        for search_root in search_roots:
            if search_root.exists():
                found.extend(path for path in search_root.rglob(filename) if path.is_file())
        if found:
            plan.files_to_move.extend(found)
        else:
            plan.files_not_found.append(filename)

    folder_resolved = [folder.resolve() for folder in plan.folders_to_move]
    filtered_files = []
    for path in plan.files_to_move:
        resolved = path.resolve()
        if any(is_relative_to(resolved, folder) for folder in folder_resolved):
            continue
        filtered_files.append(path)
    plan.files_to_move = unique_existing(filtered_files)
    plan.folders_to_move = unique_existing(plan.folders_to_move)
    return plan


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def archive_destination(root: Path, archive_dir: Path, source: Path) -> Path:
    relative = source.relative_to(root)
    return archive_dir / relative


def move_path(root: Path, archive_dir: Path, source: Path) -> Path:
    destination = archive_destination(root, archive_dir, source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        suffix = datetime.now().strftime("%H%M%S%f")
        destination = destination.with_name(f"{destination.name}.{suffix}")
    shutil.move(str(source), str(destination))
    return destination


def execute_plan(root: Path, plan: ResetPlan, dry_run: bool) -> None:
    print(f"Archive folder: {plan.archive_dir}")
    print("Raw bank folders preserved:")
    for item in plan.raw_preserved:
        print(f"  PRESERVE {item}")

    print("Planned folder moves:")
    for folder in plan.folders_to_move:
        print(f"  MOVE {folder} -> {archive_destination(root, plan.archive_dir, folder)}")

    print("Planned file moves:")
    for file in plan.files_to_move:
        print(f"  MOVE {file} -> {archive_destination(root, plan.archive_dir, file)}")

    if dry_run:
        print("Dry run only. No files were moved.")
        return

    plan.archive_dir.mkdir(parents=True, exist_ok=False)
    for folder in plan.folders_to_move:
        if folder.exists():
            try:
                plan.moved_folders.append(move_path(root, plan.archive_dir, folder))
            except (OSError, PermissionError) as exc:
                plan.blocked_items.append(f"{folder}: {exc}")
                move_folder_contents_best_effort(root, plan, folder)
    for file in plan.files_to_move:
        if file.exists():
            try:
                plan.moved_files.append(move_path(root, plan.archive_dir, file))
            except (OSError, PermissionError) as exc:
                plan.blocked_items.append(f"{file}: {exc}")

    for folder in CLEAN_FOLDERS:
        (root / folder).mkdir(parents=True, exist_ok=True)

    write_report(root, plan)
    print("Reset complete.")
    if plan.blocked_items:
        print("Some items were blocked by Windows and remain in place:")
        for item in plan.blocked_items:
            print(f"  {item}")
    print(f"Report written to: {root / 'reset_report.md'}")


def move_folder_contents_best_effort(root: Path, plan: ResetPlan, folder: Path) -> None:
    for child in sorted(folder.rglob("*"), reverse=True):
        if not child.exists():
            continue
        if child.is_dir():
            try:
                child.rmdir()
            except OSError:
                pass
            continue
        try:
            plan.moved_files.append(move_path(root, plan.archive_dir, child))
        except (OSError, PermissionError) as exc:
            plan.blocked_items.append(f"{child}: {exc}")
    try:
        folder.rmdir()
    except OSError:
        pass


def write_report(root: Path, plan: ResetPlan) -> None:
    raw_checks = {bank: (root / "data" / bank).is_dir() for bank in RAW_BANK_FOLDERS}
    master_existed = (root / "data" / "master_index.csv").exists()
    control_existed = (root / "data" / "control.json").exists()
    processed_items = sorted(p.name for p in (root / "processed").iterdir()) if (root / "processed").exists() else []
    external_items = sorted(p.name for p in (root / "data" / "external").iterdir()) if (root / "data" / "external").exists() else []

    lines = [
        "# Reset Report",
        "",
        f"Timestamp: {plan.timestamp}",
        f"Archive folder: `{plan.archive_dir}`",
        "",
        "## Raw Folders Preserved",
        "",
        *[f"- `{path}`" for path in plan.raw_preserved],
        "",
        "## Files Moved To Archive",
        "",
        *([f"- `{path}`" for path in plan.moved_files] or ["- None"]),
        "",
        "## Folders Moved To Archive",
        "",
        *([f"- `{path}`" for path in plan.moved_folders] or ["- None"]),
        "",
        "## Files Not Found",
        "",
        *([f"- `{item}`" for item in sorted(set(plan.files_not_found))] or ["- None"]),
        "",
        "## Folders Not Found",
        "",
        *([f"- `{item}`" for item in sorted(set(plan.folders_not_found))] or ["- None"]),
        "",
        "## Final Clean Folder Structure",
        "",
        *[f"- `{folder}/`" for folder in CLEAN_FOLDERS],
        "",
        "## Blocked Items",
        "",
        *([f"- `{item}`" for item in plan.blocked_items] or ["- None"]),
        "",
        "## Safety Checklist",
        "",
        *[f"- Raw folder `{bank}` exists: `{exists}`" for bank, exists in raw_checks.items()],
        f"- `data/master_index.csv` exists: `{master_existed}`",
        f"- `data/control.json` exists: `{control_existed}`",
        f"- `processed/` contents: `{processed_items}`",
        f"- `data/external/` contents: `{external_items}`",
    ]
    (root / "reset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    dry_run = not args.confirm_reset
    if args.dry_run and args.confirm_reset:
        raise SystemExit("Use either --dry-run or --confirm-reset, not both.")

    missing_raw = check_raw_folders(root)
    if missing_raw:
        print("ERROR: Aborting reset. Missing raw bank folders:")
        for path in missing_raw:
            print(f"  {path}")
        raise SystemExit(1)

    plan = build_plan(root)
    execute_plan(root, plan, dry_run=dry_run)


if __name__ == "__main__":
    main()
