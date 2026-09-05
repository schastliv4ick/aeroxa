"""Ingestion CLI.

    python -m ingest.cli list
    python -m ingest.cli probe 102051516
    python -m ingest.cli find --from 102033000 --to 102035000 --pattern "Гражданский кодекс"
    python -m ingest.cli find --act gk-rf-1
    python -m ingest.cli run --act fz-14-ooo
    python -m ingest.cli run --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys

from ingest.indexer import ActSpec, ingest_act, load_acts
from ingest.ips_client import IpsClient, scan_titles

log = logging.getLogger("ingest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ingest", description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show the act registry and which ids are resolved")

    probe = sub.add_parser("probe", help="fetch one document's title and redaction list")
    probe.add_argument("nd")

    find = sub.add_parser("find", help="scan the nd id space for a title (search is broken upstream)")
    find.add_argument("--act", help="use the scan_hint range and title_match from acts.yaml")
    find.add_argument("--from", dest="nd_from", type=int)
    find.add_argument("--to", dest="nd_to", type=int)
    find.add_argument("--pattern")
    find.add_argument("--concurrency", type=int, default=8)

    run = sub.add_parser("run", help="fetch, chunk, embed and index acts")
    run.add_argument("--act", action="append", help="act id; repeatable. Default: all resolved acts")
    run.add_argument("--dry-run", action="store_true", help="parse and chunk but do not index")
    run.add_argument("--no-cache", action="store_true", help="ignore the on-disk HTML cache")

    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)-8s %(name)s | %(message)s")

    handlers = {"list": _cmd_list, "probe": _cmd_probe, "find": _cmd_find, "run": _cmd_run}
    return handlers[args.command](args)


def _cmd_list(_: argparse.Namespace) -> int:
    for spec in load_acts():
        marker = spec.nd or "UNRESOLVED"
        hint = f"  scan_hint={spec.scan_hint}" if spec.nd is None and spec.scan_hint else ""
        print(f"{spec.id:24} {marker:>12}  {spec.title_match}{hint}")
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    with IpsClient() as client:
        title = client.fetch_title(args.nd)
        if not title:
            print(f"nd={args.nd}: not found")
            return 1
        revisions = client.fetch_revisions(args.nd)
        current = revisions[-1]
        print(f"nd={args.nd}")
        print(f"title:     {title}")
        print(f"redactions: {len(revisions)} (0..{current.index})")
        print(f"current:   {current.index} от {current.date or 'исходная'} {current.amending_act or ''}")

        document = client.fetch_document(args.nd, current)
        from ingest.parser import parse_articles

        articles = parse_articles(document.html)
        repealed = sum(article.repealed for article in articles)
        print(f"articles:  {len(articles)} ({repealed} repealed)")
        for article in articles[:5]:
            print(f"  Статья {article.number}. {article.title[:70]}")
    return 0


def _cmd_find(args: argparse.Namespace) -> int:
    if args.act:
        spec = _require_act(args.act)
        if not spec.scan_hint:
            print(f"{spec.id}: no scan_hint in acts.yaml; pass --from/--to")
            return 1
        nd_from, nd_to, pattern = spec.scan_hint[0], spec.scan_hint[1], spec.title_match
    else:
        if args.nd_from is None or args.nd_to is None or not args.pattern:
            print("provide --act, or all of --from --to --pattern")
            return 1
        nd_from, nd_to, pattern = args.nd_from, args.nd_to, args.pattern

    span = nd_to - nd_from + 1
    print(f"scanning {span} ids for {pattern!r} — this makes {span} requests")
    hits = scan_titles(nd_from, nd_to, pattern, concurrency=args.concurrency)
    if not hits:
        print("no match; widen the range")
        return 1
    for nd, title in hits:
        print(f"  nd={nd}  {title}")
    print("\nPut the correct id into ingest/acts.yaml.")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    specs = load_acts()
    if args.act:
        wanted = set(args.act)
        specs = [spec for spec in specs if spec.id in wanted]
        missing = wanted - {spec.id for spec in specs}
        if missing:
            print(f"unknown act id(s): {', '.join(sorted(missing))}")
            return 1

    failures = 0
    with IpsClient() as client:
        for spec in specs:
            try:
                report = ingest_act(
                    spec, client, use_cache=not args.no_cache, dry_run=args.dry_run
                )
            except Exception as error:  # noqa: BLE001 - one bad act must not stop the run
                log.error("%s failed: %s", spec.id, error)
                failures += 1
                continue

            if report.skipped and report.reason and report.reason != "dry run":
                print(f"{report.act_id:24} SKIPPED  {report.reason}")
                continue
            prefix = "would index" if args.dry_run else "indexed"
            print(
                f"{report.act_id:24} {prefix} {report.chunks:5} chunks "
                f"from {report.articles:4} articles  "
                f"(nd={report.nd} rdk={report.revision_index} {report.revision_date or 'original'})"
            )
    return 1 if failures else 0


def _require_act(act_id: str) -> ActSpec:
    for spec in load_acts():
        if spec.id == act_id:
            return spec
    raise SystemExit(f"unknown act id: {act_id}")


if __name__ == "__main__":
    sys.exit(main())
