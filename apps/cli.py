"""``jevpilot`` command: dispatches to the applications in :mod:`apps`.

jevpilot uav-materials --demo
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

COMMANDS: dict[str, tuple[str, str]] = {
    "uav-materials": (
        "apps.uav_materials.cli",
        "UAV material selection and composite design, from a natural-language request",
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print("usage: jevpilot <command> [options]\n\ncommands:")
        for name, (_, help_text) in COMMANDS.items():
            print(f"  {name:16s} {help_text}")
        print("\nRun 'jevpilot <command> --help' for the command's options.")
        return 0 if args else 2
    if args[0] not in COMMANDS:
        print(f"jevpilot: unknown command {args[0]!r}; try 'jevpilot --help'", file=sys.stderr)
        return 2
    import importlib

    module = importlib.import_module(COMMANDS[args[0]][0])
    run: Callable[[list[str]], int] = module.main
    return run(args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
