from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


__all__ = [
    "CFOPParseError",
    "CFOPPhase",
    "CFOPPhaseBoundary",
    "FACE_ORDER",
    "F2L_PAIR_ORDER",
    "FaceletStateError",
    "ParsedCFOPResponse",
    "STATE_FORMAT",
    "VALID_MOVE_TOKENS",
    "format_facelet_state",
    "normalize_facelet_state",
    "parse_cfop_phase_response",
    "parse_cfop_phase_response_strict",
    "parse_cfop_response",
    "parse_cfop_response_strict",
]

STATE_FORMAT = "facelets-urfdlb-v1"
FACE_ORDER = ("U", "R", "F", "D", "L", "B")
F2L_PAIR_ORDER = ("FR", "FL", "BL", "BR")

_MOVE_BASES = (
    "U",
    "R",
    "F",
    "D",
    "L",
    "B",
    "M",
    "E",
    "S",
    "Uw",
    "Rw",
    "Fw",
    "Dw",
    "Lw",
    "Bw",
    "x",
    "y",
    "z",
)
VALID_MOVE_TOKENS = frozenset(
    f"{base}{suffix}" for base in _MOVE_BASES for suffix in ("", "2", "'")
)
_SETUP_MOVE_TOKENS = frozenset(
    f"{base}{suffix}" for base in ("x", "y", "z") for suffix in ("", "2", "'")
)


class FaceletStateError(ValueError):
    """Raised when a value is not a canonical 54-facelet state."""


class CFOPParseError(ValueError):
    """A classified error from the strict CFOP response parser."""

    def __init__(self, message: str, *, code: str, line: int | None = None) -> None:
        self.code = code
        self.line = line
        prefix = f"line {line}: " if line is not None else ""
        super().__init__(f"{prefix}{message}")


@dataclass(frozen=True, slots=True)
class CFOPPhase:
    """One parsed phase in the model's full-solve response."""

    stage: str
    move_tokens: tuple[str, ...]
    f2l_pairs: tuple[str, ...] = ()

    @property
    def moves(self) -> str:
        return " ".join(self.move_tokens)

    @property
    def move_count(self) -> int:
        return len(self.move_tokens)

    @property
    def canonical_line(self) -> str:
        moves_suffix = f" {self.moves}" if self.moves else ""
        if self.stage == "f2l_pair":
            labels = ",".join(pair.upper() for pair in self.f2l_pairs)
            return f"F2L {labels}:{moves_suffix}"
        if self.stage == "cross" and self.f2l_pairs:
            labels = ",".join(pair.upper() for pair in self.f2l_pairs)
            return f"CROSS[{labels}]:{moves_suffix}"
        return f"{self.stage.upper()}:{moves_suffix}"


@dataclass(frozen=True, slots=True)
class CFOPPhaseBoundary:
    """An immutable phase boundary that can be converted for CFOPSolver."""

    stage: str
    after_move: int
    f2l_pairs: tuple[str, ...] = ()

    def to_verifier_dict(self) -> dict[str, object]:
        boundary: dict[str, object] = {
            "stage": self.stage,
            "after_move": self.after_move,
        }
        if self.f2l_pairs:
            boundary["f2l_pairs"] = list(self.f2l_pairs)
        return boundary


@dataclass(frozen=True, slots=True)
class ParsedCFOPResponse:
    """Non-throwing parse result suitable for environment feedback and rewards."""

    raw_response: str
    valid: bool
    phases: tuple[CFOPPhase, ...] = ()
    move_tokens: tuple[str, ...] = ()
    boundaries: tuple[CFOPPhaseBoundary, ...] = ()
    canonical_text: str = ""
    error: str | None = None
    error_code: str | None = None
    error_line: int | None = None

    @property
    def moves(self) -> str:
        return " ".join(self.move_tokens)

    @property
    def move_count(self) -> int:
        return len(self.move_tokens)

    @property
    def phase_boundaries(self) -> tuple[dict[str, object], ...]:
        """Return fresh, JSON-compatible mappings ready for ``verify_batch``."""

        return tuple(boundary.to_verifier_dict() for boundary in self.boundaries)


def normalize_facelet_state(state: str) -> str:
    """Return a validated compact state in canonical ``URFDLB`` position order.

    This checks the stable serialized shape without duplicating CFOPSolver's
    deeper cubie, orientation, centre-frame, and parity validation.
    """

    if not isinstance(state, str):
        raise FaceletStateError("cube state must be a string")
    normalized = state.strip()
    if len(normalized) != 54 or not normalized.isascii():
        raise FaceletStateError(
            f"cube state must contain exactly 54 ASCII facelets, got {len(normalized)}"
        )
    unexpected = sorted(set(normalized).difference(FACE_ORDER))
    if unexpected:
        raise FaceletStateError(
            "cube state may contain only uppercase URFDLB facelets; "
            f"found {unexpected!r}"
        )
    counts = Counter(normalized)
    invalid_counts = {
        face: counts.get(face, 0) for face in FACE_ORDER if counts.get(face, 0) != 9
    }
    if invalid_counts:
        raise FaceletStateError(
            "cube state must contain exactly nine of each URFDLB facelet; "
            f"invalid counts were {invalid_counts}"
        )
    try:
        import cfop_solver
    except ImportError as error:
        raise RuntimeError(
            "cfop_solver is required to validate physical cube states"
        ) from error
    try:
        cfop_solver.Cube(normalized)
    except (RuntimeError, ValueError) as error:
        raise FaceletStateError(f"cube state is not physically valid: {error}") from error
    return normalized


def format_facelet_state(state: str) -> str:
    """Format a canonical state as six labelled, row-grouped prompt lines."""

    normalized = normalize_facelet_state(state)
    lines = []
    for face_index, face in enumerate(FACE_ORDER):
        facelets = normalized[face_index * 9 : (face_index + 1) * 9]
        rows = "/".join(facelets[row : row + 3] for row in range(0, 9, 3))
        lines.append(f"{face}: {rows}")
    return "\n".join(lines)


def parse_cfop_response(response: str) -> ParsedCFOPResponse:
    """Parse a model response without allowing malformed output to escape."""

    raw_response = str(response)
    try:
        return parse_cfop_response_strict(raw_response)
    except CFOPParseError as error:
        return ParsedCFOPResponse(
            raw_response=raw_response,
            valid=False,
            error=str(error),
            error_code=error.code,
            error_line=error.line,
        )


def parse_cfop_phase_response(response: str) -> ParsedCFOPResponse:
    """Parse one stage-local CFOP response without raising format errors."""

    raw_response = str(response)
    try:
        return parse_cfop_phase_response_strict(raw_response)
    except CFOPParseError as error:
        return ParsedCFOPResponse(
            raw_response=raw_response,
            valid=False,
            error=str(error),
            error_code=error.code,
            error_line=error.line,
        )


def _validated_response_lines(
    response: str,
) -> tuple[str, str, list[str]]:
    input_response = str(response)
    if not input_response:
        raise CFOPParseError(
            "response is empty",
            code="empty_response",
        )
    if input_response.endswith("\r\n"):
        raw_response = input_response[:-2]
    elif input_response.endswith("\n"):
        raw_response = input_response[:-1]
    else:
        raw_response = input_response
    lines = raw_response.split("\n")
    if lines[-1] == "":
        raise CFOPParseError(
            "blank lines are not allowed",
            code="blank_line",
            line=len(lines),
        )
    for line_number, line in enumerate(lines, start=1):
        if line in {"", "\r"}:
            raise CFOPParseError(
                "blank lines are not allowed",
                code="blank_line",
                line=line_number,
            )
        if not line.isascii():
            raise CFOPParseError(
                "response must use ASCII headers, labels, and move tokens",
                code="non_ascii",
                line=line_number,
            )
        if line != line.strip() or "\t" in line:
            raise CFOPParseError(
                "leading, trailing, and tab whitespace are not allowed",
                code="invalid_whitespace",
                line=line_number,
            )
    terminal_cr_index = (
        len(input_response) - 2
        if input_response.endswith("\r\n")
        else None
    )
    for cr_index, character in enumerate(input_response):
        if character != "\r" or cr_index == terminal_cr_index:
            continue
        line_number = input_response.count("\n", 0, cr_index) + 1
        raise CFOPParseError(
            "carriage returns are allowed only in the optional terminal CRLF",
            code="invalid_whitespace",
            line=line_number,
        )
    return input_response, raw_response, lines


def parse_cfop_phase_response_strict(response: str) -> ParsedCFOPResponse:
    """Parse exactly one CFOP phase line followed by ``END``.

    The phase may be CROSS, one F2L transition, OLL, or PLL. This is an
    explicitly stage-local SFT/evaluation protocol, not a full-cube response.
    """

    _, raw_response, lines = _validated_response_lines(response)
    if len(lines) < 2:
        raise CFOPParseError(
            "expected one phase line followed by END",
            code="expected_phase_end",
            line=len(lines) + 1,
        )
    if len(lines) > 2:
        raise CFOPParseError(
            "stage-local responses must contain exactly one phase line and END",
            code="trailing_content",
            line=3,
        )
    if lines[1] != "END":
        raise CFOPParseError(
            "expected exact END marker",
            code="expected_end",
            line=2,
        )

    phase_line = lines[0]
    if phase_line.startswith("CROSS"):
        phase = _parse_cross_phase(phase_line, 1)
    elif phase_line.startswith("F2L"):
        phase = _parse_f2l_phase(phase_line, 1)
    elif phase_line.startswith("OLL"):
        phase = _parse_named_phase(phase_line, 1, "OLL", "oll")
    elif phase_line.startswith("PLL"):
        phase = _parse_named_phase(phase_line, 1, "PLL", "pll")
    else:
        raise CFOPParseError(
            "expected CROSS, F2L, OLL, or PLL phase line",
            code="expected_phase",
            line=1,
        )

    boundary = CFOPPhaseBoundary(
        stage=phase.stage,
        after_move=phase.move_count,
        f2l_pairs=phase.f2l_pairs,
    )
    canonical_text = f"{phase.canonical_line}\nEND"
    return ParsedCFOPResponse(
        raw_response=raw_response,
        valid=True,
        phases=(phase,),
        move_tokens=phase.move_tokens,
        boundaries=(boundary,),
        canonical_text=canonical_text,
    )


def parse_cfop_response_strict(response: str) -> ParsedCFOPResponse:
    """Parse the exact full-solve grammar, raising :class:`CFOPParseError`.

    Grammar::

        [SETUP: <whole-cube rotations>]
        CROSS: <moves> | CROSS[<solved-pair>[,<solved-pair>...]]: <moves>
        [F2L <pair>[,<pair>...]: <non-empty moves>]*
        OLL: <moves>
        PLL: <moves>
        END

    Headers and pair labels are uppercase. CROSS, OLL, and PLL move lists may be
    empty. Every emitted F2L line is non-empty. Non-empty lists use one ASCII
    space between validated Singmaster tokens. Exactly one terminal LF or CRLF
    is allowed and is not part of the canonical response.
    """

    _, raw_response, lines = _validated_response_lines(response)
    phases: list[CFOPPhase] = []
    cursor = 0

    if lines[cursor].startswith("SETUP"):
        setup = _parse_named_phase(lines[cursor], cursor + 1, "SETUP", "setup")
        invalid_setup_moves = [
            token for token in setup.move_tokens if token not in _SETUP_MOVE_TOKENS
        ]
        if invalid_setup_moves:
            raise CFOPParseError(
                "SETUP may contain only whole-cube x, y, or z rotations",
                code="invalid_setup_move",
                line=cursor + 1,
            )
        phases.append(setup)
        cursor += 1

    cross_line = _line_at(lines, cursor, expected="CROSS")
    phases.append(_parse_cross_phase(cross_line, cursor + 1))
    cursor += 1

    seen_pairs: set[str] = set()
    while cursor < len(lines) and lines[cursor].startswith("F2L"):
        phase = _parse_f2l_phase(lines[cursor], cursor + 1)
        duplicate_pairs = seen_pairs.intersection(phase.f2l_pairs)
        if duplicate_pairs:
            labels = ",".join(pair.upper() for pair in sorted(duplicate_pairs))
            raise CFOPParseError(
                f"F2L pair labels may appear only once; repeated {labels}",
                code="duplicate_f2l_pair",
                line=cursor + 1,
            )
        seen_pairs.update(phase.f2l_pairs)
        phases.append(phase)
        cursor += 1

    oll_line = _line_at(lines, cursor, expected="OLL")
    phases.append(_parse_named_phase(oll_line, cursor + 1, "OLL", "oll"))
    cursor += 1

    pll_line = _line_at(lines, cursor, expected="PLL")
    phases.append(_parse_named_phase(pll_line, cursor + 1, "PLL", "pll"))
    cursor += 1

    end_line = _line_at(lines, cursor, expected="END")
    if end_line != "END":
        raise CFOPParseError(
            "expected exact END marker",
            code="expected_end",
            line=cursor + 1,
        )
    cursor += 1
    if cursor != len(lines):
        raise CFOPParseError(
            "content after END is not allowed",
            code="trailing_content",
            line=cursor + 1,
        )

    move_tokens = tuple(token for phase in phases for token in phase.move_tokens)
    offset = 0
    boundaries = []
    for phase in phases:
        offset += phase.move_count
        boundaries.append(
            CFOPPhaseBoundary(
                stage=phase.stage,
                after_move=offset,
                f2l_pairs=phase.f2l_pairs,
            )
        )
    canonical_text = "\n".join(
        [*(phase.canonical_line for phase in phases), "END"]
    )
    return ParsedCFOPResponse(
        raw_response=raw_response,
        valid=True,
        phases=tuple(phases),
        move_tokens=move_tokens,
        boundaries=tuple(boundaries),
        canonical_text=canonical_text,
    )


def _line_at(lines: list[str], index: int, *, expected: str) -> str:
    if index >= len(lines):
        raise CFOPParseError(
            f"expected {expected} line",
            code=f"expected_{expected.lower()}",
            line=index + 1,
        )
    return lines[index]


def _parse_named_phase(
    line: str,
    line_number: int,
    header: str,
    verifier_stage: str,
) -> CFOPPhase:
    prefix = f"{header}:"
    if line == prefix:
        move_tokens: tuple[str, ...] = ()
    elif line.startswith(f"{prefix} "):
        move_tokens = _parse_moves(line[len(prefix) + 1 :], line_number)
    else:
        raise CFOPParseError(
            f"expected {header}: <moves>",
            code=f"expected_{header.lower()}",
            line=line_number,
        )
    return CFOPPhase(stage=verifier_stage, move_tokens=move_tokens)


def _parse_cross_phase(line: str, line_number: int) -> CFOPPhase:
    if line.startswith("CROSS:"):
        return _parse_named_phase(line, line_number, "CROSS", "cross")
    if not line.startswith("CROSS["):
        raise CFOPParseError(
            "expected CROSS: <moves> or CROSS[FR,FL,...]: <moves>",
            code="expected_cross",
            line=line_number,
        )

    separator = line.find("]:")
    if separator < 0:
        raise CFOPParseError(
            "annotated CROSS line must close pair labels with ]:",
            code="invalid_cross",
            line=line_number,
        )
    labels_text = line[len("CROSS[") : separator]
    labels = labels_text.split(",")
    if not labels_text or any(label not in F2L_PAIR_ORDER for label in labels):
        raise CFOPParseError(
            "CROSS labels must be comma-separated FR, FL, BL, or BR",
            code="invalid_cross_pair",
            line=line_number,
        )
    if len(set(labels)) != len(labels):
        raise CFOPParseError(
            "CROSS pair labels must be unique",
            code="invalid_cross_pair",
            line=line_number,
        )

    moves_text = line[separator + len("]:") :]
    if not moves_text:
        move_tokens: tuple[str, ...] = ()
    elif moves_text.startswith(" "):
        move_tokens = _parse_moves(moves_text[1:], line_number)
    else:
        raise CFOPParseError(
            "expected one space between CROSS pair labels and moves",
            code="invalid_cross",
            line=line_number,
        )
    return CFOPPhase(
        stage="cross",
        move_tokens=move_tokens,
        f2l_pairs=tuple(label.lower() for label in labels),
    )


def _parse_f2l_phase(line: str, line_number: int) -> CFOPPhase:
    if not line.startswith("F2L "):
        raise CFOPParseError(
            "expected F2L <pair>[,<pair>...]: <moves>",
            code="invalid_f2l",
            line=line_number,
        )
    label_and_moves = line[len("F2L ") :]
    separator = label_and_moves.find(":")
    if separator < 0:
        raise CFOPParseError(
            "F2L line is missing ':'",
            code="invalid_f2l",
            line=line_number,
        )
    labels_text = label_and_moves[:separator]
    moves_text = label_and_moves[separator + 1 :]
    labels = labels_text.split(",")
    if not labels_text or any(label not in F2L_PAIR_ORDER for label in labels):
        raise CFOPParseError(
            "F2L labels must be comma-separated FR, FL, BL, or BR",
            code="invalid_f2l_pair",
            line=line_number,
        )
    if len(set(labels)) != len(labels):
        raise CFOPParseError(
            "an F2L line may not repeat a pair label",
            code="duplicate_f2l_pair",
            line=line_number,
        )
    if moves_text == "":
        raise CFOPParseError(
            "F2L move lists must contain at least one move",
            code="empty_f2l_moves",
            line=line_number,
        )
    elif moves_text.startswith(" "):
        move_tokens = _parse_moves(moves_text[1:], line_number)
    else:
        raise CFOPParseError(
            "expected one space between ':' and a non-empty F2L move list",
            code="invalid_whitespace",
            line=line_number,
        )
    return CFOPPhase(
        stage="f2l_pair",
        move_tokens=move_tokens,
        f2l_pairs=tuple(label.lower() for label in labels),
    )


def _parse_moves(moves: str, line_number: int) -> tuple[str, ...]:
    tokens = tuple(moves.split(" "))
    if not moves or any(not token for token in tokens):
        raise CFOPParseError(
            "non-empty move lists must use one ASCII space between tokens",
            code="invalid_whitespace",
            line=line_number,
        )
    for token in tokens:
        if token not in VALID_MOVE_TOKENS:
            raise CFOPParseError(
                f"invalid move token {token!r}",
                code="invalid_move",
                line=line_number,
            )
    return tokens
