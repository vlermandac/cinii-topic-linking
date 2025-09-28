import os
import csv
import curses
import random
from pathlib import Path
from textwrap import wrap
import tomllib
from rdflib import Graph, Namespace
from parser import parse_and_extract_articles_langs_from_dirs

from rich.markup import render as rich_render
from rich.style import Style
from rich.color import Color


BASE_COLORS = {
    "black":   curses.COLOR_BLACK,
    "red":     curses.COLOR_RED,
    "green":   curses.COLOR_GREEN,
    "yellow":  curses.COLOR_YELLOW,
    "blue":    curses.COLOR_BLUE,
    "magenta": curses.COLOR_MAGENTA,
    "cyan":    curses.COLOR_CYAN,
    "white":   curses.COLOR_WHITE,
}


def _nearest_base(color: Color | None) -> str:
    if not color:
        return "white"
    if color.name in BASE_COLORS:
        return color.name
    r, g, b = color.get_truecolor()
    if r == g == b:
        return "white" if r >= 128 else "black"
    m = max(r, g, b)
    if m == r:
        return "yellow" if g >= 160 else "red"
    if m == g:
        return "cyan" if b >= 160 else "green"
    return "magenta" if r >= 160 else "blue"


class RichCurses:
    """Minimal renderer to draw Rich-markup strings inside curses."""

    def __init__(self) -> None:
        self.pairs: dict[str, int] = {}

    def setup(self) -> None:
        curses.start_color()
        try:
            curses.use_default_colors()
            bg = -1
        except curses.error:
            bg = curses.COLOR_BLACK
        pid = 1
        for name, c in BASE_COLORS.items():
            curses.init_pair(pid, c, bg)
            self.pairs[name] = pid
            pid += 1

    def add_rich(self, win, y: int, x: int, markup: str, max_width: int | None = None) -> int:
        """
        Render a single line of Rich markup into a curses window at (y, x).
        Returns the x position after the write.
        """
        text = rich_render(markup)  # rich.text.Text
        plain = text.plain
        pos = 0
        width = max_width or len(plain)

        for span in text.spans:
            if span.start > pos:
                seg = plain[pos:span.start]
                if seg:
                    win.addnstr(y, x, seg, width)
                    x += len(seg)
                pos = span.start

            seg = plain[span.start:span.end]
            if not seg:
                continue

            style_obj = span.style
            if isinstance(style_obj, str):
                try:
                    style = Style.parse(style_obj)
                except Exception:
                    style = Style.null()
            elif isinstance(style_obj, Style):
                style = style_obj
            else:
                style = Style.null()

            attr = curses.A_NORMAL
            if style.bold:
                attr |= curses.A_BOLD
            if style.underline:
                attr |= curses.A_UNDERLINE
            # italics/strike are not supported by curses

            fg = style.color
            name = _nearest_base(fg)
            pair = self.pairs.get(name, self.pairs["white"])
            attr |= curses.color_pair(pair)

            win.addnstr(y, x, seg, width, attr)
            x += len(seg)
            pos = span.end

        # trailing plain text
        if pos < len(plain):
            seg = plain[pos:]
            win.addnstr(y, x, seg, width)
            x += len(seg)

        return x


os.environ.setdefault("TQDM_DISABLE", "1")

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.toml"

with open(CONFIG_PATH, "rb") as f:
    config = tomllib.load(f)

LABELS_CSV = Path(config["paths"]["benchmark"]) / "article_labels.csv"
TTL_PATH = Path(config["paths"]["benchmark"]) / "oecd_fos_2007.ttl"
RDF_DIRS = [ROOT_DIR / Path(p) for p in config["paths"]["rdf_dirs"]]
# Or the smaller sample for testing:
# RDF_DIRS = [ROOT_DIR / Path(config["paths"]["test_sample"])]

HEADER_SENTINEL = "__HEADER__"


def parse_topic_hierarchy(ttl_path: Path) -> tuple[list[tuple[str, list[str]]], dict[str, str]]:
    """
    Return:
      - majors: list of (major_label, [subtopic_label]) from SKOS ConceptScheme.
      - defs:   dict label -> skos:definition (string), if available.
    """
    g = Graph()
    g.parse(str(ttl_path), format="turtle" if ttl_path.suffix == ".ttl" else "xml")
    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

    majors: list[tuple[str, list[str]]] = []
    defs: dict[str, str] = {}

    for subj in g.subjects(SKOS.topConceptOf, None):
        major_label_lit = g.value(subj, SKOS.prefLabel)
        if not major_label_lit:
            continue
        major_label = str(major_label_lit)
        major_def = g.value(subj, SKOS.definition)
        if major_def:
            defs[major_label] = str(major_def)

        subs: list[str] = []
        for s in g.objects(subject=subj, predicate=SKOS.narrower):
            sublabel_lit = g.value(s, SKOS.prefLabel)
            if sublabel_lit:
                sublabel = str(sublabel_lit)
                subs.append(sublabel)
                sub_def = g.value(s, SKOS.definition)
                if sub_def:
                    defs[sublabel] = str(sub_def)

        if not subs:
            for s in g.subjects(SKOS.broader, subj):
                sublabel_lit = g.value(s, SKOS.prefLabel)
                if sublabel_lit:
                    sublabel = str(sublabel_lit)
                    subs.append(sublabel)
                    sub_def = g.value(s, SKOS.definition)
                    if sub_def:
                        defs[sublabel] = str(sub_def)

        majors.append((major_label, sorted(subs, key=lambda x: x.lower())))

    majors.sort(key=lambda t: t[0].lower())
    return majors, defs


def load_existing_labels(path: Path) -> set[str]:
    labeled: set[str] = set()
    if path.exists() and path.stat().st_size:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labeled.add((row["uri"] or "").strip())
    return labeled


def build_choices(majors: list[tuple[str, list[str]]]) -> tuple[list[tuple[str, str | None]], list[int]]:
    """Flatten to rows for the middle pane"""
    rows: list[tuple[str, str | None]] = []
    selectable_idx: list[int] = []

    for major, subs in majors:
        rows.append((major, HEADER_SENTINEL))  # non-selectable header
        for sub in subs:
            rows.append((major, sub))
            selectable_idx.append(len(rows) - 1)
        rows.append((major, None))  # per-major 'only major topic'
        selectable_idx.append(len(rows) - 1)

    # single global escape at the very end
    if not rows or rows[-1] != (None, None):
        rows.append((None, None))
        selectable_idx.append(len(rows) - 1)

    return rows, selectable_idx


def clamp(v, lo, hi): return max(lo, min(hi, v))


def draw_top(rc: RichCurses, win, width: int, lines: list[str]) -> None:
    win.erase()
    y = 0
    for line in lines:
        rc.add_rich(win, y, 0, line, max_width=width-1)
        y += 1
        if y >= win.getmaxyx()[0]:
            break
    win.noutrefresh()


def draw_middle(
    rc: RichCurses,
    win,
    width: int,
    height: int,
    rows: list[tuple[str, str | None]],
    selectable_idx: list[int],
    sel_idx: int,
    scroll_top: int,
) -> None:
    """
    Draw middle list. sel_idx is absolute index into rows.
    scroll_top: first visible row index.
    """
    win.erase()
    end = min(len(rows), scroll_top + height)
    for i, row_idx in enumerate(range(scroll_top, end)):
        major, sub = rows[row_idx]
        y = i

        if major is None and sub is None:
            label = "[red][b]Could not identify topic[/b][/red]"
            is_header = False
        elif sub is None:
            label = f"[dim]{major} — only major topic[/dim]"
            is_header = False
        elif sub == HEADER_SENTINEL:
            label = f"[bold magenta]{major}[/]"
            is_header = True
        else:
            label = f"{sub}"
            is_header = False

        is_selected = (row_idx == sel_idx) and (row_idx in selectable_idx)

        if is_selected:
            win.attron(curses.A_REVERSE)
            rc.add_rich(win, y, 0, f"> {label}", max_width=width-1)
            win.attroff(curses.A_REVERSE)
        else:
            prefix = "  " if not is_header else ""
            rc.add_rich(win, y, 0, f"{prefix}{label}", max_width=width-1)

    win.noutrefresh()


def draw_bottom(rc: RichCurses, win, width: int, confidence: str) -> None:
    win.erase()
    msg = (
        f"[bold]Confidence[/]: [cyan]{confidence.upper()}[/]  "
        "Keys: [bold]↑/↓[/], [bold]PgUp/PgDn[/], [bold]Home/End[/], "
        "[bold]Ctrl+U[/]/[bold]Ctrl+D[/]=half page, "
        "[bold]?[/]=definition, "
        "[bold]Enter[/]=select, [bold]h/m/l[/]=confidence, [bold]q[/]=quit"
    )
    rc.add_rich(win, 0, 0, msg, max_width=width-1)
    win.noutrefresh()


def show_definition_popup(stdscr, rc: RichCurses, title: str, text: str | None) -> None:
    """Centered popup with a border and wrapped definition text"""
    H, W = stdscr.getmaxyx()
    w = min(W - 4, max(50, int(W * 0.8)))
    h = min(H - 4, max(8, int(H * 0.6)))
    y = (H - h) // 2
    x = (W - w) // 2

    win = stdscr.derwin(h, w, y, x)
    win.erase()
    win.border()

    # title
    title_str = f" Definition — {title} "
    rc.add_rich(win, 0, max(2, (w - len(title_str)) // 2),
                f"[bold yellow]{title_str}[/]", max_width=w-4)

    # body
    body = text.strip() if text else "(No definition available.)"
    lines: list[str] = []
    for line in body.splitlines() or [""]:
        lines += wrap(line, w - 4) or [""]

    max_body = h - 3
    for i, ln in enumerate(lines[:max_body]):
        rc.add_rich(win, 1 + i, 2, ln, max_width=w - 4)

    rc.add_rich(win, h - 2, 2,
                "[dim]Press any key to close[/dim]", max_width=w - 4)
    win.noutrefresh()
    curses.doupdate()
    stdscr.getch()  # wait


def label_loop(stdscr, articles, majors, defs: dict[str, str], csv_path: Path):
    # init colors
    rc = RichCurses()
    rc.setup()

    curses.curs_set(0)  # hide cursor
    stdscr.nodelay(False)
    stdscr.keypad(True)

    rows, selectable_idx = build_choices(majors)

    # CSV setup
    write_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    f = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        f, fieldnames=["uri", "major_topic", "subtopic", "confidence"])
    if write_header:
        writer.writeheader()

    confidence = "high"  # default to high
    total = len(articles)
    done = 0

    for uri, art in articles:
        title = (art.get("title") or "").strip()
        abstract = (art.get("abstract") or "").strip()

        # window layout
        H, W = stdscr.getmaxyx()
        prompt_h = 1

        # build the text first, so we know how tall top needs to be
        top_lines: list[str] = []
        top_lines.append(f"[bold cyan]Progress:[/] {done}/{total}    "
                         f"[dim](h/m/l to set confidence; Enter to select; q to quit)[/dim]")
        wrap_width = max(10, W - 2)
        top_lines.append(f"[bold yellow]URI:[/] {uri}")
        top_lines.append("[bold blue]Title[/]:")
        for seg in wrap(title, wrap_width):
            top_lines.append(seg)

        top_lines.append("[bold blue]Abstract[/]:")

        for line in abstract.splitlines() or [""]:
            if not line:
                top_lines.append("")  # preserve paragraph breaks minimally
            else:
                top_lines.extend(wrap(line, wrap_width))

        # size panes so the title+abstract is visible
        min_mid_h = 3  # keep the list tiny if the text is huge
        max_top_h = max(1, H - prompt_h - min_mid_h)
        top_h = min(max_top_h, len(top_lines))
        mid_h = H - top_h - prompt_h
        if mid_h < min_mid_h:
            mid_h = min_mid_h
            top_h = H - mid_h - prompt_h  # re-adjust top

        # create windows
        top_win = stdscr.derwin(top_h, W, 0, 0)
        mid_win = stdscr.derwin(mid_h, W, top_h, 0)
        bot_win = stdscr.derwin(prompt_h, W, H - prompt_h, 0)

        # selection state for middle list
        sel_pos = 0
        sel_idx = selectable_idx[sel_pos]
        scroll_top = 0

        while True:
            stdscr.erase()
            draw_top(rc, top_win, W, top_lines)

            # visual improvments

            # keep selection visible
            desired_top = scroll_top
            if sel_idx < scroll_top:
                desired_top = sel_idx
            elif sel_idx >= scroll_top + mid_h:
                desired_top = sel_idx - mid_h + 1

            prev_idx = sel_idx - 1
            if 0 <= prev_idx < len(rows) and rows[prev_idx][1] == HEADER_SENTINEL:
                if prev_idx < desired_top:
                    desired_top = prev_idx

            if sel_pos == 0:
                desired_top = 0

            scroll_top = desired_top

            draw_middle(rc, mid_win, W, mid_h, rows,
                        selectable_idx, sel_idx, scroll_top)
            draw_bottom(rc, bot_win, W, confidence)

            curses.doupdate()
            ch = stdscr.getch()

            n = len(selectable_idx)
            HALF = max(1, mid_h // 2)

            if ch in (ord('q'), ord('Q')):
                f.flush()
                f.close()
                return  # quit program

            elif ch in (ord('h'), ord('H')):
                confidence = "high"
            elif ch in (ord('m'), ord('M')):
                confidence = "medium"
            elif ch in (ord('l'), ord('L')):
                confidence = "low"

            elif ch in (curses.KEY_UP, ord('k'), ord('K')):
                sel_pos = (sel_pos - 1) % n
                sel_idx = selectable_idx[sel_pos]

            elif ch in (curses.KEY_DOWN, ord('j'), ord('J')):
                sel_pos = (sel_pos + 1) % n
                sel_idx = selectable_idx[sel_pos]

            elif ch == curses.KEY_NPAGE:  # Page Down
                jump = max(1, mid_h - 1)
                sel_pos = (sel_pos + jump) % n
                sel_idx = selectable_idx[sel_pos]

            elif ch == curses.KEY_PPAGE:  # Page Up
                jump = max(1, mid_h - 1)
                sel_pos = (sel_pos - jump) % n
                sel_idx = selectable_idx[sel_pos]

            elif ch == curses.KEY_HOME:
                sel_pos = 0
                sel_idx = selectable_idx[sel_pos]

            elif ch == curses.KEY_END:
                sel_pos = n - 1
                sel_idx = selectable_idx[sel_pos]

            elif ch == 21:  # Ctrl+U
                sel_pos = (sel_pos - HALF) % n
                sel_idx = selectable_idx[sel_pos]

            elif ch == 4:   # Ctrl+D
                sel_pos = (sel_pos + HALF) % n
                sel_idx = selectable_idx[sel_pos]

            elif ch == ord('?'):
                major, sub = rows[sel_idx]
                if major is None and sub is None:
                    label_lookup = None
                elif sub is None:
                    label_lookup = major
                else:
                    label_lookup = sub
                definition = defs.get(label_lookup or "", None)
                show_definition_popup(
                    stdscr, rc, label_lookup or "N/A", definition)

            elif ch in (curses.KEY_ENTER, 10, 13):
                major, sub = rows[sel_idx]
                if major is None and sub is None:
                    row = {"uri": (uri or "").strip(), "major_topic": "",
                           "subtopic": "", "confidence": "skip"}
                else:
                    row = {"uri": (uri or "").strip(), "major_topic": major,
                           "subtopic": (sub or ""), "confidence": confidence}
                writer.writerow(row)
                f.flush()
                done += 1
                confidence = "high"  # reset default
                break  # next paper
            # else: ignore other keys

    f.flush()
    f.close()


def main():
    majors, defs = parse_topic_hierarchy(TTL_PATH)

    # load and filter articles
    eng_articles, _ = parse_and_extract_articles_langs_from_dirs(RDF_DIRS)
    labeled = load_existing_labels(LABELS_CSV)
    items = [(u, d) for u, d in eng_articles.items()
             if (u or "").strip() not in labeled]

    # get only articles with non-empty abstracts
    items = [(u, d) for u, d in items if (d.get("abstract") or "").strip()]

    rng = random.Random()
    rng.shuffle(items)  # randomize order

    if not items:
        print("No unlabeled articles found.")
        return

    curses.wrapper(label_loop, items, majors, defs, LABELS_CSV)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
