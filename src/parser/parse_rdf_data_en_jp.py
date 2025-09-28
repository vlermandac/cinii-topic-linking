# Parse rdf files URI, Title, and Abstract
# Output tables in Japanese and English

from pathlib import Path
from rdflib import Graph, URIRef, Literal
import os
from tqdm.auto import tqdm


def remove_html_tags(text):
    import re
    return re.sub(r'<.*?>', '', text)


def parse_and_extract_articles_langs(rdf_dir: str) -> tuple[dict, dict]:
    CIR = "https://cir.nii.ac.jp/schema/1.0/"
    DC_NS = "http://purl.org/dc/elements/1.1/"

    rdf_files = list(Path(rdf_dir).glob("*.rdf"))
    print(f"Found {len(rdf_files)} RDF files in {rdf_dir}")

    eng_articles = {}  # English
    jpn_articles = {}  # Japanese (or default)
    empty = 0

    for f in tqdm(rdf_files, desc="Parsing"):
        if os.path.getsize(f) == 0:
            empty += 1
            continue

        g = Graph()
        try:
            g.parse(f, format="application/rdf+xml")
        except Exception as e:
            print("Parse error:", e)
            continue

        # find article subject via dc:title
        subs = list(g.subjects(URIRef(DC_NS + "title"), None))
        if not subs:
            continue
        art = subs[0]
        uri = str(art)

        # Titles
        titles = list(g.objects(art, URIRef(DC_NS + "title")))
        title_en = next((t for t in titles if isinstance(
            t, Literal) and t.language == "en"), None)
        title_jp = next((t for t in titles if isinstance(t, Literal) and (
            t.language is None or t.language == "ja")), None)

        # Abstracts
        CIR_desc = URIRef(CIR + "description")
        descs = list(g.objects(art, CIR_desc))
        abs_en, abs_jp = None, None
        for desc in descs:
            notes = list(g.objects(desc, URIRef(CIR + "notation")))
            for note in notes:
                if isinstance(note, Literal):
                    if note.language == "en":
                        abs_en = remove_html_tags(str(note))
                    elif note.language is None or note.language == "ja":
                        abs_jp = remove_html_tags(str(note))

        # DOI
        pid = g.value(art, URIRef(CIR + "productIdentifier"))
        ident = g.value(pid, URIRef(CIR + "identifier")) if pid else None
        doi = str(ident) if isinstance(
            ident, Literal) and ident.datatype == URIRef(CIR + "DOI") else ""

        # Store per language
        if title_en or abs_en:
            eng_articles[uri] = {
                "uri": uri,
                "title": str(title_en) if title_en else "",
                "doi": doi,
                "abstract": abs_en if abs_en else ""
            }
        if title_jp or abs_jp:
            jpn_articles[uri] = {
                "uri": uri,
                "title": str(title_jp) if title_jp else "",
                "doi": doi,
                "abstract": abs_jp if abs_jp else ""
            }

    print(f"Parsed {len(eng_articles)} English and {
          len(jpn_articles)} Japanese articles; skipped {empty} empty files.")
    return eng_articles, jpn_articles


def parse_and_extract_articles_langs_from_dirs(rdf_dirs: list) -> tuple[dict, dict]:
    eng_articles, jpn_articles = {}, {}
    for rdf_dir in rdf_dirs:
        if not os.path.isdir(rdf_dir):
            raise ValueError(f"Directory {rdf_dir} does not exist or is not a directory.")
        eng, jpn = parse_and_extract_articles_langs(rdf_dir)
        eng_articles.update(eng)
        jpn_articles.update(jpn)
    return eng_articles, jpn_articles
